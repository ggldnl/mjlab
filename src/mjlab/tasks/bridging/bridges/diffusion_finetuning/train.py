"""Fine tune the diffusion planner through a learned model of the frozen tracker.

Extends diffusion.train, the planner improvement task. That task executes planner
outputs with the frozen tracker and replays the successful executions as extra
demonstrations. Failed plans are dropped, so the planner never learns which of its
frames the tracker could not follow.

This task keeps that loop and adds a learning signal from every executed plan. A
feasibility critic, see critic.py, learns per frame tracking scores from the
rollouts. The planner is then fine tuned with its usual loss plus a feasibility
loss: the critic scores the planner's own output and the gradient flows back
through the critic into the denoiser. The critic is retrained every cycle on
rollouts of the current planner, so it is corrected where the planner learns to
exploit it.

That correction alone is too slow. In a test with the feasibility loss alone, one
cycle of 200 updates broke the planner: the gate pass rate went from 40% to 0%,
foot placement errors grew twentyfold, while the demonstration loss did not move.
The feasibility gradient only reaches the last denoising rung, which the
demonstration loss, spread over all noise levels, hardly constrains. An anchor
pulls the denoiser on that rung toward a frozen copy of the planner as the run
started, as the KL term does in RLHF.

The planner output for the feasibility loss comes from the deployed sampler. The
denoising ladder runs without gradients except for the last rungs, as in DRaFT
(Clark et al., 2023). One rung is enough for the gradient to reach the denoiser.
Before the critic has seen enough executed plans, cycles only collect rollouts and
train the critic, and the planner is left untouched.

One cycle:

    collect     generate, gate and execute plans, as in diffusion.train
    label       per frame tracking scores of every executed plan, see critic.py
    critic      train on the critic replay
    planner     demonstration loss, physical replay share, feasibility loss, anchor

Checkpoints keep the diffusion.train format, so every consumer of a planner
improvement checkpoint reads them. The critic is saved alongside.

Run

1. Train a universal tracker and pretrain a planner, see the diffusion README.

2. Fine tune.

    uv run train Mjlab-G1-Diffusion-Planner-Finetune --agent.max-iterations 30 --agent.tracker-checkpoint logs/rsl_rl/g1_diffusion_universal_tracker/<run>/model_5999.pt --agent.planner-checkpoint logs/rsl_rl/g1_diffusion_kinematic_planner/<run>/model_30000.pt

3. Run walk to kick with the result.

    uv run python -m mjlab.tasks.bridging.config.g1.tests.transitions.walk2kick --bridge diffusion --bridge-checkpoint logs/rsl_rl/g1_diffusion_planner_finetune/<run>/model_30.pt

Use Mjlab-T1-Diffusion-Planner-Finetune with T1 checkpoints for the T1 pipeline.
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import copy
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields

import torch
from rsl_rl.env import VecEnv

from mjlab.tasks.bridging.bridges.diffusion.dataset.motions import (
  bridge_mask,
  encode_pose,
  integrate,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.bridge import (
  DiffusionBridge,
  GeneratedPath,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.model import Denoiser
from mjlab.tasks.bridging.bridges.diffusion.planner.process import Diffusion
from mjlab.tasks.bridging.bridges.diffusion.train import (
  EvaluatedPlans,
  PlannerImprovementRunner,
  PlannerImprovementRunnerCfg,
  training_env_cfg,
  training_runner_cfg,
)
from mjlab.tasks.bridging.bridges.diffusion_finetuning.critic import (
  CriticCfg,
  CriticTrainer,
  FeasibilityCritic,
  frame_labels,
)
from mjlab.tasks.bridging.config import ROBOTS, get_robot
from mjlab.tasks.registry import register_mjlab_task


def finetune_task_id(robot: str) -> str:
  get_robot(robot)
  return f"Mjlab-{robot.upper()}-Diffusion-Planner-Finetune"


def finetune_experiment(robot: str) -> str:
  get_robot(robot)
  return f"{robot}_diffusion_planner_finetune"


TRAIN_TASK_ID = finetune_task_id("g1")
TRAIN_EXPERIMENT = finetune_experiment("g1")


@dataclass(frozen=True)
class FinetuneCfg:
  feasibility_weight: float = 0.1
  anchor_weight: float = 1.0
  """Pull toward the planner as it was when the run started, on the gradient rungs"""
  feasibility_batch: int = 64
  feasibility_every: int = 4
  """Planner updates between feasibility losses, the sampler is the costly part"""
  gradient_rungs: int = 1
  """Last denoising rungs that keep gradients"""

  def __post_init__(self) -> None:
    if min(self.feasibility_weight, self.anchor_weight) < 0:
      raise ValueError("feasibility_weight and anchor_weight cannot be negative")
    if min(self.feasibility_batch, self.feasibility_every, self.gradient_rungs) < 1:
      raise ValueError("Feasibility batch, interval and rungs must be positive")


class RecordingBridge(DiffusionBridge):
  """DiffusionBridge that keeps its last generated batch at full precision.

  The improvement runner stores plans in half precision, which rounds world root
  positions to centimeters. The critic needs them exact.
  """

  last: tuple[torch.Tensor, GeneratedPath] | None = None

  @classmethod
  def wrap(cls, bridge: DiffusionBridge) -> RecordingBridge:
    wrapped = cls(
      bridge.process,
      bridge.normalizer,
      bridge.layout,
      bridge.history,
      bridge.future,
      bridge.fps,
      bridge.min_steps,
      bridge.robot,
      bridge.floor_height,
      bridge.contact_height,
      bridge.contact_speed,
    )
    wrapped._feet = bridge._feet
    return wrapped

  @torch.no_grad()
  def generate(
    self,
    history: torch.Tensor,
    target: torch.Tensor,
    duration: torch.Tensor,
    trace: list[GeneratedPath] | None = None,
  ) -> GeneratedPath:
    path = super().generate(history, target, duration, trace)
    self.last = (history, path)
    return path

  def take(self) -> tuple[torch.Tensor, GeneratedPath]:
    """The last generated batch, cleared after reading."""
    if self.last is None:
      raise RuntimeError("Planner evaluation did not go through the bridge")
    last, self.last = self.last, None
    return last

  def conditions(
    self, history: torch.Tensor, target: torch.Tensor, duration: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized known values and mask, as denoise builds them."""
    batch = history.shape[0]
    columns = self.process.denoiser.columns
    mask = bridge_mask(batch, columns, self.layout, self.history, self.future, duration)
    anchor = history[:, -1]
    values = history.new_zeros((batch, columns, self.layout.width))
    values[:, : self.history] = self.features(encode_pose(history, anchor))
    rows = self.history - 1 + duration
    offsets = torch.arange(self.future, device=history.device)
    indexes = torch.arange(batch, device=history.device)[:, None]
    values[indexes, rows[:, None] + offsets] = self.features(
      encode_pose(target, anchor)
    )
    return self.normalizer.normalize(values), mask


def sample_with_gradient(
  process: Diffusion,
  known_values: torch.Tensor,
  known: torch.Tensor,
  gradient_rungs: int,
  project: Callable[[torch.Tensor], torch.Tensor] | None,
  reference: Denoiser | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Diffusion.sample with gradients kept on the last rungs only.

  Also returns the drift: mean squared gap between the denoiser and the frozen
  reference on the free channels of the gradient rungs, zero without a reference.
  """
  ladder = (
    torch.linspace(
      process.cfg.steps - 1, 0, process.cfg.sample_steps, device=known_values.device
    )
    .round()
    .long()
    .unique_consecutive()
  )
  free = ~known & process.predicted
  drift = known_values.new_zeros(())
  noisy = process._inputs(torch.randn_like(known_values), known_values, known)
  for index, tick in enumerate(ladder):
    last = index == len(ladder) - 1
    tracked = index >= len(ladder) - gradient_rungs
    with torch.set_grad_enabled(tracked):
      step = tick.expand(noisy.shape[0])
      predicted = process.denoiser(noisy, known, step)
      if tracked and reference is not None:
        with torch.no_grad():
          anchored = reference(noisy, known, step)
        drift = drift + (predicted - anchored).square()[free].mean() / gradient_rungs
      clean = process._inputs(predicted, known_values, known)
      alpha = process.schedule[tick]
      noise = (noisy - alpha.sqrt() * clean) / (1 - alpha).sqrt()
      if project is not None:
        clean = process._inputs(project(clean), known_values, known)
      if last:
        return clean, drift
      next_alpha = process.schedule[ladder[index + 1]]
      noisy = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * noise
      noisy = process._inputs(noisy, known_values, known)
  raise RuntimeError("empty denoising schedule")


@dataclass
class FinetuneRunnerCfg(PlannerImprovementRunnerCfg):
  feasibility_critic: CriticCfg = field(default_factory=CriticCfg)
  finetune: FinetuneCfg = field(default_factory=FinetuneCfg)


class FinetuneRunner(PlannerImprovementRunner):
  """Planner improvement plus a feasibility loss from a learned tracker critic."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    critic_cfg = CriticCfg(**train_cfg.pop("feasibility_critic"))
    self.finetune_cfg = FinetuneCfg(**train_cfg.pop("finetune"))
    super().__init__(env, train_cfg, log_dir, device)
    self.bridge = RecordingBridge.wrap(self.planner.bridge)
    self.planner.bridge = self.bridge
    bridge = self.bridge
    critic = FeasibilityCritic(
      bridge.layout,
      bridge.normalizer,
      bridge.history,
      bridge.future,
      bridge.process.denoiser.columns,
      critic_cfg,
    )
    self.critic = CriticTrainer(critic, critic_cfg, device)
    self.reference: Denoiser | None = None

  def _evaluate_planner(self) -> tuple[EvaluatedPlans, dict[str, float]]:
    self.bridge.last = None
    evaluated, metrics = super()._evaluate_planner()
    history, path = self.bridge.take()
    window = torch.cat((history[:, :-1], path.states), dim=1)
    pose = encode_pose(window, history[:, -1])
    passed = evaluated.hard_pass.to(self.device)
    executed = passed.nonzero().flatten()
    rejected = (~passed).nonzero().flatten()
    metrics["critic/executed"] = float(executed.numel())
    metrics["critic/rejected"] = float(rejected.numel())
    columns = self.bridge.process.denoiser.columns
    if executed.numel():
      labels = frame_labels(
        evaluated.tracking_errors[executed.cpu()].to(self.device).float(),
        evaluated.fallen[executed.cpu()].to(self.device),
        self.command.tolerances * self.command.cfg.tracking_tolerance_scale,
        self.bridge.history,
        columns,
      )
      duration = path.duration[executed]
      if self.critic.ready:
        metrics.update(self.critic.check(pose[executed], duration, labels))
      self.critic.replay.add(pose[executed], duration, labels)
    # Without these the critic never sees the plans the gate stops, and the planner
    # drifts into them. The gate does not say which frame failed, so all are 0
    if rejected.numel():
      zeros = pose.new_zeros(rejected.numel(), columns)
      self.critic.replay.add(pose[rejected], path.duration[rejected], zeros)
    return evaluated, metrics

  def _feasibility_loss(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predicted infeasibility of freshly sampled plans, their drift, their score."""
    bridge = self.bridge
    pair = self.pairs.draw(self.finetune_cfg.feasibility_batch)
    target = pair.target[:, : bridge.future]
    values, mask = bridge.conditions(pair.history, target, pair.duration)
    clean, drift = sample_with_gradient(
      bridge.process,
      values,
      mask,
      self.finetune_cfg.gradient_rungs,
      bridge.floor(pair.duration),
      self.reference,
    )
    features = bridge.normalizer.denormalize(clean)
    pose = integrate(features, bridge.layout, bridge.history, pair.duration)
    feasible = torch.sigmoid(self.critic.critic(pose, pair.duration))
    frames = self.critic.scored(pair.duration)
    infeasible = (1 - feasible)[frames].mean()
    return infeasible, drift, feasible[frames].mean().detach()

  def _finetune(self, updates: int) -> dict[str, float]:
    planner = self.planner
    cfg = planner.cfg
    process = planner.bridge.process
    process.train().requires_grad_(True)
    physical_weight = cfg.physical_fraction if len(self.replay) else 0.0
    total = demo_total = feasibility_total = drift_total = predicted_total = 0.0
    feasibility_count = 0
    for update in range(updates):
      features, duration = planner.windows.sample(cfg.batch)
      demo = planner._loss(features, duration)
      loss = demo
      if physical_weight:
        states, physical_duration = self.replay.sample(cfg.batch, self.device)
        physical = planner._loss(
          planner.bridge.window_features(states, physical_duration), physical_duration
        )
        loss = (1 - physical_weight) * demo + physical_weight * physical
      if update % self.finetune_cfg.feasibility_every == 0:
        infeasible, drift, predicted = self._feasibility_loss()
        loss = loss + self.finetune_cfg.feasibility_weight * infeasible
        loss = loss + self.finetune_cfg.anchor_weight * drift
        feasibility_total += infeasible.item()
        drift_total += drift.item()
        predicted_total += predicted.item()
        feasibility_count += 1
      planner.optimizer.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(process.denoiser.parameters(), cfg.max_grad_norm)
      planner.optimizer.step()
      total += loss.item()
      demo_total += demo.item()
    planner.updates += updates
    process.eval().requires_grad_(False)
    count = max(feasibility_count, 1)
    return {
      "planner/loss": total / updates,
      "planner/real_loss": demo_total / updates,
      "planner/physical_fraction": physical_weight,
      "planner/updates": float(planner.updates),
      "finetune/infeasibility": feasibility_total / count,
      "finetune/drift": drift_total / count,
      "finetune/predicted_score": predicted_total / count,
    }

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    del init_at_random_ep_len
    if not self.tracker_loaded:
      raise ValueError(
        "Planner fine tuning requires --agent.tracker-checkpoint with a pretrained tracker"
      )
    self.logger.init_logging_writer()
    evaluator_metrics = self._validate_evaluator()
    start = self.current_learning_iteration
    if start == 0 and not self.planner.warm_started:
      print(
        f"[finetune] planner pretrain: {self.planner_cfg.bootstrap_updates} updates"
      )
      self.planner.train(self.planner_cfg.bootstrap_updates, self.replay)
    # The anchor is the planner as this run starts, after pretraining or resuming
    self.reference = copy.deepcopy(self.planner.bridge.process.denoiser)
    self.reference.eval().requires_grad_(False)
    for cycle in range(start, num_learning_iterations):
      collect_started = time.time()
      self.evaluated, metrics = self._evaluate_planner()
      collect_time = time.time() - collect_started
      learn_started = time.time()
      metrics.update(self.critic.train(self.critic.cfg.updates_per_cycle))
      if self.critic.ready:
        metrics.update(self._finetune(self.planner_cfg.updates_per_cycle))
      metrics["finetune/active"] = float(self.critic.ready)
      metrics.update(evaluator_metrics)
      learn_time = time.time() - learn_started
      self.current_learning_iteration = cycle + 1
      self.logger.log(
        it=cycle,
        start_it=start,
        total_it=num_learning_iterations,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=metrics,
        learning_rate=self.planner_cfg.learning_rate,
        # The frozen tracker acts on its mean and never samples, so it has no std
        action_std=torch.full((1,), torch.nan),
        rnd_weight=None,
      )
      if (
        self.logger.writer is not None and (cycle + 1) % self.cfg["save_interval"] == 0
      ):
        assert self.logger.log_dir is not None
        self.save(os.path.join(self.logger.log_dir, f"model_{cycle + 1}.pt"))
    if self.logger.writer is not None:
      assert self.logger.log_dir is not None
      self.save(
        os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
      )
      self.logger.stop_logging_writer()

  def save(self, path: str, infos=None) -> None:
    super().save(
      path, {**(infos or {}), "feasibility_critic": self.critic.state_dict()}
    )

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    if load_cfg is None and infos and "feasibility_critic" in infos:
      self.critic.load_state_dict(infos["feasibility_critic"])
    return infos


def finetune_runner_cfg(robot: str = "g1") -> FinetuneRunnerCfg:
  base = training_runner_cfg(robot)
  values = {item.name: getattr(base, item.name) for item in fields(base) if item.init}
  values["experiment_name"] = finetune_experiment(robot)
  return FinetuneRunnerCfg(**values)


for _robot in ROBOTS:
  register_mjlab_task(
    task_id=finetune_task_id(_robot),
    env_cfg=training_env_cfg(robot=_robot),
    play_env_cfg=training_env_cfg(play=True, robot=_robot),
    rl_cfg=finetune_runner_cfg(_robot),
    runner_cls=FinetuneRunner,
  )

__all__ = ["TRAIN_EXPERIMENT", "TRAIN_TASK_ID"]
