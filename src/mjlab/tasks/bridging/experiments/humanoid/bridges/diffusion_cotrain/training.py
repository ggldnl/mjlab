"""Alternate PPO tracker updates with tracker-outcome planner updates."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from rsl_rl.env import VecEnv
from rsl_rl.utils import check_nan
from torch import nn

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.dataset.motions import (
  bridge_mask,
  encode,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.planner.bridge import (
  DiffusionBridge,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker.env_cfg import (
  COMMAND,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion_cotrain.command import (
  CoTrainingCommand,
  PlanEpisodes,
  PlanReplay,
)


class TrackerOutcomeCritic(nn.Module):
  """Predict physical tracking quality from a normalized kinematic plan."""

  def __init__(self, features: int, hidden: int = 256) -> None:
    super().__init__()
    self.frames = nn.Sequential(
      nn.Linear(2 * features, hidden),
      nn.SiLU(),
      nn.Linear(hidden, hidden),
      nn.SiLU(),
    )
    self.output = nn.Sequential(
      nn.Linear(hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, 1)
    )

  def forward(
    self, plan: torch.Tensor, known: torch.Tensor, valid: torch.Tensor
  ) -> torch.Tensor:
    if plan.shape != known.shape or valid.shape != plan.shape[:2]:
      raise ValueError("plan, known and valid masks have incompatible shapes")
    frames = self.frames(torch.cat((plan, known.float()), dim=-1))
    mask = valid[..., None]
    pooled = (frames * mask).sum(1) / mask.sum(1).clamp_min(1)
    duration = valid.float().mean(1, keepdim=True)
    return self.output(torch.cat((pooled, duration), dim=-1)).squeeze(-1).sigmoid()


@dataclass(frozen=True)
class PlannerUpdateCfg:
  batch: int = 32
  critic_updates: int = 8
  learning_rate: float = 1e-5
  critic_learning_rate: float = 1e-4
  self_imitation_weight: float = 0.1
  max_grad_norm: float = 1.0


class PlannerTrainer:
  """Use tracker outcomes as a differentiable critic for the planner."""

  def __init__(
    self,
    bridge: DiffusionBridge,
    replay: PlanReplay,
    cfg: PlannerUpdateCfg,
    checkpoint: Path,
    device: str,
  ) -> None:
    if (
      min(
        cfg.batch,
        cfg.critic_updates,
        cfg.learning_rate,
        cfg.critic_learning_rate,
        cfg.max_grad_norm,
      )
      <= 0
      or cfg.self_imitation_weight < 0
    ):
      raise ValueError("planner update settings must be positive")
    self.bridge = bridge
    self.replay = replay
    self.cfg = cfg
    self.device = device
    self.metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
    self.critic = TrackerOutcomeCritic(bridge.layout.width).to(device)
    self.planner_optimizer = torch.optim.AdamW(
      bridge.process.denoiser.parameters(), lr=cfg.learning_rate
    )
    self.critic_optimizer = torch.optim.AdamW(
      self.critic.parameters(), lr=cfg.critic_learning_rate
    )
    if "cotrain_critic_state" in self.metadata:
      self.critic.load_state_dict(self.metadata["cotrain_critic_state"])
    if "cotrain_planner_optimizer" in self.metadata:
      self.planner_optimizer.load_state_dict(self.metadata["cotrain_planner_optimizer"])
    if "cotrain_critic_optimizer" in self.metadata:
      self.critic_optimizer.load_state_dict(self.metadata["cotrain_critic_optimizer"])
    self._freeze_planner()

  def _freeze_planner(self) -> None:
    self.bridge.process.eval().requires_grad_(False)

  def _batch_tensors(
    self, batch: PlanEpisodes
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    full_states = torch.cat((batch.history[:, :-1], batch.path), dim=1)
    clean = self.bridge.normalizer.normalize(
      encode(full_states, batch.history[:, -1], self.bridge.fps)
    )
    known = bridge_mask(
      clean.shape[0],
      clean.shape[1],
      self.bridge.layout,
      self.bridge.history,
      self.bridge.future,
      batch.duration,
    )
    rows = self.bridge.history - 1 + batch.duration
    time_index = torch.arange(clean.shape[1], device=clean.device)[None]
    valid = time_index <= rows[:, None] + self.bridge.future - 1
    return clean, known, valid

  def _sample_last_step_grad(
    self, known_values: torch.Tensor, known: torch.Tensor
  ) -> torch.Tensor:
    process = self.bridge.process
    ladder = (
      torch.linspace(
        process.cfg.steps - 1,
        0,
        process.cfg.sample_steps,
        device=known_values.device,
      )
      .round()
      .long()
      .unique_consecutive()
    )
    noisy = torch.where(known, known_values, torch.randn_like(known_values))
    with torch.no_grad():
      for index, tick in enumerate(ladder[:-1]):
        step = tick.expand(noisy.shape[0])
        clean = torch.where(known, known_values, process.denoiser(noisy, known, step))
        alpha = process.schedule[tick]
        next_alpha = process.schedule[ladder[index + 1]]
        noise = (noisy - alpha.sqrt() * clean) / (1 - alpha).sqrt()
        noisy = torch.where(
          known,
          known_values,
          next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * noise,
        )
    final_step = ladder[-1].expand(noisy.shape[0])
    return torch.where(
      known, known_values, process.denoiser(noisy.detach(), known, final_step)
    )

  def update(self, planner_updates: int) -> dict[str, float]:
    if len(self.replay) < self.cfg.batch:
      return {"planner/skipped": 1.0}
    critic_loss = 0.0
    self.critic.train().requires_grad_(True)
    for _ in range(self.cfg.critic_updates):
      batch = self.replay.sample(self.cfg.batch, self.device)
      clean, known, valid = self._batch_tensors(batch)
      loss = (self.critic(clean, known, valid) - batch.outcome).square().mean()
      self.critic_optimizer.zero_grad(set_to_none=True)
      loss.backward()
      self.critic_optimizer.step()
      critic_loss += loss.item()

    self.critic.eval().requires_grad_(False)
    self.bridge.process.train().requires_grad_(True)
    planner_loss = 0.0
    predicted_score = 0.0
    for _ in range(planner_updates):
      batch = self.replay.sample(self.cfg.batch, self.device)
      replay_clean, known, valid = self._batch_tensors(batch)
      generated = self._sample_last_step_grad(
        torch.where(known, replay_clean, torch.zeros_like(replay_clean)), known
      )
      score = self.critic(generated, known, valid).mean()
      best = batch.outcome >= batch.outcome.median()
      imitation = self.bridge.process.loss(
        replay_clean[best],
        known[best],
        valid[best],
        normalizer=self.bridge.normalizer,
        layout=self.bridge.layout,
        fps=self.bridge.fps,
      )
      loss = -score + self.cfg.self_imitation_weight * imitation
      self.planner_optimizer.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(
        self.bridge.process.denoiser.parameters(), self.cfg.max_grad_norm
      )
      self.planner_optimizer.step()
      planner_loss += loss.item()
      predicted_score += score.item()
    self._freeze_planner()
    return {
      "planner/loss": planner_loss / planner_updates,
      "planner/predicted_outcome": predicted_score / planner_updates,
      "planner/critic_loss": critic_loss / self.cfg.critic_updates,
      "planner/replay_size": float(len(self.replay)),
    }

  def save(self, path: Path, iteration: int) -> None:
    saved = dict(self.metadata)
    saved["ema"] = {
      name: value.detach().cpu()
      for name, value in self.bridge.process.denoiser.state_dict().items()
    }
    saved["iteration"] = iteration
    saved["cotrain_critic_state"] = {
      name: value.detach().cpu() for name, value in self.critic.state_dict().items()
    }
    saved["cotrain_planner_optimizer"] = self.planner_optimizer.state_dict()
    saved["cotrain_critic_optimizer"] = self.critic_optimizer.state_dict()
    torch.save(saved, path)


class AlternatingRunner(MjlabOnPolicyRunner):
  """Freeze the planner during PPO and freeze PPO during planner updates."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    tracker_checkpoint = Path(train_cfg.pop("tracker_checkpoint", ""))
    self.alternate_every = int(train_cfg.pop("alternate_every", 25))
    self.planner_updates = int(train_cfg.pop("planner_updates", 4))
    update_cfg = PlannerUpdateCfg(
      batch=int(train_cfg.pop("planner_batch", 32)),
      critic_updates=int(train_cfg.pop("critic_updates", 8)),
      learning_rate=float(train_cfg.pop("planner_learning_rate", 1e-5)),
      critic_learning_rate=float(train_cfg.pop("critic_learning_rate", 1e-4)),
      self_imitation_weight=float(train_cfg.pop("self_imitation_weight", 0.1)),
    )
    if self.alternate_every < 1 or self.planner_updates < 1:
      raise ValueError("alternation settings must be positive")
    super().__init__(env, train_cfg, log_dir, device)
    if self.is_distributed:
      raise ValueError("alternating co-training currently supports one GPU")
    command = self.env.unwrapped.command_manager.get_term(COMMAND)
    if not isinstance(command, CoTrainingCommand):
      raise TypeError("AlternatingRunner requires CoTrainingCommand")
    self.command = command
    self.planner_trainer = PlannerTrainer(
      command.planner,
      command.replay,
      update_cfg,
      command.cfg.planner_checkpoint,
      device,
    )
    self._pretrained_tracker_loaded = False
    if tracker_checkpoint.is_file():
      super().load(
        str(tracker_checkpoint),
        load_cfg={"actor": True, "critic": True, "optimizer": True},
        map_location=device,
      )
      self._pretrained_tracker_loaded = True
      print(f"[cotrain] tracker {tracker_checkpoint}")

  def save(self, path: str, infos=None) -> None:
    super().save(path, infos)
    planner_path = Path(path).with_name(
      Path(path).name.replace("model_", "planner_", 1)
    )
    self.planner_trainer.save(planner_path, self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    self._pretrained_tracker_loaded = True
    return infos

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    if not self._pretrained_tracker_loaded:
      raise ValueError(
        "Alternating co-training requires a compatible --agent.tracker-checkpoint"
      )
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )
    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()
    self.logger.init_logging_writer()
    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      started = time.time()
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)
          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )
          self.alg.process_env_step(obs, rewards, dones, extras)
          intrinsic = (
            self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
          )
          self.logger.process_env_step(rewards, dones, extras, intrinsic)
        collect_time = time.time() - started
        self.alg.compute_returns(obs)
      update_started = time.time()
      loss_dict = self.alg.update()
      if (it + 1) % self.alternate_every == 0:
        loss_dict.update(self.planner_trainer.update(self.planner_updates))
      learn_time = time.time() - update_started
      self.current_learning_iteration = it
      loss_dict["curriculum/cross_trajectory_fraction"] = (
        self.command.cross_trajectory_fraction
      )
      rnd_weight = None
      if self.cfg["algorithm"]["rnd_cfg"]:
        assert self.alg.rnd is not None
        rnd_weight = self.alg.rnd.weight
      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=rnd_weight,
      )
      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        assert self.logger.log_dir is not None
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
    if self.logger.writer is not None:
      assert self.logger.log_dir is not None
      self.save(
        os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
      )
      self.logger.stop_logging_writer()
