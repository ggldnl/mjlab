"""Train with uv run python -m mjlab.tasks.bridging.baseline.train --help."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import json
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.tasks.bridging.baseline.model import (
  TransitionPolicy,
  fingerprint,
  proximity_targets,
  returns_to_go,
)
from mjlab.tasks.bridging.baseline.scene import SceneCfg, TransitionScene


@dataclass(frozen=True)
class LearningCfg:
  scene: SceneCfg = field(default_factory=SceneCfg)
  entries: tuple[int, ...] | None = None
  distances: tuple[float, ...] = (0.2, 0.4, 0.6)
  durations: tuple[float, ...] = (0.4, 0.8, 1.2)
  bootstrap_batches: int = 8
  proximity_updates: int = 32
  proximity_decay: float = 0.95
  learning_rate: float = 0.0003
  ppo_epochs: int = 4
  minibatch_size: int = 1024
  success_bonus: float = 1.0
  entropy_coef: float = 0.001
  warm_start_updates: int = 0

  def __post_init__(self) -> None:
    if self.entries is not None and (
      not self.entries
      or len(set(self.entries)) != len(self.entries)
      or min(self.entries) < 0
    ):
      raise ValueError("entries must be nonempty, nonnegative and unique")
    for name in (
      "proximity_updates",
      "ppo_epochs",
      "minibatch_size",
    ):
      if getattr(self, name) < 1:
        raise ValueError(f"{name} must be positive")
    if self.bootstrap_batches < 2:
      raise ValueError("bootstrap_batches must be at least two")
    if self.warm_start_updates < 0:
      raise ValueError("warm_start_updates cannot be negative")
    if not 0 < self.proximity_decay <= 1:
      raise ValueError("proximity_decay must be in (0, 1]")
    for values in (self.distances, self.durations):
      if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("distances and durations must be finite and positive")
    if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
      raise ValueError("learning_rate must be finite and positive")
    if any(
      not math.isfinite(value) or value < 0
      for value in (self.success_bonus, self.entropy_coef)
    ):
      raise ValueError("Reward coefficients must be finite and nonnegative")


@dataclass(frozen=True)
class TrainCfg(LearningCfg):
  num_envs: int = 64
  iterations: int = 1000
  seed: int = 0
  save_interval: int = 25
  output: Path | None = None

  def __post_init__(self) -> None:
    super().__post_init__()
    for name in ("num_envs", "iterations", "save_interval"):
      if getattr(self, name) < 1:
        raise ValueError(f"{name} must be positive")


def update_ppo(
  model: TransitionPolicy,
  optimizer: torch.optim.Optimizer,
  observations: torch.Tensor,
  actions: torch.Tensor,
  old_log_prob: torch.Tensor,
  returns: torch.Tensor,
  cfg: TrainCfg,
) -> float:
  with torch.no_grad():
    advantages = returns - model.value(observations)
    advantages = (advantages - advantages.mean()) / advantages.std(
      unbiased=False
    ).clamp_min(1e-6)
  loss = observations.new_zeros(())
  for _ in range(cfg.ppo_epochs):
    for ids in torch.randperm(len(observations), device=observations.device).split(
      cfg.minibatch_size
    ):
      distribution = model.distribution(observations[ids])
      ratio = (distribution.log_prob(actions[ids]).sum(-1) - old_log_prob[ids]).exp()
      surrogate = torch.minimum(
        ratio * advantages[ids], ratio.clamp(0.8, 1.2) * advantages[ids]
      )
      loss = (
        -surrogate.mean()
        + 0.5 * (model.value(observations[ids]) - returns[ids]).square().mean()
        - cfg.entropy_coef * distribution.entropy().sum(-1).mean()
      )
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(
        [*model.actor.parameters(), *model.critic.parameters(), model.log_std], 1.0
      )
      optimizer.step()
  return float(loss.detach())


def fit_proximity(
  model: TransitionPolicy,
  optimizer: torch.optim.Optimizer,
  observations: torch.Tensor,
  targets: torch.Tensor,
  updates: int,
  batch_size: int,
) -> float:
  loss = observations.new_zeros(())
  positive = torch.nonzero(targets > 0).flatten()
  negative = torch.nonzero(targets == 0).flatten()
  for _ in range(updates):
    if positive.numel() and negative.numel():
      count = max(1, batch_size // 2)
      ids = torch.cat(
        (
          positive[torch.randint(len(positive), (count,), device=observations.device)],
          negative[torch.randint(len(negative), (count,), device=observations.device)],
        )
      )
    else:
      ids = torch.randint(
        len(observations),
        (min(batch_size, len(observations)),),
        device=observations.device,
      )
    loss = (model.potential(observations[ids]) - targets[ids]).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  return float(loss.detach())


def warm_start_actor(
  model: TransitionPolicy,
  observations: torch.Tensor,
  actions: torch.Tensor,
  updates: int,
  learning_rate: float,
) -> float:
  """Initialize actions from successful executions of the frozen incoming skill."""
  optimizer = torch.optim.Adam(model.actor.parameters(), lr=learning_rate)
  loss = observations.new_zeros(())
  for _ in range(updates):
    ids = torch.randint(
      len(observations), (min(1024, len(observations)),), device=observations.device
    )
    loss = (model(observations[ids]) - actions[ids]).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  return float(loss.detach())


def train(
  cfg: TrainCfg,
  scene: TransitionScene | None = None,
  on_update: Callable[[dict, TransitionPolicy], None] | None = None,
  on_save: Callable[[TransitionPolicy, dict], None] | None = None,
) -> Path:
  torch.manual_seed(cfg.seed)
  owns_scene = scene is None
  scene = scene or TransitionScene(cfg.scene, cfg.num_envs)
  output = cfg.output or Path(
    f"logs/baseline/{cfg.scene.pair}/seed_{cfg.seed}/model.pt"
  )
  if (
    output.exists()
    or (output.parent / "training.jsonl").exists()
    or (output.parent / "config.json").exists()
  ):
    if owns_scene:
      scene.close()
    raise FileExistsError(f"Refusing to replace an existing run: {output}")
  try:
    entries = (
      cfg.entries if cfg.entries is not None else tuple(range(len(scene.run.entries)))
    )
    if any(index >= len(scene.run.entries) for index in entries):
      raise ValueError("Training entries exceed the selector table")
    bootstrap_batches = max(cfg.bootstrap_batches, 2 * len(entries))
    for duration in cfg.durations:
      low, high = scene.run.command.cfg.duration_s_range
      if not low <= duration <= high:
        raise ValueError(f"Duration must be between {low} and {high} seconds")
    replay: deque[tuple[torch.Tensor, torch.Tensor]] = deque(
      maxlen=max(32, bootstrap_batches)
    )
    warm_observations, warm_actions = [], []
    for index in range(bootstrap_batches):
      scene.select_entry(entries[(index // 2) % len(entries)])
      scene.reset(cfg.seed + index)
      if index % 2 == 0:
        scene.nominal(perturb=(0.0, 0.1, 0.3, 0.6)[(index // 2) % 4])
      else:
        scene.interrupt(cfg.distances[index % len(cfg.distances)])
      observation = scene.observe().clone()
      first_action = scene.skill_action(scene.target_skill).clone()
      _, success, _ = scene.successor(collect=bool(cfg.warm_start_updates))
      print(
        f"Bootstrap {index + 1}/{bootstrap_batches}: "
        f"entry={scene.cfg.entry}, success={float(success.float().mean()):.3f}",
        flush=True,
      )
      replay.append((observation, success.float()))
      if bool(success.any()):
        if cfg.warm_start_updates:
          warm_observations.append(
            torch.stack(scene.successor_observations)[:, success].flatten(0, 1)
          )
          warm_actions.append(
            torch.stack(scene.successor_actions)[:, success].flatten(0, 1)
          )
        else:
          warm_observations.append(observation[success])
          warm_actions.append(first_action[success])
    boot_observations = torch.cat([item[0] for item in replay])
    boot_targets = torch.cat([item[1] for item in replay])
    if not bool((boot_targets > 0).any()):
      raise RuntimeError(
        "No successful incoming executions. Check the skill checkpoint and selector entry before training"
      )
    if not bool((boot_targets == 0).any()):
      raise RuntimeError(
        "No failed incoming executions. Increase bootstrap_batches or use more difficult distances to identify the initiation boundary"
      )
    model = TransitionPolicy(boot_observations.shape[-1], scene.wrapped.num_actions).to(
      cfg.scene.device
    )
    model.fit_normalizer(boot_observations)
    if cfg.warm_start_updates:
      model.fit_normalizer(torch.cat(warm_observations))
      warm_loss = warm_start_actor(
        model,
        torch.cat(warm_observations),
        torch.cat(warm_actions),
        cfg.warm_start_updates,
        cfg.learning_rate,
      )
      print(f"Warm start action error: {warm_loss:.6f}", flush=True)
    actor_optimizer = torch.optim.Adam(
      [*model.actor.parameters(), *model.critic.parameters(), model.log_std],
      lr=cfg.learning_rate,
    )
    predictor_optimizer = torch.optim.Adam(
      model.proximity.parameters(), lr=cfg.learning_rate
    )
    fit_proximity(
      model,
      predictor_optimizer,
      boot_observations,
      boot_targets,
      cfg.proximity_updates * 10,
      cfg.minibatch_size,
    )
    # Keep measured entry examples even when recent transition attempts all fail
    anchors = (boot_observations, boot_targets)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(json.dumps(asdict(cfg), default=str))
    config["bootstrap_batches"] = bootstrap_batches
    config["entries"] = list(entries)
    config["scene"]["source_checkpoint"] = str(scene.checkpoints[scene.source])
    config["scene"]["target_checkpoint"] = str(scene.checkpoints[scene.target_skill])
    config["scene"]["selector_path"] = str(cfg.scene.selector_path.resolve())
    fingerprints = {
      name: fingerprint(Path(config["scene"][name]))
      for name in ("source_checkpoint", "target_checkpoint", "selector_path")
    }
    (output.parent / "config.json").write_text(
      json.dumps(config, indent=2), encoding="utf-8"
    )
    with (output.parent / "training.jsonl").open("x", encoding="utf-8") as log:
      for iteration in range(cfg.iterations):
        started = time.perf_counter()
        first_step = scene.env.common_step_counter
        generator = torch.Generator().manual_seed(
          cfg.seed + bootstrap_batches + iteration
        )
        entry_index = entries[
          int(torch.randint(len(entries), (), generator=generator).item())
        ]
        scene.select_entry(entry_index)
        scene.reset(cfg.seed + bootstrap_batches + iteration)
        distance = cfg.distances[int(torch.randint(len(cfg.distances), ()).item())]
        duration = cfg.durations[int(torch.randint(len(cfg.durations), ()).item())]
        scene.interrupt(distance)
        observations, actions, log_probs, potentials = [], [], [], []
        steps = max(1, round(duration / scene.env.step_dt))
        with torch.no_grad():
          for _ in range(steps):
            scene.hold_target()
            observation = scene.observe().clone()
            distribution = model.distribution(observation)
            action = distribution.sample()
            observations.append(observation)
            actions.append(action)
            log_probs.append(distribution.log_prob(action).sum(-1))
            potentials.append(model.potential(observation))
            scene.step(action)
          endpoint = scene.observe().clone()
          potentials.append(model.potential(endpoint))
          endpoint_errors, _ = scene.endpoint()
          downstream_rms, success, action_jump = scene.successor()
          reward = torch.stack(potentials).diff(dim=0)
          reward[-1] += cfg.success_bonus * success.float()
          returns = returns_to_go(reward)
        observations_tensor = torch.stack(observations)
        collected = time.perf_counter()
        ppo_loss = update_ppo(
          model,
          actor_optimizer,
          observations_tensor.flatten(0, 1),
          torch.stack(actions).flatten(0, 1),
          torch.stack(log_probs).flatten(),
          returns.flatten(),
          cfg,
        )
        trace = torch.cat((observations_tensor, endpoint[None]), 0)
        targets = proximity_targets(success, steps + 1, cfg.proximity_decay)
        replay.append((trace.flatten(0, 1), targets.flatten()))
        predictor_loss = fit_proximity(
          model,
          predictor_optimizer,
          torch.cat([anchors[0], *[item[0] for item in replay]]),
          torch.cat([anchors[1], *[item[1] for item in replay]]),
          cfg.proximity_updates,
          cfg.minibatch_size,
        )
        record = {
          "iteration": iteration + 1,
          "entry": entry_index,
          "success_rate": float(success.float().mean()),
          "ppo_loss": ppo_loss,
          "proximity_loss": predictor_loss,
          "distance": distance,
          "duration_s": duration,
          "reward_mean": float(reward.sum(0).mean()),
          "action_jump_rad": float(action_jump.mean()),
          "transition_steps": steps,
          "physics_steps": scene.env.common_step_counter - first_step,
          "collect_s": collected - started,
          "learn_s": time.perf_counter() - collected,
          "target_errors": endpoint_errors.mean(0).tolist(),
          "downstream_rms": downstream_rms.mean(0).tolist(),
        }
        log.write(json.dumps(record) + "\n")
        log.flush()
        if on_update is not None:
          on_update(record, model)
        if iteration % 10 == 0:
          print(json.dumps(record), flush=True)
        if (iteration + 1) % cfg.save_interval == 0 or iteration + 1 == cfg.iterations:
          metadata = {
            "config": config,
            "iteration": iteration + 1,
            "channels": list(scene.channels),
            "fingerprints": fingerprints,
          }
          model.save(output, metadata)
          if on_save is not None:
            on_save(model, metadata)
    return output
  finally:
    if owns_scene:
      scene.close()


def main() -> None:
  print(train(tyro.cli(TrainCfg, config=mjlab.TYRO_FLAGS)))


if __name__ == "__main__":
  main()
