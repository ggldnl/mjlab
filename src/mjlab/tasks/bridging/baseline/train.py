"""Train with uv run python -m mjlab.tasks.bridging.baseline.train --help."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.tasks.bridging.baseline.learning import (
  Replay,
  UpstreamAdam,
  fit_proximity,
  update_ppo,
)
from mjlab.tasks.bridging.baseline.model import (
  UPSTREAM_COMMIT,
  TransitionPolicy,
  fingerprint,
)
from mjlab.tasks.bridging.baseline.outcome import KICK_MIN_SPEED_M_S
from mjlab.tasks.bridging.baseline.rollout import collect
from mjlab.tasks.bridging.baseline.scene import SceneCfg, TransitionScene


@dataclass(frozen=True)
class LearningCfg:
  scene: SceneCfg = field(default_factory=SceneCfg)
  entries: tuple[int, ...] | None = None
  distance: float = 0.4
  max_transition_steps: int = 100
  num_rollouts: int = 10000
  num_trans_batches: int = 256
  bootstrap_batches: int = 8
  initiation_start: float = 0.0
  initiation_end: float = 1.0
  proximity_epochs: int = 5
  proximity_decay: float = 0.95
  proximity_learning_rate: float = 0.0001
  replay_size: int = 1_000_000
  learning_rate: float = 0.0001
  lr_decay: bool = True
  ppo_epochs: int = 10
  minibatch_size: int = 64
  clip_param: float = 0.2
  entropy_coef: float = 0.001

  def __post_init__(self) -> None:
    if self.entries is not None and (
      not self.entries
      or len(set(self.entries)) != len(self.entries)
      or min(self.entries) < 0
    ):
      raise ValueError("entries must be nonempty, nonnegative and unique")
    for name in (
      "max_transition_steps",
      "num_rollouts",
      "num_trans_batches",
      "bootstrap_batches",
      "proximity_epochs",
      "ppo_epochs",
      "minibatch_size",
      "replay_size",
    ):
      if getattr(self, name) < 1:
        raise ValueError(f"{name} must be positive")
    if self.minibatch_size < 2:
      raise ValueError(
        "minibatch_size must be at least two for collected/online replay sampling"
      )
    if not 0 <= self.initiation_start < self.initiation_end <= 1:
      raise ValueError(
        "Initiation trajectory portion must satisfy 0 <= start < end <= 1"
      )
    if not 0 < self.proximity_decay <= 1:
      raise ValueError("proximity_decay must be in (0, 1]")
    if not math.isfinite(self.distance) or self.distance <= 0:
      raise ValueError("distance must be finite and positive")
    for name in ("learning_rate", "proximity_learning_rate", "clip_param"):
      if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(self.entropy_coef) or self.entropy_coef < 0:
      raise ValueError("entropy_coef must be finite and nonnegative")


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


def train(
  cfg: TrainCfg,
  scene: TransitionScene | None = None,
  on_update: Callable[[dict, TransitionPolicy], None] | None = None,
  on_save: Callable[[TransitionPolicy, dict], None] | None = None,
) -> Path:
  torch.manual_seed(cfg.seed)
  owns_scene = scene is None
  output = cfg.output or Path(
    f"logs/baseline/{cfg.scene.pair}/seed_{cfg.seed}/model.pt"
  )
  if any(
    path.exists()
    for path in (
      output,
      output.parent / "config.json",
      output.parent / "training.jsonl",
    )
  ):
    raise FileExistsError(f"Refusing to replace an existing run: {output}")
  scene = scene or TransitionScene(cfg.scene, cfg.num_envs)
  try:
    entries = cfg.entries if cfg.entries is not None else (cfg.scene.entry,)
    if any(index >= len(scene.run.entries) for index in entries):
      raise ValueError("Training entries exceed the selector table")
    positive, negative = Replay(cfg.replay_size), Replay(cfg.replay_size)
    bootstrap_batches = max(cfg.bootstrap_batches, len(entries))
    for index in range(bootstrap_batches):
      scene.select_entry(entries[index % len(entries)])
      scene.reset(cfg.seed + index)
      scene.nominal()
      _, success, _ = scene.successor(collect=True)
      if bool(success.any()):
        trace = torch.stack(scene.successor_observations)
        start, end = (
          int(len(trace) * cfg.initiation_start),
          int(len(trace) * cfg.initiation_end),
        )
        if end <= start:
          raise ValueError("Initiation trajectory portion contains no control frames")
        states = trace[start:end, success].flatten(0, 1)
        positive.add(
          states, torch.ones(len(states), device=states.device), collected=True
        )
      print(
        f"Bootstrap {index + 1}/{bootstrap_batches}: success={float(success.float().mean()):.3f}",
        flush=True,
      )
    if not positive.collected:
      raise RuntimeError(
        "No successful frozen skill demonstrations for the initiation buffer"
      )
    observation_dim = positive.collected[0].shape[1] - 1
    model = TransitionPolicy(
      observation_dim,
      scene.wrapped.num_actions,
      num_primitives=2,
      primitive_index=1,
    ).to(cfg.scene.device)
    actor_optimizer = UpstreamAdam(model.policy_parameters(), lr=cfg.learning_rate)
    predictor_optimizer = UpstreamAdam(
      model.proximity.parameters(), lr=cfg.proximity_learning_rate
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(json.dumps(asdict(cfg), default=str))
    config["bootstrap_batches"], config["entries"] = bootstrap_batches, list(entries)
    config["success_criterion"] = (
      {
        "foot_ball_contact_after_handoff": True,
        "min_forward_ball_speed_m_s_exclusive": KICK_MIN_SPEED_M_S,
        "full_remaining_motion": True,
        "no_native_failure": True,
      }
      if scene.target_skill == "kick"
      else {"no_native_failure": True}
    )
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
    episode = 0
    with (output.parent / "training.jsonl").open("x", encoding="utf-8") as log:
      for iteration in range(cfg.iterations):
        started = time.perf_counter()
        first_step = scene.env.common_step_counter
        chunks, records = [], []
        entry = entries[0]
        while (
          scene.env.common_step_counter - first_step
        ) * cfg.num_envs < cfg.num_rollouts or not chunks:
          # Use a separate generator because source resets seed the simulator RNG
          generator = torch.Generator().manual_seed(
            cfg.seed + bootstrap_batches + episode
          )
          entry = entries[int(torch.randint(len(entries), (), generator=generator))]
          scene.select_entry(entry)
          scene.reset(cfg.seed + bootstrap_batches + episode)
          scene.interrupt(cfg.distance)
          batch, record = collect(
            scene, model, cfg.max_transition_steps, cfg.proximity_decay
          )
          chunks.append(batch)
          record.pop("success")
          records.append(record)
          episode += 1
        batch = {
          name: torch.cat([chunk[name] for chunk in chunks]) for name in chunks[0]
        }
        collected = time.perf_counter()
        targets = batch["targets"]
        good = targets > 0
        positive.add(batch["observations"][good], targets[good])
        negative.add(batch["observations"][~good], targets[~good])
        model.proximity_rms.update(batch["observations"])
        predictor_loss = fit_proximity(
          model,
          predictor_optimizer,
          positive,
          negative,
          cfg.proximity_epochs * (cfg.num_rollouts // cfg.minibatch_size + 1),
          cfg.minibatch_size,
        )
        multiplier = max(1 - iteration / cfg.iterations, 0.0) if cfg.lr_decay else 1.0
        ppo_loss = update_ppo(model, actor_optimizer, batch, cfg, multiplier)
        record: dict = {
          name: sum(item[name] for item in records) / len(records)
          for name in (
            "success_rate",
            "reward_mean",
            "duration_s",
            "action_jump_rad",
            "transition_steps",
          )
        }
        record.update(
          {
            "iteration": iteration + 1,
            "entry": entry,
            "distance": cfg.distance,
            "ppo_loss": ppo_loss,
            "proximity_loss": predictor_loss,
            "physics_steps": scene.env.common_step_counter - first_step,
            "collect_s": collected - started,
            "learn_s": time.perf_counter() - collected,
            "target_errors": torch.tensor([item["target_errors"] for item in records])
            .mean(0)
            .tolist(),
            "downstream_rms": torch.tensor([item["downstream_rms"] for item in records])
            .mean(0)
            .tolist(),
          }
        )
        log.write(json.dumps(record, allow_nan=False) + "\n")
        log.flush()
        if on_update is not None:
          on_update(record, model)
        print(json.dumps(record), flush=True)
        if (iteration + 1) % cfg.save_interval == 0 or iteration + 1 == cfg.iterations:
          metadata = {
            "config": config,
            "iteration": iteration + 1,
            "channels": list(scene.channels),
            "fingerprints": fingerprints,
            "upstream_commit": UPSTREAM_COMMIT,
            "training_seed_end": cfg.seed + bootstrap_batches + episode,
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
