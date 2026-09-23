"""Evaluate position-only bridge paths on held-out physical rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DEFAULT_DATASET,
  load_dataset,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.bridge import (
  DiffusionBridge,
)


@dataclass
class EvaluateCfg:
  checkpoint: Path
  dataset: Path = DEFAULT_DATASET
  count: int = 64
  duration: int = 40
  sample_steps: int = 50
  device: str = "cuda:0"
  seed: int = 0


def evaluate(cfg: EvaluateCfg) -> dict[str, float]:
  torch.manual_seed(cfg.seed)
  bridge = DiffusionBridge.load(cfg.checkpoint, cfg.device, cfg.sample_steps)
  data = load_dataset(cfg.dataset, cfg.device, "eval")
  if data.fps != bridge.fps or data.num_joints != bridge.layout.joints:
    raise ValueError("checkpoint and dataset use different robot joints or rates")
  if cfg.count < 1 or not bridge.min_steps <= cfg.duration <= bridge.max_steps:
    raise ValueError("count or duration is outside the supported range")
  length = bridge.history + cfg.duration + bridge.future - 1
  segments = data.segments(length - 1, length - 1)
  positions = segments.starts[
    torch.randint(segments.starts.numel(), (cfg.count,), device=cfg.device)
  ]
  rows = segments.order[positions[:, None] + torch.arange(length, device=cfg.device)]
  recorded = data.states[rows]
  history = recorded[:, : bridge.history]
  target_start = bridge.history - 1 + cfg.duration
  target = recorded[:, target_start : target_start + bridge.future]
  duration = torch.full((cfg.count,), cfg.duration, device=cfg.device, dtype=torch.long)
  proposed = bridge.generate(history, target, duration).states[
    :, : cfg.duration + bridge.future
  ]
  demonstrated = recorded[:, bridge.history - 1 :]
  joints = bridge.layout.joints
  middle = slice(1, cfg.duration)
  joint_error = (
    (proposed[:, middle, 13 : 13 + joints] - demonstrated[:, middle, 13 : 13 + joints])
    .square()
    .mean()
    .sqrt()
  )
  root_error = (
    (proposed[:, middle, :3] - demonstrated[:, middle, :3]).square().mean().sqrt()
  )
  joint_velocity = (
    proposed[:, 1:, 13 : 13 + joints] - proposed[:, :-1, 13 : 13 + joints]
  ) * bridge.fps
  joint_acceleration = (joint_velocity[:, 1:] - joint_velocity[:, :-1]) * bridge.fps
  proposed_root_steps = torch.linalg.vector_norm(
    proposed[:, 1:, :3] - proposed[:, :-1, :3], dim=-1
  )
  demonstrated_root_steps = torch.linalg.vector_norm(
    demonstrated[:, 1:, :3] - demonstrated[:, :-1, :3], dim=-1
  )
  proposed_joint_steps = (
    (proposed[:, 1:, 13 : 13 + joints] - proposed[:, :-1, 13 : 13 + joints])
    .square()
    .mean(dim=-1)
    .sqrt()
  )
  demonstrated_joint_steps = (
    (demonstrated[:, 1:, 13 : 13 + joints] - demonstrated[:, :-1, 13 : 13 + joints])
    .square()
    .mean(dim=-1)
    .sqrt()
  )
  boundary = cfg.duration - 1
  metrics = {
    "exact_history_end": float(torch.equal(proposed[:, 0], history[:, -1])),
    "exact_target_sequence": float(torch.equal(proposed[:, cfg.duration :], target)),
    "finite": float(torch.isfinite(proposed).all()),
    "middle_root_position_rmse_m": root_error.item(),
    "middle_joint_position_rmse_rad": joint_error.item(),
    "joint_velocity_p95_rad_s": torch.quantile(joint_velocity.abs(), 0.95).item(),
    "joint_acceleration_p95_rad_s2": torch.quantile(
      joint_acceleration.abs(), 0.95
    ).item(),
    "terminal_root_step_p95_m": torch.quantile(
      proposed_root_steps[:, boundary], 0.95
    ).item(),
    "recorded_terminal_root_step_p95_m": torch.quantile(
      demonstrated_root_steps[:, boundary], 0.95
    ).item(),
    "terminal_joint_step_p95_rad": torch.quantile(
      proposed_joint_steps[:, boundary], 0.95
    ).item(),
    "recorded_terminal_joint_step_p95_rad": torch.quantile(
      demonstrated_joint_steps[:, boundary], 0.95
    ).item(),
  }
  for key, value in metrics.items():
    print(f"[diffusion] {key}: {value:.4f}")
  return metrics


if __name__ == "__main__":
  evaluate(tyro.cli(EvaluateCfg, config=mjlab.TYRO_FLAGS))
