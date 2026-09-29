"""Evaluate generated plans against held out BABEL windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.tasks.bridging.bridges.diffusion.config import motion_patterns
from mjlab.tasks.bridging.bridges.diffusion.dataset.motions import Windows, load_motions
from mjlab.tasks.bridging.bridges.diffusion.planner.bridge import (
  DiffusionBridge,
)
from mjlab.utils.lab_api.math import quat_error_magnitude


@dataclass
class EvaluateCfg:
  checkpoint: Path
  robot: str = "g1"
  motions: tuple[str, ...] = ()
  count: int = 128
  holdout: int = 8
  sample_steps: int | None = None
  device: str = "cuda:0"
  seed: int = 0


@torch.no_grad()
def evaluate(cfg: EvaluateCfg) -> dict[str, float]:
  if cfg.count < 1:
    raise ValueError("count must be positive")
  bridge = DiffusionBridge.load(cfg.checkpoint, cfg.device, cfg.sample_steps)
  if bridge.robot != cfg.robot:
    raise ValueError(f"Checkpoint is for {bridge.robot}, not {cfg.robot}")
  corpus = load_motions(
    cfg.motions or motion_patterns(cfg.robot, "val"),
    bridge.process.denoiser.columns,
    cfg.device,
    "all",
    cfg.holdout,
    robot=cfg.robot,
  )
  if corpus.num_joints != bridge.layout.joints or corpus.fps != bridge.fps:
    raise ValueError("Motion data and checkpoint use different robot layouts or rates")
  windows = Windows(
    corpus,
    bridge.history,
    bridge.future,
    bridge.min_steps,
    bridge.max_steps,
  )
  torch.manual_seed(cfg.seed)
  states, duration = windows.states(cfg.count)
  batch = torch.arange(cfg.count, device=cfg.device)[:, None]
  offsets = torch.arange(bridge.future, device=cfg.device)
  target_rows = bridge.history - 1 + duration[:, None] + offsets
  target = states[batch, target_rows]
  path = bridge.generate(states[:, : bridge.history], target, duration).states

  root_position: list[torch.Tensor] = []
  root_orientation: list[torch.Tensor] = []
  root_linear_velocity: list[torch.Tensor] = []
  root_angular_velocity: list[torch.Tensor] = []
  joint_position: list[torch.Tensor] = []
  joint_velocity: list[torch.Tensor] = []
  for index, steps in enumerate(duration.tolist()):
    actual = states[index, bridge.history - 1 : bridge.history + steps]
    generated = path[index, : steps + 1]
    root_position.append(
      torch.linalg.vector_norm(generated[:, :3] - actual[:, :3], dim=-1)
    )
    root_orientation.append(quat_error_magnitude(generated[:, 3:7], actual[:, 3:7]))
    root_linear_velocity.append(
      torch.linalg.vector_norm(generated[:, 7:10] - actual[:, 7:10], dim=-1)
    )
    root_angular_velocity.append(
      torch.linalg.vector_norm(generated[:, 10:13] - actual[:, 10:13], dim=-1)
    )
    joints = bridge.layout.joints
    joint_position.append(
      (generated[:, 13 : 13 + joints] - actual[:, 13 : 13 + joints]).abs().mean(-1)
    )
    joint_velocity.append(
      (generated[:, 13 + joints :] - actual[:, 13 + joints :]).abs().mean(-1)
    )
  exact = torch.tensor(
    [
      torch.equal(path[index, int(steps)], target[index, 0])
      for index, steps in enumerate(duration)
    ],
    device=cfg.device,
    dtype=torch.float,
  )
  result = {
    "exact_boundary_rate": float(exact.mean()),
    "root_position_ade_m": float(torch.cat(root_position).mean()),
    "root_orientation_ade_rad": float(torch.cat(root_orientation).mean()),
    "root_linear_velocity_ade_mps": float(torch.cat(root_linear_velocity).mean()),
    "root_angular_velocity_ade_radps": float(torch.cat(root_angular_velocity).mean()),
    "joint_position_mae_rad": float(torch.cat(joint_position).mean()),
    "joint_velocity_mae_radps": float(torch.cat(joint_velocity).mean()),
  }
  for name, value in result.items():
    print(f"{name}: {value:.6f}")
  return result


if __name__ == "__main__":
  evaluate(tyro.cli(EvaluateCfg, config=mjlab.TYRO_FLAGS))
