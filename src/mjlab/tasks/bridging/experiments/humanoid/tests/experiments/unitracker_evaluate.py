"""Measure a universal tracker over one retargeted motion clip.

Run:

    uv run python -m \
      mjlab.tasks.bridging.experiments.humanoid.tests.experiments.unitracker_evaluate \
      --motion data/lafan1_g1/motions/walk1_subject1.npz
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
  CHANNELS,
  Tolerances,
  channel_errors,
  upper_body_mask,
)
from mjlab.tasks.bridging.experiments.humanoid.tests.experiments.unitracker_viewer import (
  Policy,
)
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import state
from mjlab.tasks.unitracker.config.g1.env_cfgs import unitree_g1_unitracker_env_cfg
from mjlab.tasks.unitracker.mdp import MotionCommandCfg

UNITS = ("m", "rad", "m/s", "rad/s", "rad", "rad/s", "rad", "rad/s")


@dataclass
class Config:
  motion: Path = Path("data/lafan1_g1/motions/dance1_subject1.npz")
  device: str = "cuda:0"
  max_steps: int | None = None
  output_dir: Path | None = Path("data/unitracker/evaluation")


class ClipPolicy(Protocol):
  command: Any

  def reset(self) -> None: ...

  def reference(self) -> torch.Tensor: ...

  def __call__(self, observation: torch.Tensor, /) -> torch.Tensor: ...


@torch.no_grad()
def evaluate_tracker(
  cfg: Config,
  env_cfg,
  policy_factory: Callable[[ManagerBasedRlEnv], ClipPolicy],
  tracker_name: str,
) -> torch.Tensor:
  if not cfg.motion.is_file():
    raise FileNotFoundError(cfg.motion)
  if cfg.max_steps is not None and cfg.max_steps < 1:
    raise ValueError("max_steps must be positive")

  motion_cfg = env_cfg.commands["motion"]
  assert isinstance(motion_cfg, MotionCommandCfg)
  motion_cfg.motion_file = str(cfg.motion)
  motion_cfg.debug_vis = False
  env_cfg.scene.num_envs = 1
  env_cfg.events = {}
  env_cfg.rewards = {}
  env_cfg.terminations = {}
  env_cfg.metrics = {}

  env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
  try:
    policy = policy_factory(env)
    robot: Entity = env.scene["robot"]
    upper_body = upper_body_mask(tuple(robot.joint_names), env.device)
    limits = Tolerances().tensor(env.device)
    env.reset()
    policy.reset()

    command = policy.command
    total = command.motion.time_step_total
    steps = min(total, cfg.max_steps or total)
    samples: list[torch.Tensor] = []
    root_height: list[torch.Tensor] = []
    reference_height: list[torch.Tensor] = []
    for _ in range(steps):
      reference = policy.reference()
      samples.append(channel_errors(state(env), reference[:, 0], upper_body)[0])
      root_height.append(robot.data.root_link_pos_w[0, 2].clone())
      reference_height.append(reference[0, 0, 2].clone())
      env.step(policy(torch.empty(0, device=env.device)))

    errors = torch.stack(samples)
    heights = torch.stack(root_height)
    reference_heights = torch.stack(reference_height)
    lost = (heights - reference_heights).abs() > 0.25
    lost |= errors[:, 1] > 0.8
    lost_frames = torch.where(lost)[0]
    valid_steps = int(lost_frames[0]) if lost_frames.numel() else steps
    measured = errors[: max(valid_steps, 1)]
    passed = measured <= limits
    quantiles = torch.quantile(
      measured, torch.tensor((0.5, 0.9, 0.99), device=errors.device), dim=0
    )
    means = measured.mean(0)
    maxima, worst_frames = measured.max(0)

    print(
      f"{tracker_name} | {cfg.motion} | {steps} frames | {steps * env.step_dt:.2f} s"
    )
    if lost_frames.numel():
      print(
        f"tracking lost at frame {valid_steps} ({valid_steps * env.step_dt:.2f} s); "
        f"statistics use the {valid_steps} pre-loss frames"
      )
    else:
      print("tracking remained valid for the complete clip")
    print(f"strict pass {100 * passed.all(1).float().mean():.2f}%")
    print(
      f"{'channel':<22} {'mean':>9} {'p50':>9} {'p90':>9} {'p99':>9} "
      f"{'max':>9} {'limit':>9} {'pass':>8} {'worst':>7}"
    )
    for index, (name, unit) in enumerate(zip(CHANNELS, UNITS, strict=True)):
      print(
        f"{name + ' (' + unit + ')':<22} "
        f"{means[index]:9.4f} {quantiles[0, index]:9.4f} "
        f"{quantiles[1, index]:9.4f} {quantiles[2, index]:9.4f} "
        f"{maxima[index]:9.4f} {limits[index]:9.4f} "
        f"{100 * passed[:, index].float().mean():7.2f}% "
        f"{int(worst_frames[index]):7d}"
      )

    if cfg.output_dir is not None:
      cfg.output_dir.mkdir(parents=True, exist_ok=True)
      suffix = "" if tracker_name == "unitracker" else f"_{tracker_name}"
      output = cfg.output_dir / f"{cfg.motion.stem}{suffix}_errors.npz"
      np.savez_compressed(
        output,
        channels=np.asarray(CHANNELS),
        errors=errors.cpu().numpy(),
        tolerances=limits.cpu().numpy(),
        passed=(errors <= limits).cpu().numpy(),
        strict_pass=(errors <= limits).all(1).cpu().numpy(),
        tracking_lost=lost.cpu().numpy(),
        root_height=heights.cpu().numpy(),
        reference_root_height=reference_heights.cpu().numpy(),
        fps=np.asarray([1.0 / env.step_dt], dtype=np.float32),
      )
      print(f"saved {output}")
    return errors
  finally:
    env.close()


def evaluate(cfg: Config) -> torch.Tensor:
  return evaluate_tracker(
    cfg,
    unitree_g1_unitracker_env_cfg(play=True),
    Policy,
    "unitracker",
  )


if __name__ == "__main__":
  evaluate(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
