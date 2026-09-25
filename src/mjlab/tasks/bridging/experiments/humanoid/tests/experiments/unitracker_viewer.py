"""Play UniTracker on a LAFAN1 clip with its reference ghost.

Run:

    uv run python -m \
      mjlab.tasks.bridging.experiments.humanoid.tests.experiments.unitracker_viewer
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
import tyro
import viser

import mjlab
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import BaseAction
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.execution.tracker import (
  UniTrackerExecutor,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
  CHANNELS,
  channel_errors,
  upper_body_mask,
)
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import state
from mjlab.tasks.unitracker.config.g1.env_cfgs import unitree_g1_unitracker_env_cfg
from mjlab.tasks.unitracker.mdp import MotionCommand, MotionCommandCfg
from mjlab.viewer import ViserPlayViewer


@dataclass
class Config:
  motion: Path = Path("data/lafan1_g1/motions/dance1_subject1.npz")
  device: str = "cuda:0"
  port: int = 8080
  show_path: bool = False


class Policy:
  """Feed the active motion command to the released UniTracker policy."""

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self.env = env
    self.robot: Entity = env.scene["robot"]
    self.command = cast(MotionCommand, env.command_manager.get_term("motion"))
    self.executor = UniTrackerExecutor(env)
    self.upper_body = upper_body_mask(tuple(self.robot.joint_names), env.device)
    self.last_step = -1
    self.info = ""

  def reset(self) -> None:
    action = self.env.action_manager.get_term("joint_pos")
    assert isinstance(action, BaseAction)
    # Match the tracker training reset contract: there is no previous action yet.
    self.env.action_manager.initialize_action(
      torch.zeros_like(self.env.action_manager.action)
    )
    self.last_step = int(self.command.time_steps[0])

  def reference(self) -> torch.Tensor:
    offsets = torch.arange(self.executor.horizon, device=self.env.device)
    rows = (self.command.time_steps[:, None] + offsets).clamp_max(
      self.command.motion.time_step_total - 1
    )
    motion = self.command.motion
    root_pos = motion.body_pos_w[rows, 0] + self.env.scene.env_origins[:, None]
    return torch.cat(
      (
        root_pos,
        motion.body_quat_w[rows, 0],
        motion.body_lin_vel_w[rows, 0],
        motion.body_ang_vel_w[rows, 0],
        motion.joint_pos[rows],
        motion.joint_vel[rows],
      ),
      dim=-1,
    )

  @torch.no_grad()
  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    del obs
    step = int(self.command.time_steps[0])
    if step < self.last_step:
      self.reset()
    self.last_step = step
    reference = self.reference()
    current = state(self.env)
    errors = channel_errors(current, reference[:, 0], self.upper_body)[0]
    self.info = " | ".join(
      f"{name}: {float(value):.3f}"
      for name, value in zip(CHANNELS, errors, strict=True)
    )
    return self.executor(current[:, None], reference)


def run(cfg: Config) -> None:
  if not cfg.motion.is_file():
    raise FileNotFoundError(cfg.motion)
  env_cfg = unitree_g1_unitracker_env_cfg(play=True)
  motion_cfg = env_cfg.commands["motion"]
  assert isinstance(motion_cfg, MotionCommandCfg)
  motion_cfg.motion_file = str(cfg.motion)
  motion_cfg.debug_vis = True
  env_cfg.scene.num_envs = 1
  env_cfg.events = {}
  env_cfg.rewards = {}
  env_cfg.terminations = {}
  env_cfg.metrics = {}
  env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
  wrapped = RslRlVecEnvWrapper(env)
  policy = Policy(env)
  policy.reset()

  command = policy.command
  server = viser.ViserServer(port=cfg.port, label="UniTracker LAFAN1")
  if cfg.show_path:
    points = command.motion.body_pos_w[:, 0].cpu().numpy().astype(np.float64)
    server.scene.add_spline_catmull_rom(
      "/reference/root", points=points, color=(255, 140, 50)
    )
  try:
    ViserPlayViewer(
      wrapped,
      policy,
      viser_server=server,
      info_provider=lambda _: policy.info,
      record_dir="videos/unitracker",
      record_name=cfg.motion.stem,
    ).run()
  finally:
    wrapped.close()


if __name__ == "__main__":
  run(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
