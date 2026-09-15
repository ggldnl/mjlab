"""Place a selected state against a live tracking reference."""

from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.selector.table import Entry
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.mdp.commands import (
  JumpCommand,
)
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)


def prepare(env: ManagerBasedRlEnv, entry: Entry, command_name: str = "motion") -> None:
  """Select the recorded clip, scale, and phase."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, JumpCommand):
    raise TypeError("A selector entry requires a clip tracker")
  files = [Path(path).name for path in command.cfg.motion_files]
  if files.count(entry.motion_file) != 1:
    raise ValueError(f"Entry clip {entry.motion_file} does not uniquely match {files}")
  if not np.isfinite(entry.reference).all() or entry.motion_scale <= 0:
    raise ValueError("Entry reference or scale is invalid")
  command.motion_ids[:] = files.index(entry.motion_file)
  command.scales[:] = entry.motion_scale
  rewind(env, entry.frame, command_name)


def target(
  env: ManagerBasedRlEnv, entry: Entry, command_name: str = "motion"
) -> torch.Tensor:
  """Move a recorded state with its reference to the live reference."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, JumpCommand):
    raise TypeError("A selector entry requires a clip tracker")
  state = torch.as_tensor(entry.state, device=env.device).expand(env.num_envs, -1)
  reference = torch.as_tensor(entry.reference, device=env.device).expand(
    env.num_envs, -1
  )
  placed = torch.cat([command.body_pos_w[:, 0], command.body_quat_w[:, 0]], dim=-1)
  return place(state, reference, placed)


def place(
  state: torch.Tensor, reference: torch.Tensor, placed: torch.Tensor
) -> torch.Tensor:
  """Move states from their recorded reference to a placed reference."""
  rotation = quat_mul(
    yaw_quat(placed[:, 3:7]), quat_conjugate(yaw_quat(reference[:, 3:7]))
  )
  result = state.clone()
  result[:, :3] = placed[:, :3] + quat_apply(rotation, state[:, :3] - reference[:, :3])
  result[:, 3:7] = quat_mul(rotation, state[:, 3:7])
  result[:, 7:10] = quat_apply(rotation, state[:, 7:10])
  result[:, 10:13] = quat_apply(rotation, state[:, 10:13])
  return result


def rewind(env: ManagerBasedRlEnv, frame: int, command_name: str = "motion") -> None:
  """Set the tracker phase without moving its reference."""
  command = env.command_manager.get_term(command_name)
  if isinstance(command, JumpCommand):
    command.time_steps[:] = frame
    command.motion_done[:] = False
    command.update_relative_body_poses()


def restore_action(env: ManagerBasedRlEnv, action: torch.Tensor) -> None:
  """Restore a recorded actuator command without stepping physics."""
  env.action_manager.process_action(action)
  env.action_manager.apply_action()
