"""Restore entry context and place recorded states against a skill's reference."""

from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.selector.state import (
  place_with_reference,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.table import Entry
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.mdp.commands import (
  JumpCommand,
)


def require_context(entry: Entry) -> None:
  """Reject older entries instead of guessing missing resumption inputs."""
  if entry.previous_action is None or entry.reference is None:
    raise ValueError(
      "Entry lacks resumption context. Rerun selector.record, then selector.build"
    )
  joints = (entry.state.size - 13) // 2
  if (
    entry.previous_action.shape != (joints,)
    or not np.isfinite(entry.previous_action).all()
  ):
    raise ValueError("Entry has an invalid preceding action")
  if entry.reference.shape != (7,):
    raise ValueError("Entry has an invalid reference root pose")


def prepare(env: ManagerBasedRlEnv, entry: Entry, command_name: str = "motion") -> None:
  """Select the recorded clip, scale and phase before placing its reference."""
  require_context(entry)
  if not entry.motion_file:
    if entry.reference is not None and np.isnan(entry.reference).all():
      return
    raise ValueError("Tracking entry lacks its recorded clip")
  if command_name not in env.command_manager.active_terms:
    raise ValueError(f"Tracking entry requires command {command_name}")
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, JumpCommand):
    raise TypeError("Tracking entry requires a clip tracker")
  if (
    not entry.motion_file
    or entry.reference is None
    or not np.isfinite(entry.reference).all()
  ):
    raise ValueError("Tracking entry lacks its recorded reference")
  files = [Path(path).name for path in command.cfg.motion_files]
  if files.count(entry.motion_file) != 1:
    raise ValueError(f"Entry clip {entry.motion_file} does not uniquely match {files}")
  motion_id = files.index(entry.motion_file)
  length = int(command.motion.time_step_total_per_motion[motion_id])
  if (
    not 0 <= entry.frame < length
    or not np.isfinite(entry.motion_scale)
    or entry.motion_scale <= 0
  ):
    raise ValueError("Entry phase or scale is invalid")
  command.motion_ids[:] = motion_id
  command.scales[:] = entry.motion_scale
  rewind(env, entry.frame, command_name)


def target(
  env: ManagerBasedRlEnv, entry: Entry, command_name: str = "motion"
) -> torch.Tensor:
  """The recorded robot state under the reference's current placement."""
  require_context(entry)
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, JumpCommand):
    raise TypeError("A reference target requires a clip tracker")
  placed = torch.cat([command.body_pos_w[:, 0], command.body_quat_w[:, 0]], dim=-1)
  assert entry.reference is not None
  states = (
    torch.as_tensor(entry.state, device=env.device)
    .unsqueeze(0)
    .expand(env.num_envs, -1)
  )
  reference = (
    torch.as_tensor(entry.reference, device=env.device)
    .unsqueeze(0)
    .expand(env.num_envs, -1)
  )
  return place_with_reference(states, reference, placed)


def rewind(env: ManagerBasedRlEnv, frame: int, command_name: str = "motion") -> None:
  """Restore phase without moving the reference or replacing the real preceding action."""
  if command_name not in env.command_manager.active_terms:
    return
  command = env.command_manager.get_term(command_name)
  if isinstance(command, JumpCommand):
    command.time_steps[:] = frame
    command.motion_done[:] = False
    command.update_relative_body_poses()


def restore_action(env: ManagerBasedRlEnv, action: torch.Tensor) -> None:
  """Restore a recorded actuator command for an oracle experiment, without stepping physics."""
  env.action_manager.process_action(action)
  env.action_manager.apply_action()
