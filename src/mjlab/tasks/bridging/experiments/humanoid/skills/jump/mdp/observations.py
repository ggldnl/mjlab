"""Observation terms specific to the goal-conditioned jump.

Everything the tracking task already exposes (anchor error, body poses, joint state) is
reused as is. What is added here is the goal and the clock: what the policy is being asked
to do, and where in the jump it is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from .commands import JumpCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _cmd(env: ManagerBasedRlEnv, command_name: str) -> JumpCommand:
  return cast(JumpCommand, env.command_manager.get_term(command_name))


def jump_goal_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Remaining displacement to the landing target, in the robot's heading frame.

  Five numbers: dx, dy, the turn the jump makes, its apex, and the time left until
  touchdown. This is what makes the policy goal-conditioned rather than clip-conditioned.
  """
  return _cmd(env, command_name).goal_b


def jump_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Progress through the clip, as a sine/cosine pair plus the raw scalar.

  ASAP feeds the raw phase. The trigonometric encoding is added because a jump has sharply
  distinct moments (crouch, extension, flight, landing) and a single saturating scalar makes
  them hard to separate early in training.
  """
  phase = _cmd(env, command_name).phase
  return torch.cat(
    [phase, torch.sin(2.0 * torch.pi * phase), torch.cos(2.0 * torch.pi * phase)],
    dim=-1,
  )


def jump_airborne(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Whether the reference is past touchdown, as a float flag."""
  return _cmd(env, command_name).has_landed.float().unsqueeze(-1)


def motion_body_pos_error_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Per-body reference error in the robot's frame, flattened.

  Critic only, matching ASAP's dif_local_rigid_body_pos.
  """
  command = _cmd(env, command_name)
  error_w = command.body_pos_relative_w - command.robot_body_pos_w
  heading = yaw_quat(command.robot_anchor_quat_w)
  heading = heading[:, None, :].repeat(1, error_w.shape[1], 1)
  return quat_apply_inverse(heading, error_w).reshape(env.num_envs, -1)
