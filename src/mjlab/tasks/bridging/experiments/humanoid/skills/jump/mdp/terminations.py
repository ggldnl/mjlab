"""Termination terms for the jump.

The tracking task's terminations (anchor height, anchor orientation, end-effector
drift) are reused from the env config. What is added here is the end of the clip --
which is a time-out, not a failure -- and ASAP's `terminate_when_motion_far`, whose
threshold is the thing the curriculum moves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from .commands import JumpCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _cmd(env: ManagerBasedRlEnv, command_name: str) -> JumpCommand:
  return cast(JumpCommand, env.command_manager.get_term(command_name))


def motion_ended(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """The clip ran out.

  Register this with `time_out=True`. The jump is finished at that point and the
  value function should bootstrap, not treat it as a failure -- getting this wrong
  teaches the policy that completing the motion is bad.
  """
  return _cmd(env, command_name).motion_done


def motion_too_far(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  """Any tracked body more than `threshold` from where the reference says it is.

  ASAP's global tracking failure, and the term the curriculum tightens over
  training. Unlike the z-only variants this is a full 3D distance, so it also
  catches a policy that jumps the right height in the wrong direction.
  """
  command = _cmd(env, command_name)
  error = torch.norm(command.body_pos_w - command.robot_body_pos_w, dim=-1)
  return torch.any(error > threshold, dim=-1)
