"""When a bridging attempt is over, and whether it ended well.

Two of these are failures and one is not, and keeping them apart matters more here than
usual. `hole_closed` is a truncation: the policy ran the whole hole plus its tail and the
episode simply ran out, so the value function must bootstrap rather than treat it as a
dead end. The other two are real failures and must not bootstrap.

Termination on lost tracking is what stops the reward from being gameable. Every tracking
term is positive, so surviving is always worth something, and without this a policy could
settle into standing safely upright collecting a small kernel forever instead of crossing
the hole. Cutting the episode the moment tracking is lost removes that option: the small
reward stops arriving.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.commands import BridgeCommand


def _command(env: ManagerBasedRlEnv, command_name: str) -> BridgeCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, BridgeCommand)
  return term


def hole_closed(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """The hole and its tail are behind us. A time-out, not a failure."""
  command = _command(env, command_name)
  return command.step >= command.gap + command.tail


def lost_tracking(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  """The robot is further from the reference than any recovery is going to fix."""
  command = _command(env, command_name)
  error = (command.robot.data.root_link_pos_w - command.reference_now()[:, 0:3]).norm(
    dim=-1
  )
  return error > threshold


def fell_over(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.7
) -> torch.Tensor:
  """The torso is tipped past recovering.

  Read off projected gravity rather than height, because a deep crouch and a fall reach
  the same root height and only one of them is a failure. This corpus is full of deep
  crouches, since that is what a body does before it jumps.
  """
  asset = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b[:, 2] > -threshold
