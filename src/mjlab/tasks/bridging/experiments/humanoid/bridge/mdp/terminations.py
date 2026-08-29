"""When a window is over, and whether it ended well.

    deadline_reached   the clock ran out. A time-out, so the value function bootstraps
    strayed            the robot walked away from the target. A failure
    fell_over          the torso tipped past recovering. A failure

The distinction matters: bootstrapping off a failure tells the critic that a robot on the
floor is worth whatever it happened to estimate.

`strayed` is what keeps the reward honest. Every term is positive, so surviving always pays
something, and without this a policy could stand safely upright collecting a small kernel
for the whole window instead of crossing it.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.commands import BridgeCommand


def _command(env: ManagerBasedRlEnv, command_name: str) -> BridgeCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, BridgeCommand)
  return term


def deadline_reached(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """The clock ran out.

  A time-out, not a failure. The final step is still scored: terminations run before
  rewards, so the step that trips this is the one `arrival` is paid for and the one the
  arrival is latched from.
  """
  command = _command(env, command_name)
  return command.step >= command.deadline


def strayed(
  env: ManagerBasedRlEnv, command_name: str, margin: float = 1.5
) -> torch.Tensor:
  """The robot is further from the target than the time left can recover.

  Measured against the distance the window opened at, not an absolute bound: a window that
  opens two metres out and one that opens at arm's length are different questions. The
  margin is generous on purpose. A bridge may take a run-up, turn around, or spend the
  first half of a window gathering itself, and all of those open the distance before
  closing it. It may not walk away.
  """
  command = _command(env, command_name)
  distance = (command.robot.data.root_link_pos_w - command.target[:, 0:3]).norm(dim=-1)
  return distance > command.start_distance + margin


def fell_over(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.7
) -> torch.Tensor:
  """The torso is tipped past recovering.

  Read off projected gravity, not root height: a deep crouch and a fall reach the same
  height and only one is a failure. A dataset built from a jumping skill is full of deep
  crouches, since that is what a body does before it leaves the ground.
  """
  asset = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b[:, 2] > -threshold
