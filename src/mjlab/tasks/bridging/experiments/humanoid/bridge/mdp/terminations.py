"""When a window is over, and whether it ended well.

One of these is a time-out and two are failures, and the difference decides what the value
function is told. `deadline_reached` is the window running its course: the policy was
asked for a fixed number of steps and it produced them, so the state it ends in is worth
bootstrapping from. The other two are the episode being cut short, and bootstrapping there
would tell the critic that a robot on the floor is worth whatever it happened to estimate.

`strayed` is what keeps the reward honest. Every term is positive, so surviving always
pays something, and without this a policy could settle into standing safely upright
collecting a small kernel for the whole window instead of crossing it. Ending the episode
when the robot is further from the target than it started removes that option: the small
reward stops arriving.
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

  A time-out, not a failure. The final step is still scored on the way out: terminations
  are computed before rewards, so the step that trips this is the one `arrival` is paid
  for and the one the arrival is latched from.
  """
  command = _command(env, command_name)
  return command.step >= command.deadline


def strayed(
  env: ManagerBasedRlEnv, command_name: str, margin: float = 1.5
) -> torch.Tensor:
  """The robot is further from the target than any recovery in the time left will fix.

  Measured against the distance the window opened at rather than an absolute bound,
  because a window that opens two metres out and one that opens at arm's length are not
  the same question. The margin is generous on purpose: a bridge is free to take a run-up,
  to turn around, to spend the first half of a window gathering itself, and all of those
  open the distance before they close it. It is not free to walk away.
  """
  command = _command(env, command_name)
  distance = (command.robot.data.root_link_pos_w - command.target[:, 0:3]).norm(dim=-1)
  return distance > command.start_distance + margin


def fell_over(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.7
) -> torch.Tensor:
  """The torso is tipped past recovering.

  Read off projected gravity rather than root height, because a deep crouch and a fall
  reach the same height and only one of them is a failure. A bank built from a jumping
  skill is full of deep crouches, since that is what a body does before it leaves the
  ground.
  """
  asset = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b[:, 2] > -threshold
