"""When a bridging attempt is over, and whether it ended well.

Two of these are failures and one is not, and keeping them apart matters more here than
usual. `window_done` is a truncation: the policy ran the hole, handed over, and drove the
whole second window afterwards, and the episode simply ran out, so the value function must
bootstrap rather than treat it as a dead end. The other two are real failures and must not
bootstrap.

That distinction is doing more work than it used to. The episode no longer stops shortly
after the hole closes, so falling over during the resume -- which is what a bad hand-off
looks like from the outside, a second later -- is now inside the episode and reachable by
these terms. A bridge that arrives with the wrong momentum used to be truncated before the
consequence showed up, and truncation bootstraps: the value function was told the state
was worth whatever it estimated, rather than being shown a robot on the floor.

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


def window_done(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """The hole, the hand-off and the whole second window are behind us.

  A time-out, not a failure. The last frame is scored on the way out: terminations are
  computed before rewards, so the step that trips this is still paid for.
  """
  command = _command(env, command_name)
  return command.step >= command.gap + command.resume


def lost_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  resume_threshold: float,
  stray_margin: float,
) -> torch.Tensor:
  """The robot is further from where it should be than any recovery is going to fix.

  Three bounds, because there are three situations and only two of them have a reference.

  Inside the hole of a coherent window the reference is one of many ways across and the
  robot is not required to be on it, so the bound is loose and is there to catch a bridge
  that has stopped trying. Past the hand-off the reference is the motion that is supposed
  to be under way, and being half a metre from it is not a stylistic difference -- it is
  the composition having failed, whatever the robot goes on to do. Ending the episode there
  rather than letting it run is what makes that a failure the value function has to account
  for.

  Inside the hole of a spliced window there is nothing to be off. The reference there is
  the arrival frame, held (see mdp/commands.py), so the distance being measured is distance
  from the destination, and the only thing that can be said about it is that it should not
  be growing without bound. `stray_margin` is the slack on top of the distance the hole
  opened at: a bridge is free to take a run-up, to turn around, to spend the first half of
  the hole gathering itself, and is not free to walk away. Nothing tighter is available
  without inventing a trajectory, which is the thing the splice exists to stop doing.
  """
  command = _command(env, command_name)
  error = (command.robot.data.root_link_pos_w - command.reference_now()[:, 0:3]).norm(
    dim=-1
  )
  bound = torch.where(
    command.tracked(),
    torch.full_like(error, threshold),
    command.start_distance + stray_margin,
  )
  bound = torch.where(
    command.resuming(), torch.full_like(error, resume_threshold), bound
  )
  return error > bound


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
