"""What the kick adds to the pass: a toe to strike with, and a strike that has to be fast.

Everything else comes from `passing.mdp` unchanged and is re-exported here, so a config
reaches both through one namespace the way the pass's own does.

The pass learned to shove the ball along the ground with the sole, because that is the
cheapest way for a standing humanoid to move a 0.425 kg ball and nothing in its reward
distinguished it from a strike. What separates the two is not where the ball sits or how
high the foot is, because the same rigid foot makes both contacts with the same capsules.
It is how fast the foot is moving when it arrives and how fast the ball leaves:

    a shove    slow foot, contact lasting many steps, ball at well under a metre per second
    a strike   fast foot, contact over in a step or two, ball at the commanded speed

So the kick measures the approach from the toe, charges for contact made with a slow toe,
and pays a rung of reward for the ball reaching the speed it was asked for. The ball itself
stays inside the reach of a standing robot, which is the one thing an earlier version of
this task got wrong: a ball at the edge of the reach envelope makes approach_toe a gradient
that points at the balance boundary, and the policy follows it over.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

# The pass's whole namespace, which itself re-exports the velocity task's and mjlab's. The
# kick changes four terms out of thirty and inherits the rest
from mjlab.tasks.bridging.experiments.humanoid.skills.passing.mdp import *  # noqa: F401, F403
from mjlab.tasks.bridging.experiments.humanoid.skills.passing.mdp import (
  BALL_RADIUS,
  COMMAND_NAME,
  ROBOT,
  PassCommand,
  ball_pos_w,
  heading_quat,
  pass_window_open,
  phase,
  touching_ball,
)
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


##
# The toe.
##

STRIKE_BODY = "right_ankle_roll_link"
"""The body the striking toe belongs to. The pass measures from a `right_foot` site sitting
at (0.04, 0, -0.037) in this frame, which is the middle of the sole."""

TOE_OFFSET = (0.12, 0.0, -0.02)
"""Where the toe is, in the striking body's frame, in metres.

Read off the collision geometry rather than chosen: the foot capsules run forward to
x = 0.132 at z = -0.025 with a 0.01 radius, so the front surface of the foot is at 0.142 and
this sits 0.022 inside it, 0.015 above the sole. Eight centimetres ahead of the site the
pass measures from, which is the whole difference between a toe and a sole.

Standing flat the ankle body sits 0.035 above the floor, which puts this point at 0.015.
Worth knowing before reading any height off it."""

STRIKE_SPEED = 1.5
"""Toe speed, in m/s, above which a contact counts as a strike rather than a shove.

The separator this task is built on. A foot pushing a ball moves at the ball's speed, which
is under a metre per second for anything that stays a push. Launching a 0.425 kg ball at
the low end of the command range takes a toe doing two to three, so this sits between the
two with room either side.

The one number to move if the policy finds a way to satisfy it while still pushing. Raise
it and a legitimate but gentle strike starts being charged, lower it and a brisk shove
stops being."""


def _toe_offset_w(env: ManagerBasedRlEnv) -> tuple[Entity, int, torch.Tensor]:
  """The striking body, its index, and the toe offset rotated into world axes."""
  robot: Entity = env.scene[ROBOT]
  index = robot.body_names.index(STRIKE_BODY)
  offset = torch.tensor(TOE_OFFSET, device=env.device).expand(env.num_envs, 3)
  return robot, index, quat_apply(robot.data.body_link_quat_w[:, index], offset)


def toe_pos_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """World position of the striking toe, shaped (num_envs, 3).

  Carried on the body rather than declared as a site in the robot's XML. A site would be
  the tidier answer and would be shared by every task on this robot, which is the reason
  not to: one skill's idea of where a toe is does not belong in an asset eleven other
  environments load. Add one temporarily if you want to see it in the viewer.
  """
  robot, index, offset_w = _toe_offset_w(env)
  return robot.data.body_link_pos_w[:, index] + offset_w


def toe_vel_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """World velocity of the striking toe, shaped (num_envs, 3).

  The body's own velocity plus the rotational contribution, which is not a refinement here
  but most of the quantity. A kick is mostly ankle, knee and hip rotation, so a toe twelve
  centimetres out from the body frame gets much of its speed from the cross product rather
  than from the body's linear velocity.
  """
  robot, index, offset_w = _toe_offset_w(env)
  omega = robot.data.body_link_ang_vel_w[:, index]
  return robot.data.body_link_lin_vel_w[:, index] + torch.cross(omega, offset_w, dim=-1)


def toe_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """How fast the striking toe is moving, shaped (num_envs,)."""
  return torch.norm(toe_vel_w(env), dim=-1)


def toe_pos_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The toe seen from the robot's heading frame, shaped (num_envs, 3)."""
  robot: Entity = env.scene[ROBOT]
  return quat_apply_inverse(
    heading_quat(env), toe_pos_w(env) - robot.data.root_link_pos_w
  )


def toe_vel_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The toe's velocity in the robot's heading frame, shaped (num_envs, 3).

  In the observation because it is what shove_cost gates on, for the reason the pass gives
  about its stance window: a term that switches on at a speed the policy cannot see changes
  for no visible reason.
  """
  return quat_apply_inverse(heading_quat(env), toe_vel_w(env))


def toe_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Height of the striking toe above the floor, shaped (num_envs, 1).

  No longer a gate on anything, kept as an observation. Foot clearance is what decides
  whether a swing meets the ball or the floor first, and the toe's z in toe_pos_b is
  measured from a root that moves, so it does not answer the same question.
  """
  return toe_pos_w(env)[:, 2:3]


def toe_surface_gap(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Distance from the toe to the ball's surface, shaped (num_envs,).

  The pass's ball_surface_gap with the toe in place of the foot site. Same floor at zero
  and same reason for subtracting the radius: a kernel on the centre distance peaks at a
  value no foot can reach.
  """
  gap = torch.norm(ball_pos_w(env) - toe_pos_w(env), dim=-1) - BALL_RADIUS
  return torch.clamp(gap, min=0.0)


##
# Rewards.
##


def approach_toe(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
  """The pass's approach term measured from the toe. Shaped (num_envs,).

  Gated the same way the pass gates its own: paid before contact and after the stance
  window, so a foot parked against the ball earns nothing.

  A pure position kernel with no balance qualifier, which is safe only for as long as its
  maximum sits inside the reach of a robot that stays upright. See the ball's forward range
  in kick_env_cfg.py for what keeps it there, and for why moving the ball out to the edge
  of that reach to rule out the shove is the wrong lever to pull.
  """
  score = torch.exp(-torch.square(toe_surface_gap(env)) / std**2)
  active = pass_window_open(env) & ~phase(env).touched
  return torch.where(active, score, torch.zeros_like(score))


def launch_progress(
  env: ManagerBasedRlEnv, command_name: str = COMMAND_NAME
) -> torch.Tensor:
  """How near the ball got to the commanded speed, ignoring direction. (num_envs,).

  The rung the pass is missing. Its ladder steps from touching the ball, worth one, to
  matching a two dimensional velocity, worth five, and a policy that has just learned to
  touch has no gradient telling it that hitting harder is the way up. This fills the gap
  with the one thing that separates a strike from a shove after the fact: how fast the ball
  left.

  Against the commanded speed rather than a constant, so it saturates at the goal instead
  of asking for ever harder contact. Ignoring direction on purpose, because aim is what
  pass_quality is for and asking for both at once is the step that was too large.

  Reads the same latched maximum pass_quality does, which the tracker only writes once the
  striking foot has touched, so this needs no gating of its own.
  """
  p = phase(env)
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, PassCommand)
  target = torch.norm(command.target_vel_w, dim=-1).clamp(min=1.0e-3)
  return torch.clamp(p.max_speed / target, max=1.0)


def shove_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Touching the ball with a slow toe. Shaped (num_envs,), to be weighted down.

  The term that names the failure directly. A shove is a slow contact held for as long as
  it takes to move the ball, and a strike is a fast one over in a step or two, so this
  charges per step for the first and costs a clean kick almost nothing.

  Speed rather than height, which is what an earlier version of this gated on. A toe kick
  of a ball resting on the floor meets it below the ball's centre with the foot low, so a
  height gate charges the motion the task is asking for while a heel planted, toes up
  shovel passes it untouched. Speed makes no such mistake and needs no tuning against the
  ball's radius.

  Charging the trailing steps of a real strike, once the foot has given its speed to the
  ball and is still in contact, is intentional and is the duration half of the argument
  above.
  """
  return (touching_ball(env) & (toe_speed(env) < STRIKE_SPEED)).float()
