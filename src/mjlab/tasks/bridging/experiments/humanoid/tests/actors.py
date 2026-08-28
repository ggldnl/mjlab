"""The five skills, declared once, with whatever each needs telling when it takes over.

A transition script pairs two of these and nothing else. Keeping them here rather than in
the transition scripts means a skill's quirks live with the skill instead of being written
out again, or imported sideways from another couple's file, every time it appears.

Three things a skill may declare, all optional:

    enter   place its reference, or clear whatever else it keeps per episode
    ready   whether its precondition is met, so the switch can fire on the world rather
            than on a button
    place   where its object starts, for a skill that has one

`ready` and `place` are two halves of one idea and they use the same numbers. A skill was
trained with its object in a particular box relative to the robot, and that box is what its
initiation set means in the world: the ball a stride ahead of the kicking foot, the crate a
metre and a half in front. So the object is put out on the same line the skill expects it,
just much further away, and the switch fires when walking has brought it back into the box.
The numbers come from the skills' own configs rather than being restated here, because a
trigger that disagreed with the skill about what "in reach" means is worse than no trigger.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.jump import JUMP_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.jump.mdp.commands import (
  JumpCommand,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.kick import KICK_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.kick import mdp as kick_mdp
from mjlab.tasks.bridging.experiments.humanoid.skills.kick.kick_env_cfg import (
  BALL_FORWARD_RANGE,
  BALL_LATERAL_RANGE,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.push import PUSH_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.push import mdp as push_mdp
from mjlab.tasks.bridging.experiments.humanoid.skills.push.push_env_cfg import (
  BOX_FORWARD_RANGE,
  BOX_LATERAL_RANGE,
  BOX_YAW_RANGE,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.run import RUN_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.walk import WALK_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import Actor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

ROBOT = "robot"

APPROACH_RANGE = (2.5, 4.0)
"""How far ahead an object is put, in metres, measured on whatever line its own skill
measures from. Far enough that the robot has to walk to it for a second or two, which is
the point: the switch has to fire out of ordinary locomotion rather than out of a reset."""

REACH_SLACK = 0.12
"""Grown onto the skill's own box sideways, in metres, before the trigger will fire.

The trained boxes are tight -- the ball's is eight centimetres deep and ten wide -- because
a reset can place an object exactly. Walking cannot. A policy that drifts a hand's breadth
off its line would never fire a trigger sized to the reset, and the transition would look
impossible when what failed was the trigger.

Sideways only, and that is the change that made the kick work. Grown onto the *forward*
range as well, this decided when the switch fired, and it fired at the near edge of the
grown box: the ball's own range is 0.24 to 0.32 m, so a trigger watching (0.12, 0.44) fires
the moment the ball passes 0.44 m, which is sixteen centimetres beyond the far end of
anything the kick was trained on -- for a box eight centimetres deep, a different task.
Forward, the trigger now aims at the middle of the real box and the slack has no say in
it; sideways it is still a gate, because a lateral miss is a miss whenever it happens."""


def _reaches(
  delta: torch.Tensor,
  forward_range: tuple[float, float],
  lateral_range: tuple[float, float],
) -> torch.Tensor:
  """Whether the switch should fire now, given where the object will be relative to where
  the robot will be.

  Fires when the predicted arrival has brought the object to the middle of the box the
  skill was trained with, rather than the first moment it is anywhere inside it. The object
  closes in monotonically while the robot walks at it, so the first is one particular
  moment and the second is whichever moment the box's near edge happens to fall on -- and
  those differ by most of the box.

  The floor is there so a robot that has already walked past its object does not fire on
  the way out the other side.
  """
  low, high = forward_range
  middle = 0.5 * (low + high)
  return (
    (delta[:, 0] <= middle)
    & (delta[:, 0] > low - REACH_SLACK)
    & (delta[:, 1].abs() < lateral_range[1] + REACH_SLACK)
  )


##
# The jump: a reference to place, no object, no precondition. Fires on the button.
##

JUMP_DISTANCE = 1.55
"""How far the jump is asked to go, in metres. Fixed rather than sampled, so the profiled
rollout and the clip the skill is handed at the transition are the same jump."""


def anchor_jump(
  env: ManagerBasedRlEnv, frame: int, pos: torch.Tensor, heading: torch.Tensor
) -> None:
  """Pin the jump's clip so it travels along `heading` and arrives at `pos` at `frame`.

  The jump is a motion tracker: it follows a clip anchored somewhere in the world, and in
  its own environment that anchor is the origin because its reset teleports the robot onto
  the clip's first frame. Here the robot is wherever the previous skill left it, so the
  clip has to come to the robot instead.

  `pos` is where the robot is *going* to be, not where it is. Anchoring to the robot's
  actual pose at hand-over would slide the clip onto whatever the bridge managed, which
  erases the arrival error rather than making the skill cope with it.

  Placement and nothing else. This used to return the state the clip holds at the entry
  frame, for the harness to aim the bridge at, and that was wrong in a way that took a
  ghost to see: the clip is a retargeted human motion, so at frame 90 it stands with 3.8 cm
  of foot through the floor and descending at twice the rate the policy tracking it
  descends. The state to arrive in is the one the policy produced there, which the harness
  records itself.

  `set_goal` and not `apply_goals`, which is the same assignment plus a snap of the robot
  onto the clip's opening frame. That snap is right when an evaluation is restarting this
  skill from scratch and catastrophic here: it teleports the robot away mid-transition, and
  what it looks like from the outside is the bridge losing a robot that was taken from it.
  """
  env_ids = torch.arange(env.num_envs, device=env.device)
  command = env.command_manager.get_term("motion")
  assert isinstance(command, JumpCommand)
  command.set_goal(env_ids, *command.solve_goal(JUMP_DISTANCE))
  command.anchor_to_robot(env_ids, start_frame=frame, at_pos=pos, at_quat=heading)


##
# The kick and the push: an object to meet, and a precondition that says when.
##


def ball_in_reach(env: ManagerBasedRlEnv, at: torch.Tensor) -> torch.Tensor:
  """Will the ball be in front of the kicking foot once the bridge has crossed?

  Measured from where the robot is *going* to be, not where it is. A trigger that fired on
  the present would be a stride late every time: by the time control changed the robot
  would have walked through the ball.
  """
  robot = env.scene[ROBOT]
  ball = env.scene[kick_mdp.BALL]
  # The foot carried along with the root. A foot swings while a body travels, and over a
  # bridge window it is the travel that decides whether the ball ends up in front of it.
  foot = robot.data.site_pos_w[:, robot.site_names.index(kick_mdp.KICK_SITE)]
  foot = foot + (at - robot.data.root_link_pos_w)
  delta = quat_apply_inverse(
    yaw_quat(robot.data.root_link_quat_w), ball.data.root_link_pos_w - foot
  )
  return _reaches(delta, BALL_FORWARD_RANGE, BALL_LATERAL_RANGE)


def box_in_reach(env: ManagerBasedRlEnv, at: torch.Tensor) -> torch.Tensor:
  """Will the crate be in front of the robot once the bridge has crossed?

  Centre to centre, the way the push's own placement measures it, so the near face sits one
  half-extent closer than the numbers say.
  """
  robot = env.scene[ROBOT]
  box = env.scene[push_mdp.BOX]
  delta = quat_apply_inverse(
    yaw_quat(robot.data.root_link_quat_w), box.data.root_link_pos_w - at
  )
  return _reaches(delta, BOX_FORWARD_RANGE, BOX_LATERAL_RANGE)


def clear_kick_phase(
  env: ManagerBasedRlEnv, frame: int, pos: torch.Tensor, heading: torch.Tensor
) -> None:
  """Clear the kick's episode state as it takes over. No reference to place, so no target.

  The kick keeps per-episode state outside the manager system -- whether the ball has been
  touched, how fast it has gone -- refreshed by a reset event of its own. This arena copies
  a skill's entities, commands, sensors and observations, and deliberately not its events,
  because most of a skill's reset events move the robot. So that one never runs, and
  `ball_contact`, which is a real observation the policy reads, would stay latched from a
  previous hand-over: the kick would take over believing it had already struck the ball.

  Harmless on the first transition of a session and wrong on every one after it, which is
  the worst way for a bug to behave.
  """
  del frame, pos, heading
  kick_mdp.reset_kick_phase(env, torch.arange(env.num_envs, device=env.device))


def _place(func, **params) -> EventTermCfg:
  """A skill's own placement, run at reset with the object put much further out.

  Its own, so the object lands on the line the skill expects rather than on a line invented
  here: the ball squares up with the kicking foot, the crate with the robot's heading. Only
  the distance changes, and only so there is a walk to do before the switch.
  """
  return EventTermCfg(func=func, mode="reset", params=params)


PLACE_BALL = _place(
  kick_mdp.reset_ball_near_foot,
  forward_range=APPROACH_RANGE,
  lateral_range=BALL_LATERAL_RANGE,
)

PLACE_BOX = _place(
  push_mdp.reset_box_in_front,
  forward_range=APPROACH_RANGE,
  lateral_range=BOX_LATERAL_RANGE,
  yaw_range=BOX_YAW_RANGE,
)


WALK = Actor("walk", WALK_TASK_ID)
RUN = Actor("run", RUN_TASK_ID)
JUMP = Actor("jump", JUMP_TASK_ID, enter=anchor_jump)
KICK = Actor(
  "kick", KICK_TASK_ID, enter=clear_kick_phase, ready=ball_in_reach, place=PLACE_BALL
)
PUSH = Actor("push", PUSH_TASK_ID, ready=box_in_reach, place=PLACE_BOX)
