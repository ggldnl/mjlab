"""The skills, declared once, with whatever each needs telling when it takes over.

A transition script pairs two of these and nothing else, so a skill's quirks live with the
skill rather than being written out again in every couple's file.

Three things a skill may declare, all optional:

    enter   place its reference, or clear whatever else it keeps per episode
    ready   whether its precondition is met, so the switch fires on the world rather than
            on a button
    place   where its object starts, for a skill that has one

`ready` and `place` are two halves of one idea and share their numbers. A skill was trained
with its object in a particular box relative to the robot, and that box is what its
shortlist means in the world: the ball a stride ahead of the striking foot, the crate a
metre and a half in front. So the object is put out on the line the skill expects, much
further away, and the switch fires when walking has brought it back into the box. The
numbers come from the skills' own configs rather than being restated here, because a
trigger that disagreed with the skill about what "in reach" means is worse than no trigger.

Which of the two an entering skill is decides what a transition to it measures. The pass,
the kick and the push are about the world: a good arrival with the ball in the wrong place
is still a failure, so those couples measure the bridge and the placement together. The
jump, the front kick and the punch combo have nothing on the floor and can happen anywhere,
so a transition to one of them measures only whether the robot arrived in the state the
entry frame asks for.

The kick is the one that needs both halves at once, and it is why `Actor.arrive` takes a
frame. Its window is fifty frames of walking at the ball, so an entry is a point on a
run-up rather than a stance, and where the robot has to stand depends on how much of that
run-up is left. See `arrive_at_kick`.
"""

from __future__ import annotations

from typing import Callable

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.jump import JUMP_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.mdp.commands import (
  JumpCommand,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.kick import KICK_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.kick import mdp as kick_mdp
from mjlab.tasks.bridging.experiments.humanoid.skills.martial import MARTIAL_TASK_IDS
from mjlab.tasks.bridging.experiments.humanoid.skills.passing import PASS_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.passing import mdp as pass_mdp
from mjlab.tasks.bridging.experiments.humanoid.skills.passing.pass_env_cfg import (
  BALL_FORWARD_RANGE as PASS_FORWARD_RANGE,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.passing.pass_env_cfg import (
  BALL_LATERAL_RANGE,
  COMMAND_SPEED_RANGE,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.push import PUSH_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.push import mdp as push_mdp
from mjlab.tasks.bridging.experiments.humanoid.skills.push.push_env_cfg import (
  BOX_FORWARD_RANGE,
  BOX_LATERAL_RANGE,
  BOX_YAW_RANGE,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.walk import WALK_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import Actor, Knob
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_from_euler_xyz,
  yaw_quat,
)

ROBOT = "robot"

Ready = Callable[["ManagerBasedRlEnv", torch.Tensor], torch.Tensor]
"""What `Actor.ready` takes. Named so the factories below can say what they return."""

Arrive = Callable[["ManagerBasedRlEnv", int], torch.Tensor]
"""Likewise for `Actor.arrive`. The int is the entry frame being aimed at."""

APPROACH_RANGE = (2.5, 4.0)
"""How far ahead an object is put, in metres, on whatever line its own skill measures from.
Far enough that the robot has to walk at it for a second or two, so the switch fires out of
ordinary locomotion rather than out of a reset."""

REACH_SLACK = 0.12
"""Grown onto the skill's own box sideways, in metres, before the trigger fires.

The trained boxes are tight, the ball's is 8 cm deep and 10 cm wide, because a reset places
an object exactly and walking cannot. A policy drifting a hand's breadth off its line would
never fire a trigger sized to the reset, and the transition would look impossible when what
failed was the trigger.

Sideways only, and that is the change that made the pass work. Applied forward as well it
decided when the switch fired, and it fired at the near edge of the grown box:

    ball's own range      0.24 to 0.32 m
    grown trigger         0.12 to 0.44 m
    fires at              0.44 m, 16 cm past anything the pass was trained on

For a box 8 cm deep that is a different task. Forward, the trigger now aims at the middle
of the real box and the slack has no say. Sideways it is still a gate, because a lateral
miss is a miss whenever it happens."""


def _reaches(
  delta: torch.Tensor,
  forward_range: tuple[float, float],
  lateral_range: tuple[float, float],
) -> torch.Tensor:
  """Whether the switch should fire now, given where the object will be relative to where
  the robot will be.

  Fires when the predicted arrival puts the object at the middle of the box the skill was
  trained with, not the first moment it is anywhere inside it. The object closes in
  monotonically while the robot walks at it, so the middle is one particular moment and the
  near edge is wherever the box happens to start. Those differ by most of the box.

  The floor stops a robot that has already walked past its object from firing on the way
  out the other side.
  """
  low, high = forward_range
  middle = 0.5 * (low + high)
  return (
    (delta[:, 0] <= middle)
    & (delta[:, 0] > low - REACH_SLACK)
    & (delta[:, 1].abs() < lateral_range[1] + REACH_SLACK)
  )


##
# The ball skills and the push: an object to meet, and a precondition that says when.
##


def _ball_in_reach(forward_range: tuple[float, float]) -> Ready:
  """A precondition that fires when the ball reaches a given box in front of the foot.

  A factory rather than one function, because a ball skill's box is its own: the pass
  strikes at a quarter of a metre and a skill that swings at one would strike at half. A
  trigger sized to one fires at the wrong moment for the other, and getting that wrong does
  not look like a mistimed switch, it looks like the entering skill being unable to do its
  job.

  The pass is the only caller. The kick has a ball too and does not use this: it strikes on
  a run-up rather than from a stand, so what it needs is not a moment when the ball is in a
  box but a place to be standing when it takes over. See `arrive_at_kick`.
  """

  def ready(env: ManagerBasedRlEnv, at: torch.Tensor) -> torch.Tensor:
    """Will the ball be in front of the striking foot once the bridge has crossed?

    Measured from where the robot is going to be, not where it is. A trigger firing on the
    present is a stride late every time: by the time control changes the robot has walked
    through the ball.
    """
    robot = env.scene[ROBOT]
    ball = env.scene[pass_mdp.BALL]
    # The foot carried along with the root. A foot swings while a body travels, and over a
    # bridge window it is the travel that decides whether the ball ends up in front of it
    foot = robot.data.site_pos_w[:, robot.site_names.index(pass_mdp.PASS_SITE)]
    foot = foot + (at - robot.data.root_link_pos_w)
    delta = quat_apply_inverse(
      yaw_quat(robot.data.root_link_quat_w), ball.data.root_link_pos_w - foot
    )
    return _reaches(delta, forward_range, BALL_LATERAL_RANGE)

  return ready


def _arrive_at_ball(forward_range: tuple[float, float]) -> Arrive:
  """Where the root has to stand for the ball to sit at the middle of this skill's box.

  The inverse of `_ball_in_reach`, and deliberately so: that one asks whether a predicted
  arrival happens to put the ball in the right place, this one solves for the arrival that
  does. Same geometry, same numbers, opposite direction.

  Measured through the striking foot, because the box is. The foot's offset from the root is
  taken as it is now rather than as it will be at the entry pose, which is the one
  approximation left in here: over a window the foot swings through most of a stride, so
  this carries whatever part of that has not happened yet. Bounded by a stride and shrinking
  as the walk settles, against a prediction error that compounded three separate sources and
  was bounded by none of them.

  Heading now, not at arrival. The bridge is asked to hold the heading it has, and the
  target state carries the entry frame's pelvis twist on top of it, so the direction the box
  is measured along is the one the robot is already facing.
  """

  def arrive(env: ManagerBasedRlEnv, frame: int) -> torch.Tensor:
    # The frame does not enter into it. The pass wants the ball a quarter of a metre in
    # front of the striking foot and that is the same demand at every entry: its window is
    # a standing shove, not an approach. The kick's is the other case, see arrive_at_kick
    del frame
    robot = env.scene[ROBOT]
    ball = env.scene[pass_mdp.BALL]
    middle = 0.5 * (forward_range[0] + forward_range[1])

    heading = yaw_quat(robot.data.root_link_quat_w)
    ahead = torch.zeros_like(robot.data.root_link_pos_w)
    ahead[:, 0] = middle
    want_foot = ball.data.root_link_pos_w - quat_apply(heading, ahead)

    foot = robot.data.site_pos_w[:, robot.site_names.index(pass_mdp.PASS_SITE)]
    return want_foot - (foot - robot.data.root_link_pos_w)

  return arrive


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


def arrive_at_box(env: ManagerBasedRlEnv, frame: int) -> torch.Tensor:
  """Where the root has to stand for the crate to sit at the middle of the push's box.

  The inverse of `box_in_reach`, the way `_arrive_at_ball` is the inverse of
  `_ball_in_reach`: that one asks whether a predicted arrival happens to put the crate in
  the right place, this one solves for the arrival that does. Same numbers, opposite
  direction, centre to centre because the push's own placement measures that way.

  The frame does not enter into it. The push's window is a stance it opens from and holds,
  not an approach, so every entry wants the robot in the same place.

  Declaring it is what puts the target on the crate rather than wherever the walk's momentum
  was heading, and it is the same call that draws the line of entry states while the walk
  closes in, and that prints the standing error at the hand-over. One answer, used three
  times, which is the point of there being only one.
  """
  del frame
  robot = env.scene[ROBOT]
  box = env.scene[push_mdp.BOX]
  ahead = torch.zeros_like(robot.data.root_link_pos_w)
  ahead[:, 0] = 0.5 * (BOX_FORWARD_RANGE[0] + BOX_FORWARD_RANGE[1])
  return box.data.root_link_pos_w - quat_apply(
    yaw_quat(robot.data.root_link_quat_w), ahead
  )


def clear_pass_phase(
  env: ManagerBasedRlEnv,
  pos: torch.Tensor,
  heading: torch.Tensor,
  frame: int = 0,
  values: dict[str, float] | None = None,
) -> None:
  """Clear the ball skill's episode state as it takes over. No reference, so no target.

  Both the pass and the kick keep per-episode state outside the manager system, whether the
  ball has been touched and how fast it has gone, refreshed by a reset event of their own.
  This arena copies a skill's entities, commands, sensors and observations, and deliberately
  not its events, since most of a skill's reset events move the robot. So that one never
  runs and `ball_contact`, a real observation the policy reads, stays latched from a
  previous hand-over: the skill takes over believing it has already struck the ball.

  Harmless on the first transition of a session and wrong on every one after, which is the
  worst way for a bug to behave.

  Named for the pass because that is where it lives. A second ball skill inherits the same
  phase tracker along with the rest of the environment, and if that ever stops being true
  this splits in two.
  """
  del pos, heading, frame, values
  pass_mdp.reset_pass_phase(env, torch.arange(env.num_envs, device=env.device))


def _place(func, **params) -> EventTermCfg:
  """A skill's own placement, run at reset with the object put much further out.

  The skill's own, so the object lands on the line it expects rather than one invented here:
  the ball squares up with the kicking foot, the crate with the robot's heading. Only the
  distance changes, and only so there is a walk to do before the switch.
  """
  return EventTermCfg(func=func, mode="reset", params=params)


PLACE_BALL = _place(
  pass_mdp.reset_ball_near_foot,
  forward_range=APPROACH_RANGE,
  lateral_range=BALL_LATERAL_RANGE,
)
"""One placement for any ball skill. Where the ball starts is the same problem either way,
a few metres out on the striking foot's line, and only where it has to end up when the
switch fires differs. That difference lives in the trigger."""

PLACE_BOX = _place(
  push_mdp.reset_box_in_front,
  forward_range=APPROACH_RANGE,
  lateral_range=BOX_LATERAL_RANGE,
  yaw_range=BOX_YAW_RANGE,
)


##
# The run: no object and no precondition, but a goal it shares with the walk.
##


RUN_SPEED = 3.0
"""How fast the run is asked to go, in m/s, when it is being profiled and when it takes
over. Fixed rather than sampled, for the same reason the jump's distance is: the rollout
the frames are picked out of and the run that happens at the transition have to be the same
run, or a frame chosen for its momentum is handed to a policy asked for a different one.

Short of the 4.5 the skill's curriculum climbs to. The bridge has to make up the difference
from walking pace inside its window, and every extra metre per second is 0.33 s of the
budget at the 3 m/s^2 a body sustains."""


def enter_run(
  env: ManagerBasedRlEnv,
  pos: torch.Tensor,
  heading: torch.Tensor,
  frame: int = 0,
  values: dict[str, float] | None = None,
) -> None:
  """Pin the twist to a straight run.

  Unlike every other skill here the run has no command of its own: it reads the same twist
  term the walk does, so taking over means saying what that term now holds. `profile` is
  where it matters most. It rolls the run out from a bare reset, and a reset samples the
  twist from whatever range the arena inherited, which is the walk's. Left alone, the
  rollout every target frame is picked out of is a policy trained to run being asked to
  stroll, and its frames carry a walk's momentum under a run's name.

  The flags go with the speed. A reset can mark this environment standing, in which case
  the term zeroes the command it was just given on the next step, or world-framed, in which
  case it rewrites it from a stale copy.
  """
  del pos, heading, frame, values
  twist = env.command_manager.get_term("twist")
  assert isinstance(twist, UniformVelocityCommand)
  twist.vel_command_b[:, 0] = RUN_SPEED
  twist.vel_command_b[:, 1] = 0.0
  twist.vel_command_b[:, 2] = 0.0
  twist.vel_command_w[:] = twist.vel_command_b
  twist.is_standing_env[:] = False
  twist.is_world_env[:] = False
  twist.is_heading_env[:] = False


##
# The clip trackers with nothing on the floor: a reference to place and no precondition.
##


def anchor_clip(
  env: ManagerBasedRlEnv,
  pos: torch.Tensor,
  heading: torch.Tensor,
  frame: int = 0,
  values: dict[str, float] | None = None,
) -> None:
  """Pin a single-clip tracker's reference so it plays from its opening at `pos`, facing
  `heading`.

  Every tracker here: the jump, the front kick and the punch combo. Each carries one clip at
  a pinned scale, has no goal to be conditioned on, and reads the reference every step, so
  placement is the whole of taking over.

  The jump is one of them again. It used to deploy the distilled student of the continuous
  jump, which reads a goal and its own body and no reference at all, so a distance had to be
  written and the frame had nowhere to go. What it deploys now is Mjlab-G1-Jump: one clip,
  jump_forward_level3, 1.54 m every time, tracked at inference the way the other two are.
  Nothing to tell it, and a frame that means something again.

  These are the cleanest test the bridge gets. A ball or a crate makes a hand-over about
  where the robot ends up in the world, and a good arrival with the object in the wrong
  place still fails. Here nothing is on the floor and a jump or a strike can happen
  anywhere, so all that is left is whether the robot arrived in the pose and at the
  velocities the entry frame asks for. Which is what the bridge is actually for.

  `frame` is the frame of the clip the entry was recorded at, and the clip resumes there:
  `anchor_to_robot` winds `time_steps` to it, so the first step after the hand-over reads
  the reference at that frame and the next one carries on from it. The clip frame, not the
  step count: these policies reset into a sampled frame of their clip, so a state recorded
  50 steps into an episode is nowhere in particular. The selector records both and the
  entry carries this one. The front kick
  opens on half a second of held stance, so frame zero would stand still for that half
  second after the bridge had already delivered the robot into the kick; the jump opens on a
  stand and a settle, and entering at zero would replay the whole run-up to a crouch the
  bridge has already put the robot in.
  """
  del values
  env_ids = torch.arange(env.num_envs, device=env.device)
  command = env.command_manager.get_term("motion")
  assert isinstance(command, JumpCommand)
  command.anchor_to_robot(env_ids, start_frame=frame, at_pos=pos, at_quat=heading)


##
# The kick: a tracker whose window is a run-up, and a ball it has to arrive relative to.
##


def arrive_at_kick(env: ManagerBasedRlEnv, frame: int) -> torch.Tensor:
  """Where the root has to be at `frame` for the clip's swing to pass through the real ball.

  The kick is the one skill here whose target cannot be predicted from momentum. Its clip
  runs in for a metre and strikes, the ball sits at the point the converter measured the
  sole closing on it fastest, and both of those are fixed relative to each other inside the
  clip. So the ball decides where the robot has to be, and the ballistic placement, which
  puts the target wherever the robot's own momentum would carry it, aims the kick at empty
  floor. Declaring this is what stops that.

  It is `JumpCommand.anchor_to_robot` read backwards, and deliberately built out of that
  method's own pieces so the two cannot disagree. Anchoring turns the clip by
  `anchor_yaw_for` and then slides it until the root at the entry frame sits on the target,
  which carries the clip's ball to

      target + R(anchor_yaw) * (ball_clip - clip_root(frame))

  and this solves that for the target given where the ball actually is.

  Which makes the answer depend on the frame, unlike every other `arrive` here. The window
  is fifty frames of walking at the ball, about a metre of it, so entering early means
  standing a metre back and entering late means standing a stride back. That dependence is
  the whole reason `Actor.arrive` is handed a frame.

  Heading now, not at arrival, matching `_arrive_at_ball`: the bridge is asked to hold the
  heading the robot already has, and the clip is anchored along that same heading.
  """
  command = kick_mdp.command(env)
  robot = env.scene[ROBOT]
  ball = env.scene[kick_mdp.BALL]

  turn = command.anchor_yaw_for(yaw_quat(robot.data.root_link_quat_w))
  zero = torch.zeros_like(turn)
  ahead = command.ball_target - command.clip_root_at(frame)

  out = robot.data.root_link_pos_w.clone()
  out[:, 0:2] = (
    ball.data.root_link_pos_w[:, 0:2]
    - quat_apply(quat_from_euler_xyz(zero, zero, turn), ahead)[:, 0:2]
  )
  return out


def enter_kick(
  env: ManagerBasedRlEnv,
  pos: torch.Tensor,
  heading: torch.Tensor,
  frame: int = 0,
  values: dict[str, float] | None = None,
) -> None:
  """Anchor the kick's clip, and clear what it has latched about the ball.

  Both halves are needed and neither is optional. The clip has to be placed, like any
  tracker's, and the strike latches have to be cleared, like the pass's: `ball_contact` is
  an observation the policy reads, so a kick taking over with it set from a previous
  hand-over opens believing it has already struck.
  """
  anchor_clip(env, pos, heading, frame, values)
  kick_mdp.reset_kick_phase(env)


##
# What each skill can be told while it drives.
##


def _twist(env: ManagerBasedRlEnv, values: dict[str, float]) -> None:
  """Write a forward speed, a sideways speed and a heading into the velocity command.

  The walk and the run share this term, so only whichever of them owns the world writes it.
  `Run.condition` is what enforces that.

  The two flags are set every step because the term applies them every step, and they are
  sampled at reset and never asked for here. A reset can mark this environment standing, in
  which case the term zeroes the command just written on the next step, or world-framed, in
  which case it rewrites it from a stale copy. One reset in ten used to hand the leaving
  skill a robot told to stand still.
  """
  twist = env.command_manager.get_term("twist")
  assert isinstance(twist, UniformVelocityCommand)
  twist.vel_command_b[:, 0] = values["forward"]
  twist.vel_command_b[:, 1] = values["lateral"]
  twist.heading_target[:] = values["heading"]
  twist.is_heading_env[:] = True
  twist.is_standing_env[:] = False
  twist.is_world_env[:] = False


TWIST_CONTROLS = (
  Knob("forward", -0.5, 2.0, 0.05, 1.0, "Forward speed, m/s."),
  Knob("lateral", -0.5, 0.5, 0.05, 0.0, "Sideways speed, m/s."),
  Knob("heading", -3.14, 3.14, 0.05, 0.0, "Where to face, radians."),
)
"""What a locomotion skill takes. The run declares the same three with a faster default,
because a run asked for a walking speed decelerates back into a walk and undoes the
transition in the second after it was measured."""


def _launch(env: ManagerBasedRlEnv, values: dict[str, float]) -> None:
  """Aim the ball: how fast to send it, and how far off the robot's heading.

  The strike skills are conditioned on a world-frame ball velocity, sampled per episode from
  a speed and a heading offset. Setting it by hand is the same two numbers, resolved against
  the robot's heading now, which is what makes the slider mean the same thing while the robot
  is turning.

  Written every step and idempotent, so it can be applied from the moment the bridge is aimed
  rather than at the instant control changes. A skill with a goal has to have been told it
  before it takes over, or its first step is spent reading whatever the last reset sampled.
  """
  command = env.command_manager.get_term(pass_mdp.COMMAND_NAME)
  assert isinstance(command, pass_mdp.PassCommand)
  robot = env.scene[pass_mdp.ROBOT]
  quat = robot.data.root_link_quat_w
  yaw = torch.atan2(
    2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
    1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
  )
  direction = yaw + values["ball_heading"]
  command.target_vel_w[:, 0] = values["ball_speed"] * torch.cos(direction)
  command.target_vel_w[:, 1] = values["ball_speed"] * torch.sin(direction)


def _launch_controls(speed_range: tuple[float, float]) -> tuple[Knob, ...]:
  """The strike controls, over the speed range the skill was actually trained on.

  Read off the skill's own config rather than restated, because a slider that lets you ask
  for a launch outside the trained range produces a failure that looks like the transition's
  and is not.
  """
  low, high = speed_range
  return (
    Knob(
      "ball_speed",
      low,
      high,
      0.1,
      0.5 * (low + high),
      "How hard to send the ball, m/s.",
    ),
    Knob(
      "ball_heading", -1.0, 1.0, 0.05, 0.0, "Where to send it, radians off the heading."
    ),
  )


WALK = Actor("walk", WALK_TASK_ID, controls=TWIST_CONTROLS, condition=_twist)
JUMP = Actor("jump", JUMP_TASK_ID, enter=anchor_clip)
FRONT_KICK = Actor("front_kick", MARTIAL_TASK_IDS["front_kick"], enter=anchor_clip)
PUNCH_COMBO = Actor("punch_combo", MARTIAL_TASK_IDS["punch_combo"], enter=anchor_clip)
PASS = Actor(
  "pass",
  PASS_TASK_ID,
  enter=clear_pass_phase,
  ready=_ball_in_reach(PASS_FORWARD_RANGE),
  place=PLACE_BALL,
  arrive=_arrive_at_ball(PASS_FORWARD_RANGE),
  controls=_launch_controls(COMMAND_SPEED_RANGE),
  condition=_launch,
)
KICK = Actor(
  "kick",
  KICK_TASK_ID,
  enter=enter_kick,
  place=PLACE_BALL,
  arrive=arrive_at_kick,
)
"""The football kick. A tracker, so it anchors a clip, and a ball skill, so it has to arrive
somewhere in particular, which no other actor here is both of.

No `ready`. The pass fires on its ball reaching a box because the pass strikes from a stand
and the only question is when the ball arrives. The kick answers that question the other
way round: `arrive_at_kick` says where the robot has to be for the swing to meet the ball,
and the switch solves the window rather than waiting for a moment. What is left for the
operator is when to commit, which is the button.

No controls either. The clip is the whole instruction: where the ball goes, how hard and in
which direction are facts dataset.py measured out of it, not a command to be set."""

PUSH = Actor(
  "push",
  PUSH_TASK_ID,
  ready=box_in_reach,
  place=PLACE_BOX,
  arrive=arrive_at_box,
)
