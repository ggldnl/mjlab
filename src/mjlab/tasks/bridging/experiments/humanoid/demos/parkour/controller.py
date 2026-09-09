"""What decides which skill runs, where the robot has to be before it starts, and when.

Three things, kept apart:

    RULES        what an obstacle asks for: which skill, and how to work out the pose the
                 robot has to be in before that skill will work. A table
    Controller   reads the scene, steers the walk onto that pose, fires the switch, and
                 tells each skill what it is looking at
    Bridge       gets the robot from wherever the walk left it to the pose. See bridge.py

The controller sees everything and the skills see what it hands them. It reads obstacle
poses and sizes straight off the scene, solves an approach for each one, and points the
climb's obstacle observation at the box in play. A skill knows how it has to be spoken to
and nothing about courses.

Getting the robot in front of the face
--------------------------------------

This is the whole difficulty and it is not a tuned offset.

A hurdle is easy: stand `hurdle_takeoff` back from the near face along the hurdle's own
normal, facing it, and jump.

A box is not. The climb was retargeted from a human motion together with its obstacle, and
the two are one rigid thing: the clip is only physical against that box at that pose. So the
pose the robot has to arrive in is not a distance somebody chose, it is whatever puts the
real obstacle exactly where the reference expects its own.

That cannot be read off the manifest, for two reasons that both bite. The manifest measures
the box against the robot at frame zero, and a hand-over resumes the clip a second into it,
by which point the reference has walked. And `anchor_to_robot` takes a direction of travel
rather than a heading, and a clip's pelvis sits twelve to twenty degrees off its own line of
travel through a run-up. So `Controller.place` anchors, measures where the clip's box
actually landed, corrects, and anchors again, and it reports back the three angles that
follow from that: the pose to arrive in, the heading to hold there, and the heading the
anchor was given. They are all different and conflating any two is the bug it exists to
prevent.

One thing follows that is easy to get wrong, and was. The pose to arrive in depends on
which entry the hand-over resumes at, because the reference is a moving body and where its
obstacle sits relative to it changes down the clip. So the entry has to be chosen before the
pose is solved, and `ready` chooses it first and solves everything else at its frame. Solved
at the table's first row instead, which is what this used to do, the robot is walked to the
spot one entry needs and then handed the state of another: a pose the skill really passes
through, standing somewhere it never passes through it. The climb has one entry and hid it;
the jump has six and does not.

Held short of it, and that is not a detail. A crossing has to cover ground, so the walk
cannot drive at the pose it wants the robot to end up in: parked on it the bridge is asked
for a window of zero and walked through it the window comes out negative, which is a switch
that can never fire again rather than one that fires late. `Controller.stand_off` is the
distance to hold, and it is the distance a crossing covers over the middle of the window the
bridge trained on.

Arrive turned five degrees and the reference climbs a box five degrees off the real one, so
the switch also waits on alignment. `approach.yaw_tolerance` is the climb's own
APPROACH_YAW_RANGE, which is the spread its reference state initialization was trained
across: firing outside it hands the policy a start it never saw.

Four phases per obstacle, two of them bridges:

    cruise     the walk drives, steered onto the approach line and held short of it
    bridge     out, to the traversal skill's entry, at the commanded pose
    traverse   the traversal skill drives, over or onto the obstacle
    bridge     back, to the walk's entry, pointed down the lane

The demo ends when the robot is on the floor, which is what a trip over a hurdle and a fall
off a box both look like.

Run

See run.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.arena import (
  Focus,
  command_name,
  obstacle_names,
)
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.bridge import (
  SLACK,
  Bridge,
  quat_from_yaw,
)
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.course import (
  BOX,
  HURDLE,
  Course,
  Obstacle,
  Settings,
  climb_box,
)
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.pool import SkillPool
from mjlab.tasks.bridging.experiments.humanoid.selector import Reach
from mjlab.tasks.bridging.experiments.humanoid.skills.climb import CLIMB_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.mdp.commands import (
  JumpCommand,
)
from mjlab.tasks.bridging.experiments.humanoid.tests.actors import JUMP, WALK
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import (
  BRIDGE_GROUP,
  ROBOT,
  Actor,
  crossing,
  fresh_obs,
  state,
)

##
# The roster.
##

MOTION = "motion"
"""What a clip tracker calls its reference command, before the arena namespaces it."""


def anchor_for(skill: str):
  """Pin one clip tracker's reference, at the command name the arena gave it.

  `tests.actors.anchor_clip` looks the term up as `motion`, which is right anywhere one
  tracker is loaded and wrong here: the jump and the climb both call their reference that,
  so the arena registers them apart and this resolves the name that skill actually got. See
  `arena.PRIVATE_COMMANDS`.
  """
  term_name = command_name(skill, MOTION)

  def enter(env, pos, heading, frame: int = 0, values=None) -> None:
    del values
    command = env.command_manager.get_term(term_name)
    env_ids = torch.arange(env.num_envs, device=env.device)
    command.anchor_to_robot(env_ids, start_frame=frame, at_pos=pos, at_quat=heading)

  return enter


def rewind(env: ManagerBasedRlEnv, skill: str, frame: int) -> bool:
  """Put a clip tracker's reference back to the frame the hand-over aims at.

  A reference plays whether or not anything is reading it: `_update_command` advances
  `time_steps` every step of the environment, and the bridge takes about a second to cross.
  So a clip pinned when the switch fires has run a second on by the time the skill it belongs
  to starts driving, and the policy takes over chasing a reference sixty frames ahead of the
  robot. For the climb that is the difference between a reference standing in front of the
  box and one already on top of it, which is as survivable as it sounds.

  Rewinding rather than re-anchoring, and that is the point. Anchoring again at hand-over
  would pin the clip to wherever the robot actually arrived, erasing the arrival error the
  bridge is measured on and dragging the climb's own obstacle off the real one. The placement
  is right already; only the clock is wrong, so only the clock is put back.
  """
  try:
    command = env.command_manager.get_term(command_name(skill, MOTION))
  except (KeyError, ValueError):
    return False
  if not isinstance(command, JumpCommand):
    return False
  command.time_steps[:] = frame
  command.motion_done[:] = False
  return True


JUMP_SKILL = Actor(JUMP.name, JUMP.task, enter=anchor_for(JUMP.name))
"""The jump, with its reference pinned at the command the arena gave it rather than at
`motion`. Otherwise as declared in `tests.actors`."""

CLIMB = Actor("climb", CLIMB_TASK_ID, enter=anchor_for("climb"))
"""The climb. A clip tracker like the jump, and the one that makes the placement matter: its
reference carries an obstacle, so pinning it anywhere but the solved approach pose puts a
phantom box beside the real one and the policy climbs the phantom.

Declared here rather than in `tests.actors` only because the demo is where it is used."""

CRUISE_SKILL = WALK
"""What drives between obstacles. The walk, because it is the skill that turns while going
forward, and every obstacle sits on its own approach line."""

ROSTER: tuple[Actor, ...] = (WALK, JUMP_SKILL, CLIMB)
"""Everything the demo loads. The bridge is appended by the pool."""


##
# The rules.
##


def wrap(angle: torch.Tensor) -> torch.Tensor:
  """An angle folded into (-pi, pi]."""
  return torch.atan2(torch.sin(angle), torch.cos(angle))


def rotate(vec: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  """`(N, 2)` turned by `(N,)` radians about z."""
  cos, sin = torch.cos(yaw), torch.sin(yaw)
  return torch.stack(
    [vec[:, 0] * cos - vec[:, 1] * sin, vec[:, 0] * sin + vec[:, 1] * cos], dim=-1
  )


def yaw_of(quat: torch.Tensor) -> torch.Tensor:
  """The yaw of a quaternion, `(N, 4)` wxyz -> `(N,)`."""
  return 2.0 * torch.atan2(quat[:, 3], quat[:, 0])


Approach = Callable[
  ["Controller", Obstacle, torch.Tensor, torch.Tensor],
  tuple[torch.Tensor, torch.Tensor],
]
"""What a rule uses to work out where the robot has to stand: given the obstacle's world
position and yaw, the pose to arrive in."""


def approach_box(
  controller: Controller, obstacle: Obstacle, pos: torch.Tensor, yaw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """A first guess at where the robot stands to climb this box.

  A guess and nothing more, because the answer is not a distance anybody can choose. The
  clip carries its own obstacle, rigid with the motion, so the pose that works is whatever
  puts that obstacle on this one, and `Controller.place` solves for it. This only has to
  start the solve somewhere sensible: in front of the near face, square on.

  Inverting the manifest here instead would be wrong twice over. The manifest measures the
  box against the robot at frame zero, and a hand-over resumes the clip a second in, by which
  point the reference has walked; and the anchor takes a direction of travel rather than a
  heading, so the angle would be off by the pelvis twist as well.
  """
  del obstacle
  box = controller.climb_box
  offset = torch.tensor(
    [box.pos[0], box.pos[1]], device=pos.device, dtype=pos.dtype
  ).expand(pos.shape[0], 2)
  robot_yaw = wrap(yaw - box.yaw)
  return pos[:, 0:2] - rotate(offset, robot_yaw), robot_yaw


def approach_hurdle(
  controller: Controller, obstacle: Obstacle, pos: torch.Tensor, yaw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Square on to the hurdle's near face, one take-off distance back from it."""
  back = obstacle.length / 2.0 + controller.settings.approach.hurdle_takeoff
  offset = torch.tensor([back, 0.0], device=pos.device, dtype=pos.dtype).expand(
    pos.shape[0], 2
  )
  return pos[:, 0:2] - rotate(offset, yaw), yaw


@dataclass(frozen=True)
class Rule:
  """One row of the decision table: an obstacle of this kind asks for that skill."""

  kind: str
  skill: str
  approach: Approach
  """How to work out the pose the robot has to arrive in."""
  why: str

  def row(self) -> str:
    return f"| {self.kind} | {self.skill} | {self.why} |"


RULES: tuple[Rule, ...] = (
  Rule(BOX, "climb", approach_box, "a block, so get on top and down the far side"),
  Rule(HURDLE, "jump", approach_hurdle, "a bar, so clear it in one"),
)
"""The decision table. First match wins.

An obstacle no rule covers is not quietly rounded into the nearest skill: `plan` reports it
and the demo refuses to start, which is the honest behaviour and also the interesting one.
It is what a missing skill looks like from the outside."""

RULE_HEADER = ("| kind | skill | why |", "|---|---|---|")


@dataclass(frozen=True)
class Step:
  """One obstacle and the rule that claimed it. `rule` is None when none did."""

  index: int
  obstacle: Obstacle
  rule: Rule | None

  def row(self) -> str:
    named = self.rule.skill if self.rule else "NO RULE"
    return (
      f"| {self.index} | {self.obstacle.kind} | {self.obstacle.height:.2f} "
      f"| {math.degrees(self.obstacle.yaw):+.0f} | {named} |"
    )


PLAN_HEADER = ("| # | kind | height | yaw | skill |", "|---|---|---|---|---|")


def plan(course: Course, rules: tuple[Rule, ...] = RULES) -> tuple[Step, ...]:
  """Resolve every obstacle to a skill, before anything is built."""
  return tuple(
    Step(index=i, obstacle=o, rule=next((r for r in rules if r.kind == o.kind), None))
    for i, o in enumerate(course)
  )


def unsolved(steps: tuple[Step, ...]) -> tuple[Step, ...]:
  """The obstacles no rule claimed. Empty when the course is solvable."""
  return tuple(step for step in steps if step.rule is None)


def plan_lines(course: Course, steps: tuple[Step, ...]) -> list[str]:
  """The rules, the course, then the plan they produce."""
  out = ["rules:", *RULE_HEADER, *(rule.row() for rule in RULES), ""]
  out += [*course.lines(), ""]
  out += ["plan:", *PLAN_HEADER, *(step.row() for step in steps)]
  blocked = unsolved(steps)
  if blocked:
    out += [
      "",
      f"no plan: {len(blocked)} obstacle(s) no rule covers, of kind "
      f"{', '.join(sorted({s.obstacle.kind for s in blocked}))}. Add a rule, or a skill.",
    ]
  return out


##
# The decisions a run made.
##


@dataclass(frozen=True)
class Decision:
  """One hand-over, and everything that went into it. The audit trail."""

  tick: int
  leaving: str
  entering: str
  entry: str
  effort: float
  binding: str
  duration_s: float
  why: str

  def row(self) -> str:
    return (
      f"| {self.tick} | {self.leaving} | {self.entering} | {self.entry} "
      f"| {self.effort:.2f} | {self.binding} | {self.duration_s:.2f} | {self.why} |"
    )


DECISION_HEADER = (
  "| tick | from | to | entry | effort | binding | window | why |",
  "|---|---|---|---|---|---|---|---|",
)


##
# The phase machine.
##

CRUISE, BRIDGE, TRAVERSE = 0, 1, 2

SETTLE_M = 1.0
"""Metres past an obstacle's far face, along its own axis, before the return bridge opens.

Far enough that the traversal skill has landed and put a foot down. Opening the moment the
obstacle is behind would aim the return bridge at a body still in the air, whose state is
not one any walking entry sits near."""

TRAVERSE_PATIENCE = 300
"""Control steps a traversal gets before the run gives up on it. A climb is slow, and a
skill that never reaches the far side has failed rather than hung."""


class Controller:
  """Reads the scene, decides, and drives the course."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    pool: SkillPool,
    bridge: Bridge,
    course: Course,
    steps: tuple[Step, ...],
    focus: Focus,
  ) -> None:
    self.env, self.pool, self.bridge = env, pool, bridge
    self.course, self.steps, self.focus = course, steps, focus
    self.settings: Settings = course.settings
    self.robot: Entity = env.scene[ROBOT]
    self.names = obstacle_names(course)
    self.climb_box = climb_box()
    """The obstacle the climb was trained against. `approach_box` inverts it."""

    self.phase = CRUISE
    self.index = 0
    self.tick = 0
    self.cruising = CRUISE_SKILL.name
    self.entering = self.cruising
    self.traverse_until = 0
    self.entering_frame = 0
    """The clip frame the open bridge is aiming at. See `rewind`."""
    self.decisions: list[Decision] = []
    self.cleared: list[int] = []
    self.done = False
    self.fell = False

    self.pool[self.cruising].enter(env, *self.here_pose())

  ##
  # Reading the world.
  ##

  def here(self) -> torch.Tensor:
    """The robot's full state, the layout the bridge's target uses."""
    return state(self.robot)

  def here_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Where the robot is and which way it faces, for placing a skill at a reset."""
    here = self.here()
    return here[:, 0:3], here[:, 3:7]

  def obstacle_pose(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One obstacle's world position and yaw, read off the scene.

    Off the scene rather than off the course, so the controller cannot be passing because it
    was handed the answer at build time.
    """
    box: Entity = self.env.scene[self.names[index]]
    return box.data.root_link_pos_w, yaw_of(box.data.root_link_quat_w)

  @property
  def step(self) -> Step | None:
    """The obstacle being worked on, or None once the course is finished."""
    return self.steps[self.index] if self.index < len(self.steps) else None

  def reach_for(self, skill: str) -> Reach | None:
    """The entry of that skill a hand-over would aim at from where the robot is right now.

    Asked every step, because the answer moves: `nearest` ranks the entries by the rate of
    change each demands of a body currently doing this, so an entry out of reach mid-stride
    is within it a moment later. Which is the whole point of the selector, and the reason
    nothing here may cache it.
    """
    if not self.pool[skill].entries:
      return None
    return self.pool[skill].reach(as_numpy(self.here()), self.bridge.duration_range[1])

  def frame_of(self, skill: str) -> int:
    """The frame a hand-over into that skill would resume at, from here.

    The entry `nearest` picks now, not the first row of the table, and that distinction is
    the bug this replaced. The approach pose depends on the frame, because the reference is
    a moving body and where its obstacle sits relative to it changes down the clip. Solved
    at the first row and then aimed at whichever row the ranking chose, the robot is walked
    to the spot one entry needs and handed the state of another: the target is a pose the
    skill really passes through, standing somewhere the skill never passes through it.

    It hid for as long as it did because the climb has one entry, where the two agree. The
    jump has six, spread over the frames a clip travels through, and there they do not.
    """
    reach = self.reach_for(skill)
    return reach.entry.frame if reach is not None else 0

  def place(
    self,
    skill: str,
    frame: int,
    want_xy: torch.Tensor,
    want_yaw: torch.Tensor,
    obstacle: tuple[torch.Tensor, torch.Tensor] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pin a clip tracker's reference, then report the pose the robot has to arrive in.

    Three numbers come back: where the robot must stand, the heading it must hold there, and
    the heading that was handed to the anchor. They are three different angles and conflating
    any two of them is the bug this method exists to prevent.

    `anchor_to_robot` reads its quaternion as the clip's *direction of travel*, not as the
    robot's heading, and its own docstring says so: a clip is canonicalized to travel along
    its +x while the pelvis spends the run-up twelve to twenty degrees off that. So the
    heading to steer at and to aim the bridge's target at is not what goes to the anchor. It
    is what the reference itself holds at the entry frame, which is read back here rather
    than predicted.

    `obstacle` turns this from a placement into a solve, and the climb needs it. Its
    reference carries a box, rigid with the motion, and what has to line up is that box
    against the real one rather than the robot against anything. Anchoring maps the pose
    given here to the clip's placement affinely, so correcting the angle and then the
    position lands it exactly; the loop runs twice because the angle correction moves the
    position too.
    """
    command = self.env.command_manager.get_term(command_name(skill, MOTION))
    assert isinstance(command, JumpCommand)
    env_ids = torch.arange(self.env.num_envs, device=self.env.device)
    origin = self.env.scene.env_origins
    at_xy, at_yaw = want_xy.clone(), want_yaw.clone()

    # Anchor, measure, correct, anchor again, and always end on an anchor. That is why this
    # is a loop and not two statements: correcting the pose without re-anchoring leaves the
    # placement one correction stale, and the clip's box lands a few centimetres off the real
    # one every single time
    for _ in range(3 if obstacle is not None else 1):
      at_pos = torch.cat([at_xy, origin[:, 2:3]], dim=-1)
      command.anchor_to_robot(
        env_ids, start_frame=frame, at_pos=at_pos, at_quat=quat_from_yaw(at_yaw)
      )
      if obstacle is None:
        break
      # Where the clip's own box has ended up, from the anchor the call just wrote
      pos, yaw = obstacle
      box = self.climb_box
      clip_xy = torch.tensor(
        [box.pos[0], box.pos[1]], device=at_xy.device, dtype=at_xy.dtype
      ).expand(at_xy.shape[0], 2)
      here_xy = (
        rotate(clip_xy, command.anchor_yaw) + command.anchor_pos + origin[:, 0:2]
      )
      off_yaw = wrap(yaw - (box.yaw + command.anchor_yaw))
      off_xy = pos[:, 0:2] - here_xy
      if float(off_xy.norm(dim=-1).max()) < 1e-4 and float(off_yaw.abs().max()) < 1e-4:
        break
      at_yaw = at_yaw + off_yaw
      at_xy = at_xy + off_xy

    return command.body_pos_w[:, 0, 0:2], yaw_of(command.body_quat_w[:, 0]), at_yaw

  def approach_pose(
    self, step: Step, frame: int | None = None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Where the robot has to be, and facing where, before that skill will work."""
    return self.aim_for(step, frame)[0:2]

  def aim_for(
    self, step: Step, frame: int | None = None
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """The arrival pose, the heading to hand the anchor, and the frame it was solved at.

    `frame` is the entry the hand-over will resume at, and everything this returns is
    conditional on it. None asks `frame_of` for the one the ranking would pick now, which is
    what steering wants; the switch passes the frame of the reach it has actually chosen, so
    the pose and the state it is going to aim at come from the same entry.
    """
    assert step.rule is not None
    pos, yaw = self.obstacle_pose(step.index)
    frame = self.frame_of(step.rule.skill) if frame is None else frame
    want_xy, want_yaw = step.rule.approach(self, step.obstacle, pos, yaw)
    at_xy, at_yaw, anchor = self.place(
      step.rule.skill,
      frame,
      want_xy,
      want_yaw,
      obstacle=(pos, yaw) if step.obstacle.kind == BOX else None,
    )
    return at_xy, at_yaw, anchor, frame

  def offsets(
    self, step: Step, frame: int | None = None
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """How far the robot is from the approach pose: along it, across it, and in heading."""
    xy, yaw = self.approach_pose(step, frame)
    here = self.here()
    delta = xy - here[:, 0:2]
    along = delta[:, 0] * torch.cos(yaw) + delta[:, 1] * torch.sin(yaw)
    across = -delta[:, 0] * torch.sin(yaw) + delta[:, 1] * torch.cos(yaw)
    return along, across, wrap(yaw_of(here[:, 3:7]) - yaw)

  def stand_off(self, skill: str) -> float:
    """How far short of the approach pose the walk holds station, in metres.

    A crossing has to cross something. Parked on the pose the bridge is asked for a window
    of zero, which is not one it trained on, and walked through it the window comes out
    negative and the switch can never fire again. The demo deadlocked exactly there: 0.16 m
    past the spot with the solve reading -1.5 s, for as long as it was left running.

    So the walk aims to stop the distance a crossing covers over the middle of the trained
    window. At the hold point the robot is stopped, so that crossing averages the entry's
    speed with zero and covers half of it: the distance is half the entry's speed times the
    middle of the range, and the window comes out mid range by construction rather than by a
    number somebody tuned.

    Off the entry's speed and nothing else. Averaging in the robot's own does not converge,
    because a robot stepping in place still has a root velocity swinging through most of a
    stride, so the term never falls to zero and the hold point never settles. Measured with
    it in: 0.38 m held against the 0.07 m a crossing into the climb's entry covers, and a
    solve reading 3.75 s against a range topping out at 1.20.

    Forward speed only. The crossing travels along one line and the walk closes the rest
    sideways, which is what `lateral_gain` is for.
    """
    reach = self.reach_for(skill)
    if reach is None:
      return 0.0
    low, high = self.bridge.duration_range
    return 0.25 * max(float(reach.entry.state[7]), 0.0) * (low + high)

  ##
  # Steering the walk.
  ##

  def steer(self) -> None:
    """Point the walk at the next approach pose, and square it up as it arrives.

    Two objectives and one heading command, blended by distance. Far out the walk is aimed
    at the approach point, which is what actually closes the gap. Close in it is aimed at the
    obstacle's own heading, which is what the traversal skill needs, and the sideways command
    takes over the last of the cross-track error.

    Aiming at a point rather than driving the cross-track error to zero is what keeps it
    stable. Steering on the error alone asks the robot to face the line rather than converge
    on it, and a walk correcting hard while the switch is pending arrives turning.

    Forward speed is regulated rather than held. Held at `approach_speed` the walk crosses
    the whole band the switch can fire in inside about twenty control steps, and if the
    heading has not settled by then it walks through and parks past the spot, where no
    window can land a crossing on the pose and nothing recovers. Regulated onto
    `stand_off`, it converges to holding the distance a crossing needs and waits there for
    as long as the alignment takes.
    """
    if self.phase != CRUISE:
      return
    walk = self.pool[self.cruising]
    cfg = self.settings.walk
    step = self.step
    if step is None or step.rule is None:
      walk.tell(forward=cfg.cruise_speed, lateral=0.0, heading=0.0)
      return

    xy, yaw = self.approach_pose(step)
    here = self.here()
    delta = xy - here[:, 0:2]
    distance = float(delta.norm(dim=-1).mean())
    bearing = float(torch.atan2(delta[:, 1], delta[:, 0]).mean())
    face = float(yaw.mean())

    # 1 far from the approach point, 0 on it. Blends the heading from "go there" to "face
    # the way the skill needs", and fades the sideways correction in as it does
    blend = min(max(distance / max(cfg.blend_radius, 1e-6), 0.0), 1.0)
    heading = face + blend * float(wrap(torch.tensor([bearing - face]))[0])

    along, across, _ = self.offsets(step)
    lateral = float(
      torch.clamp(
        cfg.lateral_gain * across, -cfg.lateral_limit, cfg.lateral_limit
      ).mean()
    )
    # Onto the hold point, not onto the pose. Far out the gain saturates and this is the
    # cruise it always was; close in it slows, and past the hold point it steps back
    forward = float(
      torch.clamp(
        cfg.approach_gain * (along - self.stand_off(step.rule.skill)),
        -cfg.reverse_limit,
        cfg.approach_speed,
      ).mean()
    )
    walk.tell(forward=forward, lateral=lateral * (1.0 - blend), heading=heading)

  ##
  # The switch.
  ##

  def ready(self, step: Step) -> tuple[Reach, float] | None:
    """Whether to start crossing on this step, and over what window.

    Three questions, all of which have to answer yes.

    Aligned, because the traversal skill's reference is rigid with the obstacle and arriving
    turned puts the two out of register. Squarely on the line, for the same reason. And
    reachable: `Bridge.solve` says how long a window would have to be to land the robot on
    the approach pose, and the switch fires as soon as that is a window the bridge was
    trained on. A fixed window could only wait for the world to drift into agreement with
    it, which on a course means walking past the spot whenever the approach did not happen
    to line up.
    """
    assert step.rule is not None
    # The entry first, because everything below is conditional on it. Which entry a
    # hand-over would use decides where the robot has to stand for it, so an alignment
    # measured against some other entry's pose is measured against the wrong line
    reach = self.reach_for(step.rule.skill)
    if reach is None:
      return None

    tolerance = self.settings.approach
    _, across, turn = self.offsets(step, reach.entry.frame)
    if float(turn.abs().max()) > tolerance.yaw_tolerance:
      return None
    if float(across.abs().max()) > tolerance.lateral_tolerance:
      return None

    low, high = self.bridge.duration_range
    here = self.here()
    xy, yaw = self.approach_pose(step, reach.entry.frame)
    target = self.bridge.target_state(reach, xy, yaw)
    seconds, residual = self.bridge.solve(here, target, xy)
    fits = (seconds >= low) & (seconds <= high) & (residual <= SLACK)
    if not bool(fits.all()):
      return None
    return reach, float(seconds.min())

  def past(self, step: Step) -> bool:
    """Whether the robot is clear of the obstacle it just traversed.

    Measured along the obstacle's own axis, not along the lane, because an obstacle turned
    forty degrees is one the robot leaves in a different direction from the one it arrived
    in.
    """
    pos, yaw = self.obstacle_pose(step.index)
    here = self.here()
    delta = here[:, 0:2] - pos[:, 0:2]
    along = delta[:, 0] * torch.cos(yaw) + delta[:, 1] * torch.sin(yaw)
    return bool((along > step.obstacle.length / 2.0 + SETTLE_M).all())

  def down(self) -> bool:
    """Whether the robot is on the floor.

    Two checks, because either alone misses a case. A robot face down on top of a box is
    well above the floor threshold, and one sitting in a heap on the ground is still
    upright. Height catches the fall, tilt catches the sprawl.
    """
    here = self.here()
    if float(here[:, 2].min()) < self.settings.end.fall_height:
      return True
    # The body's own z axis against the world's. A quaternion's rotation of (0,0,1) has
    # z component 1 - 2(x^2 + y^2), which is the cosine of the lean
    quat = here[:, 3:7]
    lean = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    return bool(float(lean.min()) < math.cos(self.settings.end.tip_angle))

  ##
  # The transitions.
  ##

  def cross(
    self,
    entering: str,
    reach: Reach,
    at_xy: torch.Tensor,
    at_yaw: torch.Tensor,
    duration_s: float,
    why: str,
    anchor_yaw: torch.Tensor | None = None,
  ):
    """Aim the bridge at a pose and hand it the world.

    `anchor_yaw` is the heading the entering skill's clip is pinned with, which is its
    direction of travel and not the heading the robot arrives holding. None means the two are
    the same, which is true of a skill with no clip to pin.
    """
    leaving = self.driving
    self.phase, self.entering = BRIDGE, entering
    self.entering_frame = reach.entry.frame
    self.bridge.aim(
      self.pool[entering],
      reach,
      self.here(),
      at_xy,
      at_yaw,
      duration_s,
      anchor_yaw=at_yaw if anchor_yaw is None else anchor_yaw,
      frame=reach.entry.frame,
    )
    self.decisions.append(
      Decision(
        tick=self.tick,
        leaving=leaving,
        entering=entering,
        entry=reach.entry.name,
        effort=reach.effort,
        binding=reach.binding,
        duration_s=duration_s,
        why=why,
      )
    )
    print(f"  {self.decisions[-1].row()}")
    return fresh_obs(self.env)

  def hand_over(self):
    """The window closed. Whoever it was aimed at takes over."""
    gap, angle, worst = self.bridge.arrival(self.here())
    print(
      f"  arrived: {gap:.3f} m, {math.degrees(angle):.1f} degrees and {worst:.2f} rad "
      f"at the worst joint off the pose {self.entering} was promised"
    )
    # The clip has been playing throughout the crossing. Put it back before its own skill
    # reads it, or the policy starts a second into a motion the robot has not begun
    rewind(self.env, self.entering, self.entering_frame)
    if self.entering == self.cruising:
      self.phase = CRUISE
      self.index += 1
      self.done = self.index >= len(self.steps)
    else:
      self.phase = TRAVERSE
      self.traverse_until = self.tick + TRAVERSE_PATIENCE
    return fresh_obs(self.env)

  def resume(self, step: Step):
    """Back to the walk, pointed down the lane."""
    self.cleared.append(step.index)
    walk = self.pool[self.cruising]
    here = self.here()
    reach = walk.reach(here[0].detach().cpu().numpy(), self.bridge.duration_range[1])
    lane = torch.zeros(here.shape[0], device=here.device)
    duration = self.bridge.window(reach)
    # Where a body carrying this momentum would end up over that window. The walk can
    # start anywhere, so nothing demands a spot here and the ballistic placement is the
    # one the bridge trained against
    target = self.bridge.target_state(reach, here[:, 0:2], lane)
    return self.cross(
      self.cruising,
      reach,
      crossing(here, target, duration),
      lane,
      duration,
      "obstacle behind, back to the lane",
    )

  ##
  # The loop.
  ##

  @property
  def driving(self) -> str:
    """Whoever owns the world right now, by name. What the viewer shows."""
    if self.phase == BRIDGE:
      return BRIDGE_GROUP
    return self.cruising if self.phase == CRUISE else self.entering

  @torch.no_grad()
  def __call__(self, obs):
    # Which obstacle the skills are told about, before anything reads an observation
    step = self.step
    if step is not None:
      self.focus.index = step.index

    self.steer()
    if self.phase != BRIDGE:
      self.pool[self.driving].condition(self.env)

    if not self.done and self.down():
      self.fell, self.done = True, True

    if not self.done:
      if self.phase == CRUISE and step is not None and step.rule is not None:
        found = self.ready(step)
        if found is not None:
          reach, solved = found
          # Solved at the frame the reach actually resumes from. `ready` measured the
          # alignment against this same pose, so the robot was let through the gate for the
          # entry it is about to be handed rather than for whichever one the table lists
          # first
          xy, yaw, anchor, _ = self.aim_for(step, reach.entry.frame)
          obs = self.cross(
            step.rule.skill,
            reach,
            xy,
            yaw,
            self.bridge.window(reach, solved),
            step.rule.why,
            anchor_yaw=anchor,
          )
      elif self.phase == BRIDGE and self.bridge.done:
        obs = self.hand_over()
      elif self.phase == TRAVERSE and step is not None:
        if self.past(step) or self.tick >= self.traverse_until:
          obs = self.resume(step)

    self.tick += 1
    return self.pool[self.driving](obs) if self.phase != BRIDGE else self.bridge(obs)

  def report(self) -> list[str]:
    """What the run decided and how far it got."""
    return [
      "",
      "decisions:",
      *DECISION_HEADER,
      *(d.row() for d in self.decisions),
      "",
      f"cleared {len(self.cleared)} of {len(self.steps)} obstacles in {self.tick} steps"
      + (", then went down" if self.fell else ""),
    ]


def as_numpy(here: torch.Tensor) -> np.ndarray:
  """One environment's state, the shape the selector's query wants."""
  return here[0].detach().cpu().numpy()
