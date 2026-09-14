"""The plan, and the loop that runs it one action at a time.

A course is compiled once into a flat list of actions and the run is an index walking down
it. No phase machine: the phases were always the same three per obstacle, so they are
written out instead of derived.

    go_to    the walk, aimed at a point in the world
    cross    the bridge, aimed at the pose a traversal needs
    climb    that skill, until it is back on the ground standing
    go_to    the walk again, aimed at the next point

Every action answers four questions and the loop asks nothing else: start takes the world,
done says whether to advance, drive is one step of a policy, finish hands the world back.
The rules that pick a skill per obstacle live in RULES and run once, in plan.

The walk is not asked to be accurate. It stops approach.hold_back short of the pose a skill
needs and the bridge covers the rest inside a fixed approach.window_s. The pose itself is
solved rather than chosen, once per obstacle: see approach.py.

A traversal ends on the robot, not on its clip. Both skills leave the ground and come back,
so the lowest foot rising past traverse.lift_height arms the action and returning below
traverse.land_height upright ends it. That leaves the robot standing at about zero
velocity, inside the walk's initiation set, so there is no bridge on the way out.

Run

See run.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.env_cfg import (
  FOOT_BODIES,
)
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour import approach
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.approach import (
  MOTION,
  Approach,
  lean_of,
  wrap,
  yaw_of,
)
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.arena import (
  Focus,
  command_name,
  obstacle_names,
)
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.bridge import Bridge
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.course import (
  BOX,
  HURDLE,
  Course,
  Obstacle,
  Settings,
)
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.pool import SkillPool
from mjlab.tasks.bridging.experiments.humanoid.selector import Entry, Reach
from mjlab.tasks.bridging.experiments.humanoid.selector import resume as entry_resume
from mjlab.tasks.bridging.experiments.humanoid.skills.climb import CLIMB_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.mdp.commands import (
  JumpCommand,
)
from mjlab.tasks.bridging.experiments.humanoid.tests.actors import JUMP, WALK
from mjlab.tasks.bridging.experiments.humanoid.tests.entry_tolerances import (
  ToleranceOverrides,
)
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import (
  BRIDGE_GROUP,
  ROBOT,
  Actor,
  fresh_obs,
  state,
)

##
# The roster.
##


def anchor_for(skill: str):
  """Pin one clip tracker's reference, at the command name the arena gave it.

  tests.actors.anchor_clip looks the term up as motion, which is right anywhere one tracker
  is loaded and wrong here: the jump and the climb both call their reference that, so the
  arena registers them apart and this resolves the name that skill actually got. See
  arena.PRIVATE_COMMANDS.
  """
  term_name = command_name(skill, MOTION)

  def enter(env, pos, heading, frame: int = 0, values=None) -> None:
    del values
    command = env.command_manager.get_term(term_name)
    env_ids = torch.arange(env.num_envs, device=env.device)
    command.anchor_to_robot(env_ids, start_frame=frame, at_pos=pos, at_quat=heading)

  return enter


JUMP_SKILL = Actor(JUMP.name, JUMP.task, enter=anchor_for(JUMP.name))
"""The jump, with its reference pinned at the command the arena gave it rather than at
motion. Otherwise as declared in tests.actors."""

CLIMB = Actor("climb", CLIMB_TASK_ID, enter=anchor_for("climb"))
"""The climb. A clip tracker like the jump, and the one that makes the placement matter: its
reference carries an obstacle, so pinning it anywhere but the solved approach pose puts a
phantom box beside the real one and the policy climbs the phantom."""

CRUISE_SKILL = WALK
"""What drives between obstacles, and what takes back over after every traversal."""

ROSTER: tuple[Actor, ...] = (WALK, JUMP_SKILL, CLIMB)
"""Everything the demo loads. The bridge is appended by the pool."""

ENTRIES: dict[str, int] = {"climb": 49, "jump": 86, "walk": 44}
"""The clip frame this demo hands each skill over at. The nearest recorded entry wins.

By frame rather than by index into the entry table, because the index is not stable: a
rebuild that adds one entry shifts every entry after it, and this demo picked jump entry 2
when that meant frame 87 and would have silently got frame 78 after the next build. A frame
is what the choice is actually about.

Manual, and the demo's decision rather than the selector's. Fixing it is also what lets an
approach be solved once: the entry decides the clip frame, the frame decides where the
reference's obstacle sits relative to it, and that is what the arrival pose is solved
against. An entry chosen per step would be a pose that moves.

The jump is entered near frame 86, measured rather than assumed. Its entries run from frame
60 to 105, and the earliest is the one the selector rates easiest because it is the only one
standing still. It is also the start of the crouch, so a hand-over into it asks the bridge
for a whole crouch pose it cannot reproduce, and the take-off then happens in the wrong
place. Over one hurdle, as arrival gap and worst joint, and whether the jump survived:

    f068  0.135 m  0.44 rad  fell
    f076  0.135 m  0.46 rad  fell
    f087  0.039 m  0.38 rad  cleared
    f096  0.028 m  0.64 rad  cleared

Reachability is not the same question as which phase of a skill the robot should be dropped
into, which is the reason this table is hand written rather than ranked by effort."""


def entry_of(skill: str, entries: tuple[Entry, ...]) -> int:
  """Index of the entry this demo uses: the one nearest the frame ENTRIES asks for.

  Nearest rather than exact, so rebuilding the entry table moves the choice by a frame or
  two instead of breaking it or, worse, quietly pointing it somewhere else.
  """
  if not entries:
    raise SystemExit(
      f"'{skill}' has no entry table rows, so nothing can be aimed at it."
    )
  wanted = ENTRIES.get(skill)
  if wanted is None:
    return 0
  return min(range(len(entries)), key=lambda i: abs(entries[i].frame - wanted))


##
# Clips.
##


def motion_command(env: ManagerBasedRlEnv, skill: str) -> JumpCommand | None:
  """That skill's reference command, at the name the arena gave it, or None if it has none.

  None rather than a raise. The walk drives without a clip and every caller here has an
  answer for one: nothing to rewind, nothing to run out.
  """
  try:
    command = env.command_manager.get_term(command_name(skill, MOTION))
  except (KeyError, ValueError):
    return None
  return command if isinstance(command, JumpCommand) else None


def clip_left(env: ManagerBasedRlEnv, skill: str) -> int:
  """Control steps of reference that skill has left to play. Zero if it has none."""
  command = motion_command(env, skill)
  if command is None:
    return 0
  lengths = command.motion.time_step_total_per_motion[command.motion_ids]
  return int((lengths - command.time_steps).clamp(min=0).max())


##
# The rules.
##


@dataclass(frozen=True)
class Rule:
  """One row of the decision table: an obstacle of this kind asks for that skill."""

  kind: str
  skill: str
  why: str

  def row(self) -> str:
    return f"| {self.kind} | {self.skill} | {self.why} |"


RULES: tuple[Rule, ...] = (
  Rule(BOX, "climb", "a block, so get on top and down the far side"),
  Rule(HURDLE, "jump", "a bar, so clear it in one"),
)
"""The decision table. First match wins, and it is consulted once, in plan.

An obstacle no rule covers is not quietly rounded into the nearest skill: plan reports it
and the demo refuses to start, which is what a missing skill should look like."""

RULE_HEADER = ("| kind | skill | why |", "|---|---|---|")


@dataclass(frozen=True)
class Step:
  """One obstacle and the rule that claimed it. rule is None when none did."""

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
  """The rules, the course, then the skills they pick. No coordinates: those need the
  arena, and this is what --dry prints before one exists. See Controller.lines."""
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
# The actions.
##


class Action:
  """One thing the robot does, start to finish.

  The loop calls done once per step and drive once per step, and start and finish exactly
  once each at the boundaries. Nothing here holds the robot's state: an action holds what
  it was built with, plus whatever latch it needs to know it is over.
  """

  index: int = 0
  """Which obstacle this action belongs to. What the skills are told to look at."""
  why: str = ""

  def start(self, ctx: Controller) -> torch.Tensor:
    """Take the world. Returns the observation the first drive should read."""
    ctx.focus.index = self.index
    return fresh_obs(ctx.env)

  def done(self, ctx: Controller) -> bool:
    raise NotImplementedError

  def drive(self, ctx: Controller, obs) -> torch.Tensor:
    raise NotImplementedError

  def finish(self, ctx: Controller) -> None:
    """Hand the world back. The default has nothing to say."""

  @property
  def label(self) -> str:
    """Who owns the robot, by name. What the viewer shows."""
    raise NotImplementedError

  @property
  def clip(self) -> str | None:
    """Whose reference says what upright means right now. None for vertical."""
    return None

  def row(self, number: int) -> str:
    raise NotImplementedError


class GoTo(Action):
  """Walk to a point in the world and stop there, facing a given way.

  One heading command serving two objectives, blended by distance: the point while it is
  far, face once it is close. Aiming at a point rather than driving cross-track error to
  zero is what keeps it stable, since steering on the error alone asks the robot to face
  the line rather than converge on it.

  Forward speed is regulated on the distance left, not held. Held, the walk crosses the
  arrival radius in about twenty steps and parks past the point.
  """

  def __init__(
    self,
    point: torch.Tensor,
    face: float,
    radius: float,
    yaw_tolerance: float,
    speed: float,
    index: int,
    why: str,
  ) -> None:
    self.point, self.face = point, face
    self.radius, self.yaw_tolerance, self.speed = radius, yaw_tolerance, speed
    self.index, self.why = index, why

  def _delta(self, ctx: Controller) -> torch.Tensor:
    return self.point - ctx.here()[:, 0:2]

  def done(self, ctx: Controller) -> bool:
    if float(self._delta(ctx).norm(dim=-1).max()) > self.radius:
      return False
    off = wrap(yaw_of(ctx.here()[:, 3:7]) - self.face)
    return float(off.abs().max()) <= self.yaw_tolerance

  def drive(self, ctx: Controller, obs) -> torch.Tensor:
    cfg = ctx.settings.walk
    walk = ctx.pool[ctx.cruising]
    here = ctx.here()

    # Chase a point on the approach line rather than the mark itself. Driving straight at
    # the mark approaches it along a diagonal and leaves the whole cross-track error to be
    # killed at the end, sideways, which the walk barely does: the robot stalled after
    # every traversal with the mark half a metre square to the line and the forward command
    # reading zero. A carrot one lookahead ahead on the line is converged onto instead, and
    # the robot arrives already pointing down it
    travel = torch.tensor(
      [math.cos(self.face), math.sin(self.face)],
      device=here.device,
      dtype=here.dtype,
    )
    behind = float(((here[:, 0:2] - self.point) @ travel).mean())
    carrot = self.point + min(behind + cfg.lookahead, 0.0) * travel
    delta = carrot - here[:, 0:2]
    bearing = float(torch.atan2(delta[:, 1], delta[:, 0]).mean())

    # 1 a lookahead or more from the carrot, 0 on it. Turns the heading from "go get it"
    # into "hold what the next skill needs", and fades the sideways correction in as it
    # does.
    #
    # On the distance to the carrot rather than on how far back down the line the robot is,
    # because the second says nothing about cross-track. A robot level with the mark but
    # half a metre to the side of it read as arrived, squared up to face, and crabbed the
    # rest of the way at the sideways cap: forward zero, x not moving, which is what the
    # stall looked like from outside. Measured this way it turns, walks the half metre, and
    # squares up at the end
    gap = float(delta.norm(dim=-1).mean())
    blend = min(max(gap / max(cfg.lookahead, 1e-6), 0.0), 1.0)
    heading = self.face + blend * float(wrap(torch.tensor([bearing - self.face]))[0])

    # Split the error in the frame the robot is being pointed at, not in the approach
    # frame. Measured against face, an error square to the line reads as no forward
    # distance at all
    cos, sin = math.cos(heading), math.sin(heading)
    ahead = float((delta[:, 0] * cos + delta[:, 1] * sin).mean())
    beside = float((-delta[:, 0] * sin + delta[:, 1] * cos).mean())
    forward = min(max(cfg.approach_gain * ahead, -cfg.reverse_limit), self.speed)
    lateral = min(max(cfg.lateral_gain * beside, -cfg.lateral_limit), cfg.lateral_limit)
    walk.tell(forward=forward, lateral=lateral * (1.0 - blend), heading=heading)
    walk.condition(ctx.env)
    return walk(obs)

  @property
  def label(self) -> str:
    return CRUISE_SKILL.name

  def row(self, number: int) -> str:
    return (
      f"| {number} | go_to | {CRUISE_SKILL.name} | "
      f"{float(self.point[0, 0]):.2f}, {float(self.point[0, 1]):+.2f} at "
      f"{math.degrees(self.face):+.0f} deg | within {self.radius:.2f} m | {self.why} |"
    )


class Cross(Action):
  """Run the bridge from wherever the walk stopped to the pose a skill starts from.

  A fixed window, not a solved one. The walk has already put the robot near and stopped, so
  what is left is the same short crossing every time, and short crossings out of slow
  states are where the bridge measures best. window_s has to sit inside the range the
  bridge trained on: see config.yml.
  """

  def __init__(
    self,
    skill: str,
    at: Approach,
    reach: Reach,
    duration_s: float,
    index: int,
    why: str,
  ) -> None:
    self.skill, self.at, self.reach = skill, at, reach
    self.duration_s, self.index, self.why = duration_s, index, why

  def start(self, ctx: Controller) -> torch.Tensor:
    ctx.focus.index = self.index
    ctx.bridge.aim(
      ctx.pool[self.skill],
      self.reach,
      ctx.here(),
      self.at.xy,
      self.at.yaw,
      self.duration_s,
      anchor_yaw=self.at.anchor_yaw,
      frame=self.at.frame,
      tolerances=ctx.tolerances.for_entry(self.skill, self.at.frame),
    )
    ctx.log(
      Decision(
        tick=ctx.tick,
        leaving=ctx.leaving,
        entering=self.skill,
        entry=self.reach.entry.name,
        effort=self.reach.effort,
        binding=self.reach.binding,
        duration_s=self.duration_s,
        why=self.why,
      )
    )
    return fresh_obs(ctx.env)

  def done(self, ctx: Controller) -> bool:
    return ctx.bridge.done

  def drive(self, ctx: Controller, obs) -> torch.Tensor:
    if ctx.settings.approach.count_in:
      self.count_in(ctx)
    return ctx.bridge(obs)

  def count_in(self, ctx: Controller) -> None:
    """Hold the entering clip short of its entry and march it in as the crossing runs.

    The observation seam. Left alone the clip plays on through the window and is rewound at
    the hand-over, which steps 29 reference angles, 29 reference rates, the phase and the
    anchor error at once. Marched in, the entering skill reads a reference that has been
    approaching it and the rewind afterwards has nothing left to do.

    The frames still owed come off the window the crossing was given rather than off a clock
    kept here, so a bridge that arrives early leaves the reference a few frames short and
    the rewind closes that instead of forty. Nothing to do for a skill with no reference.
    """
    command = ctx.bridge.command
    left = int((command.window_steps - command.step).clamp(min=0).max())
    entry_resume.rewind(
      ctx.env, max(self.at.frame - left, 0), command_name(self.skill, MOTION)
    )

  def finish(self, ctx: Controller) -> None:
    """Say how close the crossing got, whichever way it ended.

    Handing over on a crossing that ran out of patience is still what happens, because a
    demo has nowhere better to put the robot, but it is the line to look for when the
    traversal goes wrong a second later.
    """
    gap, angle, worst = ctx.bridge.arrival(ctx.here())
    verdict = "arrived" if ctx.bridge.succeeded else "gave up"
    print(
      f"  {verdict}: {gap:.3f} m, {math.degrees(angle):.1f} degrees and {worst:.2f} rad "
      f"at the worst joint off the pose {self.skill} was promised, "
      f"best score {float(ctx.bridge.command.score.min()):.3f}"
    )

  @property
  def label(self) -> str:
    return BRIDGE_GROUP

  def row(self, number: int) -> str:
    return (
      f"| {number} | cross | {BRIDGE_GROUP} | {self.at.row()} | "
      f"{self.duration_s:.2f} s | {self.why} |"
    )


class Traverse(Action):
  """Run a traversal skill until the robot is back on the ground standing.

  The latch is the whole action. The robot is already standing when this starts, so it has
  to leave the ground before coming back counts. The lowest foot is what says so, which for
  the climb means both feet on the box rather than one.

  Standing, not merely grounded: a jump handed over badly lands folded at 0.99 rad and goes
  on folding with its feet on the floor throughout, which is not a body the walk can start
  from. Patience counts from the end of the skill's own clip, so a climb gets all 455 frames
  rather than a budget that runs out on top of the box.
  """

  def __init__(self, skill: str, at: Approach, index: int, why: str) -> None:
    self.skill, self.at, self.index, self.why = skill, at, index, why
    self.airborne = False
    self.grounded = 0
    self.until = 0

  def start(self, ctx: Controller) -> torch.Tensor:
    ctx.focus.index = self.index
    # The clip has been playing throughout the crossing, so it is most of a second ahead of
    # the robot by now. Put the clock back before the skill reads it, or the policy takes
    # over chasing a reference it has not caught up with. For the climb that is a reference
    # already standing on top of the box.
    #
    # The clock only. Re-anchoring here would pin the clip to wherever the robot actually
    # arrived, erasing the arrival error and dragging the climb's own obstacle off the real
    # one. The placement is right already
    entry_resume.rewind(ctx.env, self.at.frame, command_name(self.skill, MOTION))
    self.airborne, self.grounded = False, 0
    self.until = (
      ctx.tick + clip_left(ctx.env, self.skill) + ctx.settings.traverse.patience
    )
    return fresh_obs(ctx.env)

  def done(self, ctx: Controller) -> bool:
    cfg = ctx.settings.traverse
    lowest = ctx.foot_height()
    if lowest > cfg.lift_height:
      self.airborne = True
    standing = (
      self.airborne
      and lowest < cfg.land_height
      and float(lean_of(ctx.here()[:, 3:7]).max()) < cfg.stand_angle
    )
    self.grounded = self.grounded + 1 if standing else 0
    return self.grounded >= cfg.settle_steps or ctx.tick >= self.until

  def drive(self, ctx: Controller, obs) -> torch.Tensor:
    skill = ctx.pool[self.skill]
    skill.condition(ctx.env)
    return skill(obs)

  def finish(self, ctx: Controller) -> None:
    ctx.cleared.append(self.index)
    verdict = "down and standing" if self.grounded else "out of patience"
    print(f"  obstacle {self.index} {verdict} after {ctx.tick} steps, back to the walk")

  @property
  def label(self) -> str:
    return self.skill

  @property
  def clip(self) -> str | None:
    return self.skill

  def row(self, number: int) -> str:
    return (
      f"| {number} | {self.skill} | {self.skill} | obstacle {self.index} | "
      f"back on the ground | {self.why} |"
    )


ACTION_HEADER = (
  "| # | action | drives | where | until | why |",
  "|---|---|---|---|---|---|",
)


##
# The loop.
##


class Controller:
  """Compiles the plan, then walks down it."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    pool: SkillPool,
    bridge: Bridge,
    course: Course,
    steps: tuple[Step, ...],
    focus: Focus,
    tolerances: ToleranceOverrides | None = None,
  ) -> None:
    self.env, self.pool, self.bridge = env, pool, bridge
    self.course, self.steps, self.focus = course, steps, focus
    self.tolerances = tolerances or ToleranceOverrides()
    """Channels of the arrival tolerance the command line pinned. Whatever it leaves alone
    comes from tests/entry_tolerances.py, by skill and frame."""
    self.settings: Settings = course.settings
    self.robot: Entity = env.scene[ROBOT]
    self.names = obstacle_names(course)
    self.feet, _ = self.robot.find_bodies(list(FOOT_BODIES))

    self.cruising = CRUISE_SKILL.name
    """What drives between obstacles. One skill for the whole run, so a reset keeps it."""
    self.started = False
    """Whether a course has already been run. Only a second start is worth announcing."""

    self.reset()

  def reset(self) -> None:
    """Start the course over, from the first action.

    What the viewer's reset button reaches. Without it the plan outlives the world: the
    robot goes back to the start line and the cursor does not.

    The plan is compiled here rather than in the constructor so a reset re-solves it. That
    re-anchors the clips against a world that has moved and clears every latch an action
    holds, without anything having to remember to.
    """
    self.tick = 0
    self.cursor = 0
    self.leaving = self.cruising
    """Who owned the robot before the current action. Only the decisions table reads it."""
    self.decisions: list[Decision] = []
    self.cleared: list[int] = []
    self.done = False
    self.fell = False

    self.parting = torch.zeros_like(self.env.action_manager.action)
    """The last action commanded. What a ramp out of a switch starts from."""
    self.fading = 0
    """Ramped steps still owed after the most recent switch. See fade."""

    self.focus.index = 0
    self.bridge.reset()
    self.pool[self.cruising].enter(self.env, *self.here_pose())
    self.actions = self.compile()
    self.actions[0].start(self)

    if self.started:
      print("")
      print(f"reset: back to the start of the course, '{self.driving}' driving")
    self.started = True

  ##
  # Compiling.
  ##

  def compile(self) -> tuple[Action, ...]:
    """The course, as a flat list of actions. Three per obstacle plus a run out.

    The only expensive part is the approach solve, which anchors a clip a few times per
    obstacle. Done once, here, because the entry is fixed and so the answer is a constant
    of the obstacle rather than of the robot.
    """
    cfg = self.settings.approach
    out: list[Action] = []
    for step in self.steps:
      assert step.rule is not None, "unsolved steps are refused before this is reached"
      skill = step.rule.skill
      pos, yaw = self.obstacle_pose(step.index)
      reach = self.reach_for(skill)
      at = approach.solve(
        self.env, skill, step.obstacle, pos, yaw, reach.entry.frame, cfg
      )
      out.append(
        GoTo(
          point=at.hold_xy,
          face=at.face,
          radius=cfg.arrive_radius,
          yaw_tolerance=cfg.yaw_tolerance,
          speed=self.settings.walk.approach_speed,
          index=step.index,
          why=f"up to obstacle {step.index}",
        )
      )
      out.append(Cross(skill, at, reach, cfg.window_s, step.index, step.rule.why))
      out.append(Traverse(skill, at, step.index, step.rule.why))

    # The run out. Down the lane, past the last obstacle, so a course that was cleared ends
    # with the robot walking rather than standing on the spot it landed
    last = self.steps[-1].index if self.steps else 0
    out.append(
      GoTo(
        point=self.point(self.course.length, 0.0),
        face=0.0,
        radius=cfg.arrive_radius,
        yaw_tolerance=math.pi,
        speed=self.settings.walk.cruise_speed,
        index=last,
        why="course clear, down the lane",
      )
    )
    return tuple(out)

  def reach_for(self, skill: str) -> Reach:
    """The entry this demo hands over into, and what reaching it costs from the start line.

    The entry is what matters and it is fixed by ENTRIES. The effort that comes with it is
    only reported, because the window is fixed too: see Cross.
    """
    entries = self.pool[skill].entries
    return self.pool[skill].reach(
      as_numpy(self.here()),
      self.settings.approach.window_s,
      entry_of(skill, entries),
    )

  def point(self, x: float, y: float) -> torch.Tensor:
    """A world point, shaped the way an action wants it."""
    return torch.tensor([[x, y]], device=self.env.device, dtype=torch.float32).expand(
      self.env.num_envs, 2
    )

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
    """One obstacle's world position and yaw, read off the scene rather than off the
    course, so the controller cannot be passing because it was handed the answer."""
    box: Entity = self.env.scene[self.names[index]]
    return box.data.root_link_pos_w, yaw_of(box.data.root_link_quat_w)

  def foot_height(self) -> float:
    """The lowest foot, in metres off the floor. What says a traversal is over.

    The minimum of the two, so a swing foot in a normal stride does not read as a take-off
    and a climb is not called started until both feet are on the box.
    """
    z = self.robot.data.body_link_pos_w[:, self.feet, 2]
    return float((z - self.env.scene.env_origins[:, 2:3]).min(dim=-1).values.max())

  def upright(self) -> torch.Tensor:
    """What counts as upright right now, as a lean off vertical in radians.

    Vertical under the walk and the bridge, the traversal's own reference under a traversal.
    A climb mount doubles the body over to 78 degrees against a limit of 60, so measured
    against vertical this called a textbook climb a fall every time. Measured against the
    reference the check stays live: the robot is down when it leans further than the motion
    it is copying, by the margin the walk gets.
    """
    zero = torch.zeros(self.env.num_envs, device=self.env.device)
    clip = self.actions[self.cursor].clip
    if clip is None:
      return zero
    command = motion_command(self.env, clip)
    return zero if command is None else lean_of(command.body_quat_w[:, 0])

  def down(self) -> bool:
    """Whether the robot is on the floor.

    Two checks, because either alone misses a case. A robot face down on top of a box is
    well above the floor threshold, and one sitting in a heap on the ground is still
    upright. Height catches the fall, tilt catches the sprawl.
    """
    here = self.here()
    if float(here[:, 2].min()) < self.settings.end.fall_height:
      return True
    lean = lean_of(here[:, 3:7]) - self.upright()
    return bool(float(lean.max()) > self.settings.end.tip_angle)

  ##
  # Running.
  ##

  @property
  def driving(self) -> str:
    """Whoever owns the world right now, by name. What the viewer shows."""
    return self.actions[self.cursor].label

  def log(self, decision: Decision) -> None:
    self.decisions.append(decision)
    print(f"  {decision.row()}")

  def fade(self, action: torch.Tensor) -> torch.Tensor:
    """Ramp out of the parting policy's last action over the steps a switch still owes.

    The action seam, and it is spent at every switch rather than only at the hand-over: each
    one is a change of driver and each one puts a step into the joint targets. What a ramp
    cannot do is make the two policies agree, because they were fitted separately and can
    agree about the state while disagreeing about what to command in it. It spreads that
    disagreement over several control steps so the targets stay continuous.

    See config.yml for the count, and tests/stage.py for where it was measured.
    """
    steps = self.settings.approach.blend_steps
    if self.fading <= 0 or steps <= 0:
      return action
    weight = float(steps - self.fading + 1) / float(steps)
    self.fading -= 1
    return weight * action + (1.0 - weight) * self.parting

  def advance(self, obs):
    """Finish the current action and start the next, or end the run.

    The name of the action being left is recorded before the cursor moves. Reading it after
    is how the decisions table came to say every hand-over was out of the bridge: `driving`
    answers for whoever the cursor points at, and by the time the next action starts that
    is already the next action.
    """
    leaving = self.actions[self.cursor]
    leaving.finish(self)
    if self.cursor + 1 >= len(self.actions):
      self.done = True
      return obs
    self.leaving = leaving.label
    self.cursor += 1
    self.fading = self.settings.approach.blend_steps
    return self.actions[self.cursor].start(self)

  @torch.no_grad()
  def __call__(self, obs):
    if not self.done:
      if self.down():
        self.fell, self.done = True, True
      elif self.actions[self.cursor].done(self):
        obs = self.advance(obs)
    self.tick += 1
    action = self.fade(self.actions[self.cursor].drive(self, obs))
    # What the next ramp starts from, which is what was commanded rather than what the
    # policy asked for: a switch during a ramp has to leave the joints where they are
    self.parting = action.clone()
    return action

  ##
  # Printing.
  ##

  def lines(self) -> list[str]:
    """The compiled plan, with the coordinates the solve produced."""
    return [
      "",
      f"plan: {len(self.actions)} actions over {len(self.steps)} obstacles",
      *ACTION_HEADER,
      *(action.row(i) for i, action in enumerate(self.actions)),
    ]

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
