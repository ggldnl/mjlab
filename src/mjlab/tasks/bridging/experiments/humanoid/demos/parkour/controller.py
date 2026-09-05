"""What decides which skill runs next, and when the switch fires.

Two halves, kept apart on purpose:

    RULES     which skill an obstacle asks for. A table, resolved and printed before the
              robot moves, and the only place a skill choice is made
    Runner    when to hand over, to which state, over how long a window. The phase machine

The split is the point of the demo. A rule reads two numbers off an obstacle and names a
skill, so the whole decision is a table somebody can read, disagree with and edit; the
runner never chooses a skill, it only carries out what the table said. Which is what makes
the controller auditable in the sense worth claiming: not that it is simple, but that the
choice and the execution cannot be confused for one another.

Nothing here decides where a skill can be joined. That is measured, off the selector's entry
table, and asked for per hand-over with `nearest`. The rule says vault, the selector says
which vault state is easiest to reach from where the robot is at that instant, and the
bridge goes there.

Four phases per obstacle, and two of them are bridges:

    cruise     the locomotion skill drives, down the lane
    bridge     out, to the traversal skill's entry, landing at the take-off point
    traverse   the traversal skill drives, over the box
    bridge     back, to the locomotion skill's entry, so the robot resumes running

The return bridge is the half `tests.stage` has no need of, because a couple ends after one
hand-over. On a course it is worth as much as the outbound one: a robot that clears a wall
and cannot get back to running has not cleared anything, it has stopped on the far side.

Run

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.demos.parkour.controller
    uv run python -m ...demos.parkour.controller --seed 3 --count 6

    # the plan alone, no simulation, no checkpoints
    uv run python -m ...demos.parkour.controller --dry True

    # headless
    uv run python -m ...demos.parkour.controller --viewer none
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.arena import (
  course_env_cfg,
  obstacle_names,
)
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.course import (
  Course,
  Obstacle,
)
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.course import (
  generate as generate_course,
)
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.pool import SkillPool
from mjlab.tasks.bridging.experiments.humanoid.selector import Reach, nearest
from mjlab.tasks.bridging.experiments.humanoid.tests.actors import JUMP, RUN, WALK
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import (
  ARRIVE_SLACK,
  BRIDGE_GROUP,
  DEFAULT_DURATION_S,
  PROBE_S,
  ROBOT,
  Actor,
  Aimed,
  aim,
  crossing_time,
  defaults,
  facing,
  fresh_obs,
  state,
)

##
# The roster.
##

STEP = Actor("step", "Mjlab-G1-Step")
VAULT = Actor("vault", "Mjlab-G1-Vault")
CLIMB = Actor("climb", "Mjlab-G1-Climb")
"""The three traversal skills this demo needs and the repository does not have yet.

Declared here rather than in `tests.actors`, which is where a skill's quirks belong, because
that module imports each skill's task id from the skill's own package and those packages do
not exist. Moving these four lines across is the last step of adding one, not the first.

An `Actor` is a declaration and costs nothing to write down, so the demo is complete and
fails at the moment it reaches for a checkpoint, naming the one it could not find. That is
the right failure: a scaffold that cannot be built until every skill is trained cannot be
used to decide which skills to train.
"""

CRUISE_SKILL = RUN
"""What drives between obstacles. The run rather than the walk, because the interesting
hand-over is the one that has to shed real momentum, and a demo cruising at walking pace
never asks the bridge for anything."""

ROSTER: tuple[Actor, ...] = (WALK, RUN, JUMP, STEP, VAULT, CLIMB)
"""Everything the demo loads. The walk is in it and unused by the default rules, so a course
can be cruised at walking pace by changing one constant rather than by editing the roster."""


##
# The rules.
##


@dataclass(frozen=True)
class Rule:
  """One row of the decision table: an obstacle this shape asks for that skill."""

  skill: str
  height: tuple[float, float]
  """Metres, low inclusive, high exclusive."""
  depth: tuple[float, float]
  """Metres, same convention."""
  takeoff: float
  """How far before the near face the skill needs the robot when it takes over, in metres.

  The one number here the bridge actually consumes: it is where the crossing has to land, so
  it decides the window rather than merely describing the skill. Too short and the robot is
  handed the skill already under the box; too long and the skill covers the difference by
  locomoting, which is the leaving skill's job and not its own."""
  why: str
  """Why this shape asks for this skill. Printed with the plan."""

  def covers(self, obstacle: Obstacle) -> bool:
    return (
      self.height[0] <= obstacle.height < self.height[1]
      and self.depth[0] <= obstacle.depth < self.depth[1]
    )

  def row(self) -> str:
    return (
      f"| {self.skill} | {self.height[0]:.2f}-{self.height[1]:.2f} "
      f"| {self.depth[0]:.2f}-{self.depth[1]:.2f} | {self.takeoff:.2f} | {self.why} |"
    )


RULES: tuple[Rule, ...] = (
  Rule("jump", (0.0, 0.45), (0.80, 99.0), 1.20, "low and deep, so leap the span"),
  Rule(
    "step", (0.0, 0.45), (0.00, 0.80), 0.55, "low and shallow, one stride clears it"
  ),
  Rule("vault", (0.45, 0.80), (0.00, 0.90), 0.75, "hip height, a hand down and over"),
  Rule("climb", (0.80, 1.45), (0.00, 1.20), 0.60, "above hip, get on top then off"),
)
"""The decision table. First match wins, so order is meaningful: the two low rules are
separated by depth alone and the deep one is asked first.

Height and depth and nothing else. Not the family name, though the course carries one,
because a rule reading a label would be a rule agreeing with the generator by construction
and would tell you nothing about whether the robot can read the world. These read what a
perception stack would have to produce.

Bands rather than thresholds so a gap in the table is visible as a gap. An obstacle no rule
covers is not quietly rounded into the nearest skill: `plan` reports it and the demo refuses
to start, which is the honest behaviour and also the interesting one."""


RULE_HEADER = (
  "| skill | height | depth | takeoff | why |",
  "|---|---|---|---|---|",
)


@dataclass(frozen=True)
class Step:
  """One obstacle and the rule that claimed it. `rule` is None when none did."""

  index: int
  obstacle: Obstacle
  rule: Rule | None

  def row(self) -> str:
    named = self.rule.skill if self.rule else "NO RULE"
    why = self.rule.why if self.rule else "nothing in the table covers this shape"
    return (
      f"| {self.index} | {self.obstacle.family} | {self.obstacle.height:.2f} "
      f"| {self.obstacle.depth:.2f} | {named} | {why} |"
    )


PLAN_HEADER = (
  "| # | obstacle | height | depth | skill | why |",
  "|---|---|---|---|---|---|",
)


def plan(course: Course, rules: tuple[Rule, ...] = RULES) -> tuple[Step, ...]:
  """Resolve every obstacle to a skill, before anything is built.

  Runs on the course alone. No environment, no checkpoints, no selector: this is the
  symbolic half, and keeping it callable on its own is what lets `--dry` answer whether a
  course is solvable in the time it takes to print a table.
  """
  return tuple(
    Step(index=i, obstacle=o, rule=next((r for r in rules if r.covers(o)), None))
    for i, o in enumerate(course)
  )


def unsolved(steps: tuple[Step, ...]) -> tuple[Step, ...]:
  """The obstacles no rule claimed. Empty when the course is solvable."""
  return tuple(step for step in steps if step.rule is None)


def plan_lines(course: Course, steps: tuple[Step, ...]) -> list[str]:
  """The rules, then the plan they produce. What to print before a run."""
  out = ["rules:", *RULE_HEADER, *(rule.row() for rule in RULES), ""]
  out += [*course.lines(), ""]
  out += ["plan:", *PLAN_HEADER, *(step.row() for step in steps)]
  blocked = unsolved(steps)
  if blocked:
    out.append("")
    out.append(
      f"no plan: {len(blocked)} obstacle(s) no rule covers, at "
      f"{', '.join(f'{s.obstacle.height:.2f} m' for s in blocked)}. Add a rule, or a skill."
    )
  return out


##
# Reading the world.
##


def obstacle_pos(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  """Where one obstacle actually is, in world coordinates. (N, 3).

  Read off the scene rather than off the course, so a controller cannot be passing because
  it was handed the answer at build time. With `arena.lay_out_course` jittering, the course
  is a description of what was asked for and this is what is there.
  """
  box: Entity = env.scene[name]
  return box.data.root_link_pos_w


def takeoff_spot(
  env: ManagerBasedRlEnv, name: str, obstacle: Obstacle, takeoff: float
) -> torch.Tensor:
  """Where the traversal skill needs the robot standing when it takes over. (N, 3).

  On the lane, one take-off distance short of the near face. The same shape as
  `Actor.arrive`, and used the same way: handed to `aim` so the crossing is commanded to
  land there instead of wherever a decelerating body would drift to.

  Commanded here, where `tests.stage` leaves it off. That module measured the two placements
  against a ball, whose box is eight centimetres deep and sits under a foot the robot can
  shuffle: there, the bridge's own tracking error and the ballistic model's error traded off
  evenly and neither won. A take-off point does not trade off. Landing it late puts the
  robot under the box rather than in front of it, and no amount of good posture recovers a
  vault begun from the wrong side of a wall.
  """
  spot = obstacle_pos(env, name).clone()
  spot[:, 0] -= obstacle.depth / 2.0 + takeoff
  return spot


##
# The decisions a run made.
##


@dataclass(frozen=True)
class Decision:
  """One hand-over, and everything that went into it. The audit trail.

  Written when the switch fires, so it records what was decided rather than what happened.
  How it turned out is the arrival line printed at the hand-over.
  """

  tick: int
  leaving: str
  entering: str
  why: str
  entry: str
  effort: float
  binding: str
  duration_s: float

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

SETTLE_M = 1.2
"""Metres past an obstacle's far face before the return bridge opens.

Far enough that the traversal skill has landed and put a foot down. Opening the moment the
box is behind the robot would aim the return bridge at a body still in the air, whose state
is not one any locomotion entry sits near."""

TRAVERSE_PATIENCE = 200
"""Control steps a traversal gets before the run gives up on it.

A skill that never reaches the far side has failed, and a demo that waits forever for it
reads as a hang rather than as the result it is."""

FALL_HEIGHT = 0.4
"""Root height below which the robot is called down, in metres. Nothing resets it: a fall is
the result of the demo, so the run says so and stops."""


class Runner:
  """Drive a course: cruise, bridge out, traverse, bridge back, repeat."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    pool: SkillPool,
    course: Course,
    steps: tuple[Step, ...],
    commanded: bool = True,
  ) -> None:
    self.env, self.pool, self.course, self.steps = env, pool, course, steps
    self.commanded = commanded
    self.robot: Entity = env.scene[ROBOT]
    self.names = obstacle_names(course)

    command = env.command_manager.get_term("bridge")
    assert isinstance(command, Aimed)
    self.command = command

    self.knobs: dict[str, dict[str, float]] = {
      name: defaults(pool.actor(name)) for name in pool.skills
    }
    """What each skill is currently being told. `condition` reads it, and only for whoever
    is driving: two skills can share a command term, which the walk and the run do, and a
    term written by both at once holds whichever wrote last."""

    self.phase = CRUISE
    self.index = 0
    self.tick = 0
    self.cruising = CRUISE_SKILL.name
    self.entering = self.cruising
    """Who the open bridge is handing to. Read while `phase` is BRIDGE and stale otherwise."""
    self.traverse_until = 0
    self.decisions: list[Decision] = []
    self.cleared: list[int] = []
    self.done = False
    self.fell = False

    self.enter(pool.actor(self.cruising))

  ##
  # Whoever is driving.
  ##

  @property
  def driving(self) -> str:
    """Whoever owns the world right now, by name. What the viewer shows."""
    if self.phase == BRIDGE:
      return BRIDGE_GROUP
    return self.cruising if self.phase == CRUISE else self.entering

  @property
  def step(self) -> Step | None:
    """The obstacle being worked on, or None once the course is finished."""
    return self.steps[self.index] if self.index < len(self.steps) else None

  def enter(self, actor: Actor, pos=None, heading=None, frame: int = 0) -> None:
    """Hand the world to a skill that is about to take over.

    Defaults to where the robot is, which is right for a skill starting from a reset. `aim`
    is what passes a placement, because it knows where the crossing will land and a
    reference pinned to where the robot is now would slide onto the arrival and erase the
    error the hand-over is being measured on.
    """
    if actor.enter is None:
      return
    here = state(self.robot)
    actor.enter(
      self.env,
      here[:, 0:3] if pos is None else pos,
      here[:, 3:7] if heading is None else heading,
      frame,
      self.knobs.get(actor.name) or defaults(actor),
    )

  def condition(self) -> None:
    """Write the driving skill's settings into whatever it reads, every step.

    Only the driver's. A skill's conditioning is applied while it owns the world and at no
    other time, which is what lets the walk and the run share one velocity term without
    writing over each other.
    """
    if self.phase == BRIDGE:
      return
    actor = self.pool.actor(self.driving)
    if actor.condition is not None:
      actor.condition(self.env, self.knobs[actor.name])

  ##
  # Choosing.
  ##

  def reach(self, skill: str, seconds: float) -> Reach:
    """The entry of `skill` easiest to reach from where the robot is now.

    Asked every step while a switch is pending, so what the bridge aims at is whatever is
    easiest at the instant it fires rather than whatever was easiest when the approach
    began.
    """
    here = state(self.robot)[0].detach().cpu().numpy()
    return nearest(self.pool.table, skill, here, seconds)[0]

  def window(self, reach: Reach, solved: float | None) -> float:
    """How long the bridge gets, in seconds.

    A window solved from the geometry wins, because it is the one that lands the crossing
    where the entering skill needs the robot. Otherwise the entry's own floor: effort scales
    with 1/seconds, so PROBE_S * effort is the window at which this entry costs exactly one,
    a change as fast as any recorded skill performed. That is a floor and not a set point,
    hence the max against the default.

    Clamped to what the bridge trained on either way. Outside that range it is being asked a
    question it was never shown.
    """
    low, high = self.command.cfg.duration_s_range
    if solved is not None:
      return float(min(max(solved, low), high))
    return float(min(max(PROBE_S * reach.effort, DEFAULT_DURATION_S, low), high))

  ##
  # The switches.
  ##

  def approach(self) -> tuple[Reach, float] | None:
    """Whether to start crossing to the traversal skill on this step.

    Solved rather than waited for. `crossing_time` says how long a window would have to be
    to land the robot at the take-off point, and the switch fires as soon as that is a
    window the bridge was trained on. A fixed window could only wait for the world to drift
    into agreement with it, which on a course means the robot walks past its own take-off
    point whenever the approach did not happen to line up.

    The residual keeps that honest. A crossing travels along one line, so a duration decides
    how far and never which way, and a take-off point off that line stays off it. Firing on
    the duration alone would hand over on time to the wrong place.
    """
    step = self.step
    if step is None or step.rule is None:
      return None
    seconds_needed = self.command.cfg.duration_s_range[1]
    reach = self.reach(step.rule.skill, seconds_needed)

    here = state(self.robot)
    entry = torch.as_tensor(
      reach.entry.state[None], dtype=torch.float32, device=self.env.device
    )
    target = facing(entry, here)
    want = takeoff_spot(
      self.env, self.names[step.index], step.obstacle, step.rule.takeoff
    )[:, 0:2]

    seconds, residual = crossing_time(here, target, want)
    low, high = self.command.cfg.duration_s_range
    fits = (seconds >= low) & (seconds <= high) & (residual <= ARRIVE_SLACK)
    if not bool(fits.all()):
      return None
    return reach, float(seconds.min())

  def past(self) -> bool:
    """Whether the robot is clear of the obstacle it just traversed."""
    step = self.step
    if step is None:
      return True
    far = (
      obstacle_pos(self.env, self.names[step.index])[:, 0] + step.obstacle.depth / 2.0
    )
    return bool((state(self.robot)[:, 0] > far + SETTLE_M).all())

  ##
  # The transitions.
  ##

  def cross(self, entering: Actor, reach: Reach, duration_s: float, why: str, arrive):
    """Aim the bridge at one state off the entering skill's table and start the clock."""
    self.phase, self.entering = BRIDGE, entering.name
    here = state(self.robot)
    aim(
      self.env,
      self.command,
      entering,
      torch.as_tensor(
        reach.entry.state[None], dtype=torch.float32, device=self.env.device
      ),
      here,
      duration_s,
      reach.entry.frame,
      arrive,
      self.knobs.get(entering.name),
    )
    self.decisions.append(
      Decision(
        tick=self.tick,
        leaving=self.cruising if entering.name != self.cruising else self.entering,
        entering=entering.name,
        why=why,
        entry=reach.entry.name,
        effort=reach.effort,
        binding=reach.binding,
        duration_s=duration_s,
      )
    )
    print(f"  {self.decisions[-1].row()}")
    return fresh_obs(self.env)

  def hand_over(self):
    """The window closed. Whoever it was aimed at takes over."""
    if self.entering == self.cruising:
      self.phase = CRUISE
      self.index += 1
      self.done = self.index >= len(self.steps)
    else:
      self.phase = TRAVERSE
      self.traverse_until = self.tick + TRAVERSE_PATIENCE
    return fresh_obs(self.env)

  ##
  # The loop.
  ##

  @torch.no_grad()
  def __call__(self, obs):
    self.condition()
    self.fell = bool((state(self.robot)[:, 2] < FALL_HEIGHT).any())
    if self.fell:
      self.done = True

    if not self.done:
      if self.phase == CRUISE:
        found = self.approach()
        if found is not None:
          reach, solved = found
          step = self.step
          assert step is not None and step.rule is not None
          rule = step.rule
          # Bound as defaults rather than captured, so the closure holds this obstacle
          # rather than whichever one `self.index` has reached by the time it is called.
          # `commanded` decides whether there is a callable at all: `aim` reads None as
          # "predict where the crossing lands", and one returning None is not that
          spot = (
            (
              lambda env, n=self.names[step.index], o=step.obstacle, t=rule.takeoff: (
                takeoff_spot(env, n, o, t)
              )
            )
            if self.commanded
            else None
          )
          obs = self.cross(
            self.pool.actor(rule.skill),
            reach,
            self.window(reach, solved),
            rule.why,
            spot,
          )
      elif self.phase == BRIDGE:
        if bool((self.command.step >= self.command.deadline).all()):
          obs = self.hand_over()
      elif self.phase == TRAVERSE:
        if self.past() or self.tick >= self.traverse_until:
          step = self.step
          assert step is not None
          self.cleared.append(step.index)
          reach = self.reach(self.cruising, DEFAULT_DURATION_S)
          obs = self.cross(
            self.pool.actor(self.cruising),
            reach,
            self.window(reach, None),
            "obstacle behind, back to the lane",
            None,
          )

    self.tick += 1
    return self.pool[self.driving](obs)

  def report(self) -> list[str]:
    """What the run decided and how far it got. The audit trail, after the fact."""
    out = ["", "decisions:", *DECISION_HEADER, *(d.row() for d in self.decisions)]
    out.append("")
    out.append(
      f"cleared {len(self.cleared)} of {len(self.steps)} obstacles in {self.tick} steps"
      + (", then fell" if self.fell else "")
    )
    return out


##
# Running it.
##


@dataclass(frozen=True)
class Config:
  seed: int = 0
  """Which course to draw."""
  count: int = 5
  """How many obstacles."""
  jitter: float = 0.0
  """Metres each obstacle may slide along the lane, per environment. See arena."""
  commanded: bool = True
  """Command the take-off point, rather than letting the crossing land where a decelerating
  body would. See `takeoff_spot` for why this is on here and off in tests.stage."""
  dry: bool = False
  """Print the plan and stop. No simulation and no checkpoints."""
  viewer: Literal["viser", "none"] = "viser"
  patience: int = 4000
  """Control steps before a headless run gives up."""
  device: str | None = None
  seed_torch: int = 0
  scored: str | None = None
  """A skill whose reward terms the arena carries, so a hand-over into it can be scored."""


def main(cfg: Config) -> None:
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  course = generate_course(seed=cfg.seed, count=cfg.count)
  steps = plan(course)
  for line in plan_lines(course, steps):
    print(line)
  blocked = unsolved(steps)
  if blocked:
    raise SystemExit(
      "\nRefusing to start: the table does not cover every obstacle on this course."
    )
  if cfg.dry:
    return

  torch.manual_seed(cfg.seed_torch)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Before the simulation, because a missing checkpoint is the one thing this demo cannot
  # work around and finding out after the arena is up costs a minute per attempt
  for name, path in SkillPool.resolve(ROSTER).items():
    print(f"{name:<8} {path}")

  env = ManagerBasedRlEnv(
    cfg=course_env_cfg(ROSTER, course, scored=cfg.scored, jitter=cfg.jitter),
    device=device,
  )
  pool = SkillPool.load(ROSTER, env, device)
  for line in pool.lines():
    print(line)

  env.reset()
  runner = Runner(env, pool, course, steps, commanded=cfg.commanded)
  print("")
  print(f"running: {' '.join(DECISION_HEADER)}")

  if cfg.viewer == "none":
    obs = env.get_observations()
    for _ in range(cfg.patience):
      if runner.done:
        break
      obs, _, _, _, _ = env.step(runner(obs))
    else:
      print(f"\ngave up after {cfg.patience} steps in the '{runner.driving}' phase")
    for line in runner.report():
      print(line)
    env.close()
    return

  import viser

  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg
  from mjlab.viewer import ViserPlayViewer

  server = viser.ViserServer(label=f"parkour seed {cfg.seed}")
  wrapped = RslRlVecEnvWrapper(
    env, clip_actions=load_rl_cfg(CRUISE_SKILL.task).clip_actions
  )
  ViserPlayViewer(
    wrapped, runner, viser_server=server, info_provider=lambda _: runner.driving
  ).run()
  for line in runner.report():
    print(line)
  wrapped.close()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
