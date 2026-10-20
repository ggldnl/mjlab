"""Which entry points each robot gets.

One Spec per entry set: the task it runs, the command it holds while recorded, and the
half-open range of steps its entry points are sampled from. A tracker is always started at
its first frame and its steps are reference frames; any other skill counts steps from its
reset. Step 0 is the reset itself and is never recorded, so a range starts at 1, and at 2
for a tracker, whose clip advances once before the first state is recorded.

What plays the role of the phase decides the range:

    tracker     reference frames, ending before any contact with an object or take off
    periodic    one gait period once the gait has settled, so every foot phase is covered
    task        a narrow range at the start, before the object moves or the turn is under way

Run

1. Record and build every set of a robot with the ranges below.

    uv run python -m mjlab.tasks.bridging.selector --robot g1

2. Try another range for one set without editing the table.

    uv run python -m mjlab.tasks.bridging.selector --robot g1 --sets "('jump',)" --window "('jump', 1, 4, 3)"
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.bridging.config.g1.skills.climb import CLIMB_TASK_ID
from mjlab.tasks.bridging.config.g1.skills.jump import JUMP_TASK_ID
from mjlab.tasks.bridging.config.g1.skills.kick import KICK_TASK_ID
from mjlab.tasks.bridging.config.g1.skills.martial import MARTIAL_TASK_IDS
from mjlab.tasks.bridging.config.g1.skills.walk import (
  WALK_TASK_ID as G1_WALK_TASK_ID,
)
from mjlab.tasks.bridging.config.t1.skills.push import PUSH_TASK_ID as T1_PUSH_TASK_ID
from mjlab.tasks.bridging.config.t1.skills.push.command import PushCommandCfg
from mjlab.tasks.bridging.config.t1.skills.turn import TURN_TASK_ID as T1_TURN_TASK_ID
from mjlab.tasks.bridging.config.t1.skills.turn.command import TurnCommandCfg
from mjlab.tasks.bridging.config.t1.skills.walk import (
  DISTANCE_WALK_TASK_ID as T1_WALK_TASK_ID,
)
from mjlab.tasks.bridging.config.t1.skills.walk.distance import WalkDistanceCommandCfg
from mjlab.tasks.bridging.selector import Window
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

Condition = Callable[[ManagerBasedRlEnvCfg], None]


@dataclass(frozen=True)
class Spec:
  """One entry set: a task, a fixed command, and the steps its entry points come from."""

  task: str
  start: int
  stop: int
  samples: int
  condition: Condition | None = None
  """Edits the environment config so the command stays fixed. None for a clip tracker."""

  def __post_init__(self) -> None:
    if self.start < 1:
      raise ValueError("Step 0 is the reset and is never recorded, start at 1")
    Window(self.start, self.stop, self.samples)

  @property
  def window(self) -> Window:
    return Window(self.start, self.stop, self.samples)


def walk_at(speed: float) -> Condition:
  """Hold a forward velocity command, no sidestep and no turn. Zero is standing."""

  def condition(cfg: ManagerBasedRlEnvCfg) -> None:
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = (speed, speed)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    twist.heading_command = False
    twist.ranges.heading = None
    twist.rel_heading_envs = twist.rel_standing_envs = 0.0
    twist.rel_world_envs = twist.rel_forward_envs = 0.0
    twist.init_velocity_prob = 0.0

  return condition


def turn_by(angle: float) -> Condition:
  """Turn in place by a fixed angle at the fastest rate, the way the demos ask for it."""

  def condition(cfg: ManagerBasedRlEnvCfg) -> None:
    turn = cfg.commands["twist"]
    assert isinstance(turn, TurnCommandCfg)
    turn.angle_range = (angle, angle)
    turn.min_turn_rate = turn.max_turn_rate
    turn.rel_standing_envs = 0.0

  return condition


def walk_distance(distance: float) -> Condition:
  """Hold one finite walk distance while recording the entry set"""

  def condition(cfg: ManagerBasedRlEnvCfg) -> None:
    twist = cfg.commands["twist"]
    assert isinstance(twist, WalkDistanceCommandCfg)
    twist.distance_range = (distance, distance)
    twist.rel_standing_envs = 0.0
    twist.initial_heading_error = twist.initial_lateral_error = 0.0

  return condition


def push_cells(cells: int) -> Condition:
  """Push the box a fixed number of cells."""

  def condition(cfg: ManagerBasedRlEnvCfg) -> None:
    pushes = [c for c in cfg.commands.values() if isinstance(c, PushCommandCfg)]
    assert len(pushes) == 1
    pushes[0].min_cells = pushes[0].max_cells = cells

  return condition


def ball_at_strike(cfg: ManagerBasedRlEnvCfg) -> None:
  """Spawn the ball exactly where the reference strikes it. Play mode scatters it."""
  spawn = cfg.events["ball_spawn"]
  spawn.params["forward_range"] = (0.0, 0.0)
  spawn.params["lateral_range"] = (0.0, 0.0)


SPECS: dict[str, dict[str, Spec]] = {
  "g1": {
    # One gait period after three seconds of walking, the speed soccer walks after the
    # jump. The build prints the measured period, 25 steps on 2026-10-14
    "walk": Spec(G1_WALK_TASK_ID, 150, 175, 4, walk_at(0.8)),
    # The bottom of the crouch the clip starts in, the root rises from frame 4
    "jump": Spec(JUMP_TASK_ID, 2, 4, 2),
    # Standing in front of the box, before the hands reach its edge
    "climb": Spec(CLIMB_TASK_ID, 2, 10, 1),
    # The approach, before the foot meets the ball around frame 150
    "kick": Spec(KICK_TASK_ID, 95, 145, 6, ball_at_strike),
    # No pass: the robot is still landing from its reset at step 2 and the shove moves
    # the ball from step 3, so nothing on its own trajectory can be entered
    "front_kick": Spec(MARTIAL_TASK_IDS["front_kick"], 27, 60, 4),
    "punch_combo": Spec(MARTIAL_TASK_IDS["punch_combo"], 27, 55, 4),
  },
  "t1": {
    # Sokoban enters at rest and latches the segment's distance after the handover
    "walk": Spec(T1_WALK_TASK_ID, 50, 51, 1, walk_distance(0.0)),
    # A few steps in, standing with the turn just started
    "turn_left": Spec(T1_TURN_TASK_ID, 5, 10, 1, turn_by(math.pi / 2)),
    "turn_right": Spec(T1_TURN_TASK_ID, 5, 10, 1, turn_by(-math.pi / 2)),
    # The hands nearly touch the box at the start, so only the first steps
    "push": Spec(T1_PUSH_TASK_ID, 1, 3, 1, push_cells(1)),
  },
}


def specs_for(robot: str) -> dict[str, Spec]:
  """The entry sets of a robot."""
  try:
    return dict(SPECS[robot])
  except KeyError:
    raise ValueError(
      f"No entry sets for robot '{robot}'. Available: {', '.join(SPECS)}"
    ) from None
