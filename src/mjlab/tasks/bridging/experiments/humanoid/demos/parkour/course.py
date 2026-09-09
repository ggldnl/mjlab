"""The obstacle course: what is on it, where, how it is turned, and what colour.

Pure geometry and one import from the climb skill. Nothing here builds a simulation, so a
course can be generated, planned over and drawn before anything is loaded.

A course is the whole description of the world. `arena` renders it and adds no geometry of
its own, which is what keeps the picture in the viewer and the thing the robot runs into
from drifting apart. Sampling happens once, here, at generation.

Two kinds of obstacle, and the kind is what picks the skill:

    box       a block to climb onto, cross and step down the far side       -> climb
    hurdle    a low bar to clear in one jump                                -> jump

Both are solid. The robot goes over one and onto the other, and hitting either is how the
demo ends.

Every box is the same size and it is not a free parameter. The climb was retargeted against
one obstacle and is only physical against that obstacle at that pose, so the size and the
angle between the robot and the box both come out of the skill's own manifest. What the
course chooses is where each box goes and how far it is turned on the lane, which is what
makes the controller solve a different approach line every time.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.demos.parkour.course
    uv run python -m ...demos.parkour.course --seed 7 --count 8
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import tyro
import yaml

BOX = "box"
HURDLE = "hurdle"

CONFIG_PATH = Path(__file__).parent / "config.yml"
"""Where the parameters live when the caller does not say."""

CLIMB_MOTION = "climb"
"""Which of the climb skill's motions the demo builds its boxes around.

One name, out of the skill's own MOTIONS. A second climb trained on a taller box would be a
second entry there and a second kind of obstacle here, and the two would have to be told
apart by the rule table rather than by size."""

Color = tuple[float, float, float, float]
Range = tuple[float, float]


def _range(raw: Sequence[float]) -> Range:
  low, high = float(raw[0]), float(raw[1])
  if high < low:
    raise SystemExit(f"A range is [low, high], not {list(raw)}.")
  return low, high


def _color(raw: Sequence[float]) -> Color:
  if len(raw) != 4:
    raise SystemExit(f"A colour is rgba, four numbers, not {list(raw)}.")
  return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))


def climb_box():
  """The obstacle the climb was trained against, from the skill's own manifest.

  Not a course parameter and not a constant restated here. OmniRetarget retargets the human
  motion and the box together and preserves the contacts between them, so the clip is only
  physical against that box at that pose: a course that drew its own size would be asking a
  policy to climb something it has never touched.

  Falls back to the skill's nominal box when the clip has not been converted on this
  machine, which is what lets a course be drawn before the data is fetched.
  """
  from mjlab.tasks.bridging.experiments.humanoid.skills.climb.dataset import (
    NOMINAL_BOX,
    box_from_manifest,
    motion_dir,
  )

  try:
    return box_from_manifest(motion_dir(CLIMB_MOTION))
  except Exception:
    return NOMINAL_BOX


##
# The settings.
##


@dataclass(frozen=True)
class LaneCfg:
  first_run_up: float
  spacing: Range

  @staticmethod
  def of(raw: dict[str, Any]) -> LaneCfg:
    return LaneCfg(
      first_run_up=float(raw["first_run_up"]), spacing=_range(raw["spacing"])
    )


@dataclass(frozen=True)
class HurdleCfg:
  length: float
  height: float
  width: float
  yaw: Range
  clearance: float

  @staticmethod
  def of(raw: dict[str, Any]) -> HurdleCfg:
    return HurdleCfg(
      length=float(raw["length"]),
      height=float(raw["height"]),
      width=float(raw["width"]),
      yaw=_range(raw["yaw"]),
      clearance=float(raw["clearance"]),
    )


@dataclass(frozen=True)
class ApproachCfg:
  yaw_tolerance: float
  lateral_tolerance: float
  hurdle_takeoff: float

  @staticmethod
  def of(raw: dict[str, Any]) -> ApproachCfg:
    return ApproachCfg(
      yaw_tolerance=float(raw["yaw_tolerance"]),
      lateral_tolerance=float(raw["lateral_tolerance"]),
      hurdle_takeoff=float(raw["hurdle_takeoff"]),
    )


@dataclass(frozen=True)
class WalkCfg:
  blend_radius: float
  lateral_gain: float
  lateral_limit: float
  approach_gain: float
  reverse_limit: float
  approach_speed: float
  cruise_speed: float

  @staticmethod
  def of(raw: dict[str, Any]) -> WalkCfg:
    return WalkCfg(
      blend_radius=float(raw["blend_radius"]),
      lateral_gain=float(raw["lateral_gain"]),
      lateral_limit=float(raw["lateral_limit"]),
      approach_gain=float(raw["approach_gain"]),
      reverse_limit=float(raw["reverse_limit"]),
      approach_speed=float(raw["approach_speed"]),
      cruise_speed=float(raw["cruise_speed"]),
    )


@dataclass(frozen=True)
class EndCfg:
  fall_height: float
  tip_angle: float

  @staticmethod
  def of(raw: dict[str, Any]) -> EndCfg:
    return EndCfg(
      fall_height=float(raw["fall_height"]), tip_angle=float(raw["tip_angle"])
    )


@dataclass(frozen=True)
class Settings:
  """Everything the demo reads out of config.yml."""

  seed: int
  count: int
  lane: LaneCfg
  palette: tuple[Color, ...]
  box_yaw: Range
  hurdle: HurdleCfg
  approach: ApproachCfg
  walk: WalkCfg
  end: EndCfg

  @staticmethod
  def load(path: Path | None = None) -> Settings:
    """Read config.yml, or another file shaped like it."""
    where = path or CONFIG_PATH
    if not where.exists():
      raise SystemExit(f"No course config at {where}.")
    raw = yaml.safe_load(where.read_text())
    palette = tuple(_color(c) for c in raw["palette"])
    if not palette:
      raise SystemExit("The palette needs at least one colour.")
    return Settings(
      seed=int(raw["seed"]),
      count=int(raw["count"]),
      lane=LaneCfg.of(raw["lane"]),
      palette=palette,
      box_yaw=_range(raw["box"]["yaw"]),
      hurdle=HurdleCfg.of(raw["hurdle"]),
      approach=ApproachCfg.of(raw["approach"]),
      walk=WalkCfg.of(raw["walk"]),
      end=EndCfg.of(raw["end"]),
    )


##
# What comes out of it.
##


@dataclass(frozen=True)
class Obstacle:
  """One thing in the lane, fully placed."""

  kind: str
  """BOX or HURDLE. What picks the skill. See controller.RULES."""
  at: tuple[float, float]
  """Centre on the floor, in metres, in the world the course is laid out in."""
  half_size: tuple[float, float, float]
  """Half extents in the obstacle's own frame. x is along its approach."""
  yaw: float
  """How far it is turned off the lane, radians."""
  color: Color

  @property
  def length(self) -> float:
    return 2.0 * self.half_size[0]

  @property
  def width(self) -> float:
    return 2.0 * self.half_size[1]

  @property
  def height(self) -> float:
    return 2.0 * self.half_size[2]

  @property
  def near(self) -> float:
    """Where along the lane it starts. Its own turn is ignored, so this is for spacing the
    course and for printing, never for aiming at anything."""
    return self.at[0] - self.length / 2.0

  @property
  def far(self) -> float:
    """Where along the lane it ends. Same caveat as `near`."""
    return self.at[0] + self.length / 2.0

  def face_normal(self) -> tuple[float, float]:
    """The outward direction of the face the robot arrives at, in world."""
    return (-math.cos(self.yaw), -math.sin(self.yaw))

  def row(self) -> str:
    return (
      f"| {self.kind} | {self.at[0]:.2f} | {self.at[1]:+.2f} | {self.height:.2f} "
      f"| {self.length:.2f} | {self.width:.2f} | {math.degrees(self.yaw):+.0f} |"
    )


HEADER = (
  "| kind | x | y | height | length | width | yaw |",
  "|---|---|---|---|---|---|---|",
)


@dataclass(frozen=True)
class Course:
  """A line of obstacles, the settings that drew them, and the seed."""

  obstacles: tuple[Obstacle, ...]
  seed: int
  settings: Settings = field(compare=False, repr=False)

  def __len__(self) -> int:
    return len(self.obstacles)

  def __iter__(self) -> Iterator[Obstacle]:
    return iter(self.obstacles)

  def __getitem__(self, index: int) -> Obstacle:
    return self.obstacles[index]

  @property
  def length(self) -> float:
    """Where the course ends, in metres."""
    if not self.obstacles:
      return self.settings.lane.first_run_up
    return self.obstacles[-1].far + self.settings.lane.first_run_up

  def lines(self) -> list[str]:
    return [
      f"course seed {self.seed}, {len(self)} obstacles over {self.length:.1f} m",
      *HEADER,
      *(obstacle.row() for obstacle in self.obstacles),
    ]


def generate(
  settings: Settings | None = None,
  seed: int | None = None,
  count: int | None = None,
) -> Course:
  """Draw a course of alternating kinds, starting with a box.

  Alternating rather than drawn independently. Two of a kind in a row is a stretch that asks
  for the same skill twice, and a stretch with no hand-over in it is a stretch the demo has
  nothing to say about.

  Every obstacle is laid out on the lane's centre line and then turned. Turning it rather
  than moving it off the line is deliberate: it changes the approach the controller has to
  solve without changing how far the robot has to walk, so a course is a sequence of
  different alignment problems at a constant spacing.
  """
  cfg = settings or Settings.load()
  seed = cfg.seed if seed is None else seed
  count = cfg.count if count is None else count
  rng = np.random.default_rng(seed)
  box = climb_box()

  obstacles: list[Obstacle] = []
  cursor = cfg.lane.first_run_up
  for index in range(count):
    color = cfg.palette[int(rng.integers(len(cfg.palette)))]
    if index % 2 == 0:
      half = tuple(float(v) for v in box.half_size)
      obstacle = Obstacle(
        kind=BOX,
        at=(cursor + half[0], 0.0),
        half_size=half,  # ty: ignore[invalid-argument-type]
        yaw=float(rng.uniform(*cfg.box_yaw)),
        color=color,
      )
    else:
      hurdle = cfg.hurdle
      obstacle = Obstacle(
        kind=HURDLE,
        at=(cursor + hurdle.length / 2.0, 0.0),
        half_size=(hurdle.length / 2.0, hurdle.width / 2.0, hurdle.height / 2.0),
        yaw=float(rng.uniform(*hurdle.yaw)),
        color=color,
      )
    obstacles.append(obstacle)
    cursor = obstacle.far + float(rng.uniform(*cfg.lane.spacing))
  return Course(obstacles=tuple(obstacles), seed=seed, settings=cfg)


def main(
  config: Path | None = None, seed: int | None = None, count: int | None = None
) -> None:
  """Print a course. What the generator drew, before anything is built or planned."""
  course = generate(Settings.load(config), seed=seed, count=count)
  for line in course.lines():
    print(line)
  box = climb_box()
  print(
    f"\nclimb box from the skill: {box.height:.3f} m tall, "
    f"{2 * box.half_size[0]:.3f} m along the approach, {2 * box.half_size[1]:.3f} m across, "
    f"at {math.degrees(box.yaw):+.1f} degrees to the robot"
  )


if __name__ == "__main__":
  tyro.cli(main)
