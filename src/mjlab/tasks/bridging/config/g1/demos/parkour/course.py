"""Load and generate reproducible parkour courses."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import yaml

from mjlab.tasks.bridging.config.g1.skills.climb.dataset import (
  box_from_manifest,
  motion_dir,
)

TALL = "tall"
SHORT = "short"
CONFIG_PATH = Path(__file__).parent / "config" / "config.yml"

Color = tuple[float, float, float, float]
Range = tuple[float, float]


def _range(raw: Sequence[float], name: str) -> Range:
  if len(raw) != 2:
    raise ValueError(f"{name} must be [min, max]")
  low, high = float(raw[0]), float(raw[1])
  if high < low:
    raise ValueError(f"{name} must be ordered")
  return low, high


def _color(raw: Sequence[float], name: str) -> Color:
  if len(raw) != 4 or any(not 0.0 <= float(value) <= 1.0 for value in raw):
    raise ValueError(f"{name} must be four values in [0, 1]")
  return float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])


@dataclass(frozen=True)
class ShortCfg:
  height: float
  length: float
  width: Range

  @staticmethod
  def of(raw: dict[str, Any]) -> ShortCfg:
    cfg = ShortCfg(
      height=float(raw["height"]),
      length=float(raw["length"]),
      width=_range(raw["width"], "obstacles.short.width"),
    )
    if cfg.height <= 0 or cfg.length <= 0 or cfg.width[0] <= 0:
      raise ValueError("Short obstacle dimensions must be positive")
    return cfg


@dataclass(frozen=True)
class ObstacleCfg:
  count: int
  pattern: tuple[str, ...]
  first_run_up: float
  spacing: Range
  lateral_offset: Range
  yaw_degrees: Range
  goal_clearance: float
  colors: tuple[Color, ...]
  short: ShortCfg

  @staticmethod
  def of(raw: dict[str, Any]) -> ObstacleCfg:
    pattern = tuple(str(kind) for kind in raw["pattern"])
    unknown = set(pattern) - {TALL, SHORT}
    if not pattern or unknown:
      raise ValueError(
        "obstacles.pattern must contain only "
        f"'{TALL}' and '{SHORT}', got {sorted(unknown)}"
      )
    colors = tuple(
      _color(color, f"obstacles.colors[{index}]")
      for index, color in enumerate(raw["colors"])
    )
    cfg = ObstacleCfg(
      count=int(raw["count"]),
      pattern=pattern,
      first_run_up=float(raw["first_run_up"]),
      spacing=_range(raw["spacing"], "obstacles.spacing"),
      lateral_offset=_range(raw["lateral_offset"], "obstacles.lateral_offset"),
      yaw_degrees=_range(raw["yaw_degrees"], "obstacles.yaw_degrees"),
      goal_clearance=float(raw["goal_clearance"]),
      colors=colors,
      short=ShortCfg.of(raw["short"]),
    )
    if cfg.count < 1:
      raise ValueError("obstacles.count must be positive")
    if cfg.first_run_up <= 0 or cfg.spacing[0] < 0 or cfg.goal_clearance < 0:
      raise ValueError("Course clearances must be nonnegative")
    if not cfg.colors:
      raise ValueError("obstacles.colors cannot be empty")
    return cfg


@dataclass(frozen=True)
class SideObstacleCfg:
  count: int
  side_offset: Range
  yaw_degrees: Range

  @staticmethod
  def of(raw: dict[str, Any]) -> SideObstacleCfg:
    cfg = SideObstacleCfg(
      count=int(raw["count"]),
      side_offset=_range(raw["side_offset"], "side_obstacles.side_offset"),
      yaw_degrees=_range(raw["yaw_degrees"], "side_obstacles.yaw_degrees"),
    )
    if cfg.count < 0 or cfg.side_offset[0] <= 0:
      raise ValueError("Side obstacle count must be nonnegative and offsets positive")
    return cfg


@dataclass(frozen=True)
class GoalCfg:
  position: tuple[float, float]
  radius: float
  color: Color

  @staticmethod
  def of(raw: dict[str, Any]) -> GoalCfg:
    position = tuple(float(value) for value in raw["position"])
    if len(position) != 2:
      raise ValueError("goal.position must contain x and y")
    radius = float(raw["radius"])
    if radius <= 0:
      raise ValueError("goal.radius must be positive")
    return GoalCfg(
      position=position,  # ty: ignore[invalid-argument-type]
      radius=radius,
      color=_color(raw["color"], "goal.color"),
    )


@dataclass(frozen=True)
class ControllerCfg:
  position_tolerance: float
  yaw_tolerance: float
  position_gain: float
  walk_speed: float
  bridge_distance: dict[str, float]
  hurdle_takeoff: float
  bridge_duration_s: dict[str, float]
  capture_grace_s: float
  capture_tolerance_scale: float
  lift_height: float
  land_height: float
  stand_angle: float
  settle_steps: int
  stuck_steps: int
  traverse_patience: int
  fall_height: float

  @staticmethod
  def of(raw: dict[str, Any]) -> ControllerCfg:
    cfg = ControllerCfg(
      position_tolerance=float(raw["position_tolerance"]),
      yaw_tolerance=float(raw["yaw_tolerance"]),
      position_gain=float(raw["position_gain"]),
      walk_speed=float(raw["walk_speed"]),
      bridge_distance={
        skill: float(value) for skill, value in raw["bridge_distance"].items()
      },
      hurdle_takeoff=float(raw["hurdle_takeoff"]),
      bridge_duration_s={
        skill: float(value) for skill, value in raw["bridge_duration_s"].items()
      },
      capture_grace_s=float(raw["capture_grace_s"]),
      capture_tolerance_scale=float(raw["capture_tolerance_scale"]),
      lift_height=float(raw["lift_height"]),
      land_height=float(raw["land_height"]),
      stand_angle=float(raw["stand_angle"]),
      settle_steps=int(raw["settle_steps"]),
      stuck_steps=int(raw["stuck_steps"]),
      traverse_patience=int(raw["traverse_patience"]),
      fall_height=float(raw["fall_height"]),
    )
    if set(cfg.bridge_distance) != {"jump", "climb"} or set(cfg.bridge_duration_s) != {
      "jump",
      "climb",
    }:
      raise ValueError("Bridge controls must define jump and climb")
    if any(
      value <= 0
      for value in (*cfg.bridge_distance.values(), *cfg.bridge_duration_s.values())
    ):
      raise ValueError("Bridge distances and times must be positive")
    if any(
      float(value) <= 0
      for name, value in vars(cfg).items()
      if name
      not in {
        "bridge_distance",
        "bridge_duration_s",
        "traverse_patience",
        "settle_steps",
        "stuck_steps",
      }
    ):
      raise ValueError("Controller distances, gains and times must be positive")
    if cfg.settle_steps < 1 or cfg.stuck_steps < 1 or cfg.traverse_patience < 0:
      raise ValueError("Controller step counts are invalid")
    if cfg.land_height >= cfg.lift_height:
      raise ValueError("controller.land_height must be below lift_height")
    return cfg


@dataclass(frozen=True)
class SkillCfg:
  jump_entry_frame: int
  climb_entry_frame: int

  @staticmethod
  def of(raw: dict[str, Any]) -> SkillCfg:
    return SkillCfg(
      jump_entry_frame=int(raw["jump_entry_frame"]),
      climb_entry_frame=int(raw["climb_entry_frame"]),
    )


@dataclass(frozen=True)
class Settings:
  seed: int
  obstacles: ObstacleCfg
  side_obstacles: SideObstacleCfg
  goal: GoalCfg
  controller: ControllerCfg
  skills: SkillCfg

  @staticmethod
  def load(path: Path | None = None) -> Settings:
    where = path or CONFIG_PATH
    if not where.exists():
      raise FileNotFoundError(f"No parkour config at {where}")
    raw = yaml.safe_load(where.read_text())
    return Settings(
      seed=int(raw["seed"]),
      obstacles=ObstacleCfg.of(raw["obstacles"]),
      side_obstacles=SideObstacleCfg.of(raw["side_obstacles"]),
      goal=GoalCfg.of(raw["goal"]),
      controller=ControllerCfg.of(raw["controller"]),
      skills=SkillCfg.of(raw["skills"]),
    )


@dataclass(frozen=True)
class Obstacle:
  kind: str
  position: tuple[float, float]
  half_size: tuple[float, float, float]
  yaw: float
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


@dataclass(frozen=True)
class Course:
  obstacles: tuple[Obstacle, ...]
  side_obstacles: tuple[Obstacle, ...]
  goal: GoalCfg
  seed: int

  def __len__(self) -> int:
    return len(self.obstacles)

  def __iter__(self) -> Iterator[Obstacle]:
    return iter(self.obstacles)

  def __getitem__(self, index: int) -> Obstacle:
    return self.obstacles[index]

  def lines(self) -> list[str]:
    rows = [
      "| # | kind | x | y | yaw | length | width | height |",
      "|---|---|---|---|---|---|---|---|",
    ]
    rows += [
      f"| {index} | {obstacle.kind} | {obstacle.position[0]:.2f} "
      f"| {obstacle.position[1]:+.2f} | {math.degrees(obstacle.yaw):+.0f} "
      f"| {obstacle.length:.2f} "
      f"| {obstacle.width:.2f} | {obstacle.height:.2f} |"
      for index, obstacle in enumerate(self.obstacles)
    ]
    return [
      f"course seed {self.seed}, goal at {self.goal.position}",
      f"{len(self.side_obstacles)} ignored side obstacles",
      *rows,
    ]


def climb_box():
  """Return the box recorded with the climb clip."""
  return box_from_manifest(motion_dir("climb"))


def _half_size(kind: str, cfg: ObstacleCfg, box, rng: random.Random):
  if kind == TALL:
    return (
      float(box.half_size[0]),
      float(box.half_size[1]),
      float(box.half_size[2]),
    )
  width = rng.uniform(*cfg.short.width)
  return cfg.short.length / 2.0, width / 2.0, cfg.short.height / 2.0


def generate(
  settings: Settings | None = None,
  seed: int | None = None,
  count: int | None = None,
) -> Course:
  """Generate one ordered random course between the origin and the goal."""
  cfg = settings or Settings.load()
  obstacle_cfg = cfg.obstacles
  seed = cfg.seed if seed is None else seed
  count = obstacle_cfg.count if count is None else count
  if count < 1:
    raise ValueError("Obstacle count must be positive")

  goal_x, goal_y = cfg.goal.position
  distance = math.hypot(goal_x, goal_y)
  if distance <= cfg.goal.radius:
    raise ValueError("The goal must be outside its own radius from the robot")
  axis = (goal_x / distance, goal_y / distance)
  side = (-axis[1], axis[0])
  course_yaw = math.atan2(axis[1], axis[0])
  rng = random.Random(seed)
  box = climb_box()
  cursor = obstacle_cfg.first_run_up
  obstacles: list[Obstacle] = []

  for index in range(count):
    kind = obstacle_cfg.pattern[index % len(obstacle_cfg.pattern)]
    half_size = _half_size(kind, obstacle_cfg, box, rng)
    yaw_offset = math.radians(rng.uniform(*obstacle_cfg.yaw_degrees))
    along_half = (
      abs(math.cos(yaw_offset)) * half_size[0]
      + abs(math.sin(yaw_offset)) * half_size[1]
    )
    along = cursor + along_half
    lateral = rng.uniform(*obstacle_cfg.lateral_offset)
    position = (
      along * axis[0] + lateral * side[0],
      along * axis[1] + lateral * side[1],
    )
    obstacles.append(
      Obstacle(
        kind=kind,
        position=position,
        half_size=half_size,
        yaw=course_yaw + yaw_offset,
        color=rng.choice(obstacle_cfg.colors),
      )
    )
    cursor = along + along_half
    if index + 1 < count:
      cursor += rng.uniform(*obstacle_cfg.spacing)

  side_obstacles: list[Obstacle] = []
  for _ in range(cfg.side_obstacles.count):
    kind = rng.choice(obstacle_cfg.pattern)
    half_size = _half_size(kind, obstacle_cfg, box, rng)
    along = rng.uniform(obstacle_cfg.first_run_up / 2.0, distance - cfg.goal.radius)
    lateral = rng.uniform(*cfg.side_obstacles.side_offset) * rng.choice((-1.0, 1.0))
    yaw = course_yaw + math.radians(rng.uniform(*cfg.side_obstacles.yaw_degrees))
    side_obstacles.append(
      Obstacle(
        kind=kind,
        position=(
          along * axis[0] + lateral * side[0],
          along * axis[1] + lateral * side[1],
        ),
        half_size=half_size,
        yaw=yaw,
        color=rng.choice(obstacle_cfg.colors),
      )
    )

  required = cursor + obstacle_cfg.goal_clearance + cfg.goal.radius
  if required > distance:
    raise ValueError(
      f"Course needs {required:.2f} m but the goal is {distance:.2f} m away. "
      "Move the goal or reduce count, spacing, or clearance"
    )
  return Course(tuple(obstacles), tuple(side_obstacles), cfg.goal, seed)
