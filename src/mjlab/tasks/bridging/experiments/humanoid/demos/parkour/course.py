"""The obstacle course: what is on it, and where.

Pure geometry. Nothing here imports mjlab, so a course can be generated, planned over and
printed before a simulation exists. `controller.plan` does exactly that.

One obstacle is one box lying across the lane. Two numbers decide what it asks for:

    height   top face off the floor
    depth    how far the robot travels while over it

Between them they cover the range a course needs, from a kerb to a wall. Which skill each
one asks for is not decided here. That mapping is the controller's, so it lives where it
can be printed and argued with. See controller.RULES.

Sizes come off the Perceptive Humanoid Parkour spec sheet, which is the closest published
thing to this demo: steps at 0.36 m, climbs at 0.58 and higher. The height jitter is
OmniRetarget's own augmentation range, 0.8 to 1.2, for the same reason it is theirs: a
traversal skill trained on one height should not be memorising it.

Run

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.demos.parkour.course
    uv run python -m ...demos.parkour.course --seed 7 --count 8
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import tyro

LANE_WIDTH = 1.6
"""How wide a box is across the lane, in metres.

Wide enough that going round it is not a traversal. Nothing stops the locomotion skill from
steering, so an obstacle narrow enough to sidestep is one the demo can pass without ever
switching, which is the one outcome that would prove nothing."""

HEIGHT_JITTER = (0.8, 1.2)
"""What a family's nominal height is multiplied by. OmniRetarget's terrain augmentation
range, so a course asks for heights its retargeted references were augmented across."""

RUN_UP_RANGE = (6.0, 9.0)
"""Metres from one obstacle's far face to the next one's near face.

Two things have to fit in here and the second is the tight one: the robot has to get back
up to speed after a traversal, and the controller has to have the switch fire early enough
that the bridge's window closes at the take-off point. A gap shorter than the run-up plus
the longest window is a gap where the hand-over cannot be placed at all."""

FIRST_RUN_UP = 8.0
"""Metres from the start to the first obstacle's near face. Longer than the range above,
because the robot starts from a standstill and the first approach is the only one that has
to accelerate from zero."""


@dataclass(frozen=True)
class Family:
  """One kind of obstacle, at its nominal size."""

  name: str
  height: float
  depth: float


FAMILIES: tuple[Family, ...] = (
  Family("kerb", 0.36, 0.25),
  Family("span", 0.20, 1.20),
  Family("vault", 0.58, 0.45),
  Family("wall", 0.94, 0.60),
)
"""The four shapes a course is built from.

Chosen so height and depth separate them rather than the name doing it: the span is the
only low deep one and the wall the only tall one, so a controller reading two numbers off
the world reaches the same answer as one reading the label. That is the point of the
families being data rather than an enum. The rules never see this tuple.
"""


@dataclass(frozen=True)
class Obstacle:
  """One box lying across the lane."""

  family: str
  at: float
  """Centre of the box along the lane, in metres from the start."""
  height: float
  depth: float
  width: float = LANE_WIDTH

  @property
  def near(self) -> float:
    """Where the robot has to be off the floor by, in metres along the lane."""
    return self.at - self.depth / 2.0

  @property
  def far(self) -> float:
    """Where it can be back on the floor, in metres along the lane."""
    return self.at + self.depth / 2.0

  @property
  def half_size(self) -> tuple[float, float, float]:
    """Half extents, the shape `get_box_cfg` wants."""
    return (self.depth / 2.0, self.width / 2.0, self.height / 2.0)

  def row(self) -> str:
    """One markdown table row."""
    return (
      f"| {self.family} | {self.at:.2f} | {self.height:.2f} | {self.depth:.2f} "
      f"| {self.near:.2f} | {self.far:.2f} |"
    )


HEADER = (
  "| family | at | height | depth | near | far |",
  "|---|---|---|---|---|---|",
)


@dataclass(frozen=True)
class Course:
  """A line of obstacles, and the seed that drew them."""

  obstacles: tuple[Obstacle, ...]
  seed: int

  def __len__(self) -> int:
    return len(self.obstacles)

  def __iter__(self) -> Iterator[Obstacle]:
    return iter(self.obstacles)

  def __getitem__(self, index: int) -> Obstacle:
    return self.obstacles[index]

  @property
  def length(self) -> float:
    """Where the finish line goes, in metres. A run-up past the last obstacle, so a robot
    that clears it still has somewhere to land and settle."""
    if not self.obstacles:
      return FIRST_RUN_UP
    return self.obstacles[-1].far + FIRST_RUN_UP

  def lines(self) -> list[str]:
    """The course as markdown, for printing before a run."""
    return [
      f"course seed {self.seed}, {len(self)} obstacles over {self.length:.1f} m",
      *HEADER,
      *(obstacle.row() for obstacle in self.obstacles),
    ]


def generate(
  seed: int = 0,
  count: int = 5,
  families: tuple[Family, ...] = FAMILIES,
  run_up: tuple[float, float] = RUN_UP_RANGE,
  jitter: tuple[float, float] = HEIGHT_JITTER,
) -> Course:
  """Draw a course.

  Families are drawn without replacement inside each pass through the set, rather than
  independently every time. Independent draws put two walls in a row often enough to matter,
  and a course that happens to ask for the same skill twice is a course that never shows a
  hand-over the demo exists to show. Cycling guarantees every skill appears before any
  appears twice.

  Depth is not jittered. It is what decides whether a low box is stepped over or leaped, so
  moving it moves the boundary the controller's rules are written against, and a demo whose
  obstacles drift across a rule boundary is one where a wrong skill choice reads as a bad
  rule rather than as a bad draw.
  """
  rng = np.random.default_rng(seed)
  order: list[Family] = []
  while len(order) < count:
    order += list(rng.permutation(np.asarray(families, dtype=object)))
  order = order[:count]

  obstacles: list[Obstacle] = []
  cursor = FIRST_RUN_UP
  for family in order:
    height = family.height * float(rng.uniform(*jitter))
    obstacles.append(
      Obstacle(
        family=family.name,
        at=cursor + family.depth / 2.0,
        height=height,
        depth=family.depth,
      )
    )
    cursor = obstacles[-1].far + float(rng.uniform(*run_up))
  return Course(obstacles=tuple(obstacles), seed=seed)


def main(seed: int = 0, count: int = 5) -> None:
  """Print a course. What the generator drew, before anything is built or planned."""
  for line in generate(seed=seed, count=count).lines():
    print(line)


if __name__ == "__main__":
  tyro.cli(main)
