"""Entry points of every skill of a robot.

An entry point is a state the skill itself goes through, under the settings the experiments
use and with its command held fixed. specs.py says, per robot, which task each entry set
runs, which command it holds, and which range of steps its entry points come from. One
command records the rollouts and builds the entry points from them.

Run

1. Record and build every entry set of a robot. Sets whose task has no checkpoint yet are
   skipped and listed at the end.

    uv run python -m mjlab.tasks.bridging.selector --robot g1
    uv run python -m mjlab.tasks.bridging.selector --robot t1

2. Look at them.

    uv run python -m mjlab.tasks.bridging.selector.view --robot g1

Only some sets, a pinned checkpoint, or another range for one set. The sets already in
the table and not built again are kept. Lists and tuples are written as Python literals,
checkpoints as name path pairs:

    uv run python -m mjlab.tasks.bridging.selector --robot g1 --sets "('jump', 'kick')"
    uv run python -m mjlab.tasks.bridging.selector --robot g1 --checkpoints walk logs/rsl_rl/g1_walk/<run>/model_2700.pt
    uv run python -m mjlab.tasks.bridging.selector --robot g1 --sets "('jump',)" --window "('jump', 1, 4, 3)"

A task outside the pool, recorded as its own set:

    uv run python -m mjlab.tasks.bridging.selector --robot t1 --sets "('jump',)" --task "('jump', 'Mjlab-T1-Jump-Continuous')" --window "('jump', 1, 10, 1)"

Output: data/selector/states.npz for the G1, data/selector/<robot>/states.npz otherwise.
"""

from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("data") / "selector"
ROLLOUTS_PATH = DATA_DIR / "rollouts.npz"
STATES_PATH = DATA_DIR / "states.npz"
TABLE_PATH = STATES_PATH

FUTURE_STATES = 32
"""Recorded states after each entry point, at least the planner future minus one."""


def paths(robot: str) -> tuple[Path, Path]:
  """Return rollout and state paths without moving existing G1 data."""
  root = DATA_DIR if robot == "g1" else DATA_DIR / robot
  return root / "rollouts.npz", root / "states.npz"


@dataclass(frozen=True)
class Window:
  """Half-open range of steps and number of entry points to sample from it."""

  start: int
  stop: int
  samples: int

  def __post_init__(self) -> None:
    if self.start < 0 or self.stop <= self.start:
      raise ValueError("A window must satisfy 0 <= start < stop")
    if not 1 <= self.samples <= self.stop - self.start:
      raise ValueError("A window needs between one sample and one per step")
