"""Minimal entry-state selector.

Run the two steps:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.record
    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.build
    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.view

For each skill, build.py keeps complete rollouts through the configured window, picks
the medoid rollout, and samples states from it. The selected states and their original
rollout context are written to data/selector/states.npz.
"""

from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("data") / "selector"
ROLLOUTS_PATH = DATA_DIR / "rollouts.npz"
STATES_PATH = DATA_DIR / "states.npz"
TABLE_PATH = STATES_PATH


@dataclass(frozen=True)
class Window:
  """Half-open phase range and number of states to sample from it."""

  start: int
  stop: int
  samples: int

  def __post_init__(self) -> None:
    if self.start < 0 or self.stop <= self.start:
      raise ValueError("A window must satisfy 0 <= start < stop")
    if not 1 <= self.samples <= self.stop - self.start:
      raise ValueError("A window needs between one sample and one per phase")


WINDOWS = {
  "jump": Window(55, 110, 6),
  "kick": Window(95, 145, 6),
  "climb": Window(30, 55, 1),
  "front_kick": Window(27, 60, 4),
  "punch_combo": Window(27, 55, 4),
  "pass": Window(27, 50, 1),
  "walk": Window(27, 60, 1),
}
"""Manual phase window for each selectable skill."""
