"""What arch_4 needs told to it. Two halves, one of which does not exist yet.

The bridge is not trained from here. It is an mjlab task in its own right, trained once
against the motion corpus by `uv run train Mjlab-Bridge-G1` and reused by every
experiment, which is the whole point of a single bridge for the whole pool: it never sees
the skills, so it never has to be retrained when they change. What this config carries is
the path to that finished policy.

The chooser is the missing half. It is what decides which moment of the next skill to aim
at, and its fields will land here when it exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InBetweenerTraining:
  """Budget and wiring for arch_4."""

  bridge_checkpoint: Path | None = None
  """The trained bridge to load. None picks the newest under
  `logs/rsl_rl/bridge_in_betweener`, and whichever is chosen is printed, because a stale
  run outranking the intended one has cost this project a day before."""

  bridge_steps: int = 25
  """How long a hand-off is given, in control steps. The bridge was trained over holes of
  10 to 75 frames at 50 Hz, so anything in that band is a question it has been asked; a
  request outside it is not."""
