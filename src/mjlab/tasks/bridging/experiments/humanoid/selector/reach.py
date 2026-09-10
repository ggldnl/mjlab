"""How hard one entry is to get to from where the robot is now.

The caller says which entry it wants. This says what reaching it would demand.

    r = reach(table, "jump", 0, state, seconds=0.7)
    r.effort     below 1 is reachable, above 1 is asking too much
    r.binding    which channel is the hard part

Not plain distance, because the channels have very different meanings and two of them are
free. Ground position and heading cost nothing, since whoever aims the bridge picks them.
Momentum and posture are capped by what a body can do in the time available. So the
question is not how far a state is, but what rate of change getting there would demand and
whether a robot achieves that rate:

    effort = max over channels of   gap / seconds / achievable rate

achievable comes from the rollouts themselves, so an effort above 1 means the hand-over
needs a faster change than any recorded skill ever performed. See build.achievable.

This gives a necessary condition, not a sufficient one. It says impossible reliably and
possible weakly, because taking the worst channel ignores that changing two channels at
once is harder than either alone.

Nothing here picks an entry. Choosing used to live in this module, ranking the entries by
effort and handing back the easiest. It now belongs to whoever is running the transition:
demos name the entry per skill, benchmarks take it as a flag, the staging viewer puts it on
a slider. Effort is still worth computing for all of them, because it is what sizes the
bridge's window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mjlab.tasks.bridging.experiments.humanoid.selector.state import (
  CHANNELS,
  canonical,
  channel_gap,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.table import Entry, EntryTable


@dataclass(frozen=True)
class Reach:
  """How hard one entry is to get to."""

  entry: Entry
  effort: float
  """Below 1 is reachable, above 1 asks for more than a body does. See the module doc."""
  binding: str
  """The channel that set the effort. The reason this entry is as hard as it is."""
  detail: tuple[float, ...]
  """Effort per channel, in CHANNELS order. The breakdown behind binding."""

  def line(self) -> str:
    parts = " ".join(
      f"{name} {value:.2f}" for name, value in zip(CHANNELS, self.detail, strict=True)
    )
    return f"{self.entry.name:<8} effort {self.effort:5.2f}  {self.binding:<10} {parts}"


def as_canonical(state: np.ndarray) -> np.ndarray:
  """One raw state in the frame entries are stored in. (13 + 2J,) -> (13 + 2J,)."""
  return (
    canonical(torch.from_numpy(np.asarray(state, dtype=np.float64))[None])[0]
    .numpy()
    .astype(np.float64)
  )


def effort_of(
  here: np.ndarray, entry: Entry, seconds: float, rates: np.ndarray
) -> Reach:
  """Required rate of change over the rate the corpus achieved, worst channel wins.

  here must already be canonical, so a ground position never gets read as distance.
  """
  gap = channel_gap(
    torch.from_numpy(here.astype(np.float64))[None],
    torch.from_numpy(entry.state.astype(np.float64))[None],
  )[0].numpy()
  effort = gap / max(seconds, 1e-6) / np.maximum(rates, 1e-9)
  worst = int(np.argmax(effort))
  return Reach(
    entry=entry,
    effort=float(effort[worst]),
    binding=CHANNELS[worst],
    detail=tuple(float(value) for value in effort),
  )


def reaches(
  table: EntryTable, skill: str, state: np.ndarray, seconds: float
) -> tuple[Reach, ...]:
  """Every entry of that skill, in table order, with what each would demand.

  Table order, never sorted by effort. The order is the order build.py wrote, which is the
  order the skill passes through them, so index 0 is the earliest entry and stays the
  earliest entry whatever the robot is doing. A caller naming an entry by index needs that
  to hold still.

  state is where the robot is now, as a raw dataset row. It is canonicalized here, so where
  on the floor it stands and which way it faces do not affect the answer.

  seconds is the window the bridge would get. It scales every effort by the same factor.
  """
  if seconds <= 0.0:
    raise ValueError(f"A window is a positive number of seconds, not {seconds}.")
  here = as_canonical(state)
  return tuple(
    effort_of(here, entry, seconds, table.rates) for entry in table.of(skill)
  )


def reach(
  table: EntryTable, skill: str, index: int, state: np.ndarray, seconds: float
) -> Reach:
  """One entry of that skill by index, and what reaching it would demand."""
  found = reaches(table, skill, state, seconds)
  if not 0 <= index < len(found):
    raise IndexError(f"{skill} has {len(found)} entries, asked for index {index}.")
  return found[index]


def lines(found: tuple[Reach, ...]) -> list[str]:
  """The entries and their efforts, for printing before a run starts."""
  return [f"{len(found)} entries:", *(f"  {r.line()}" for r in found)]
