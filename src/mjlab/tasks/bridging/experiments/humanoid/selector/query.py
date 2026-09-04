"""Pick which entry to aim at, given where the robot is now.

Last step of the pipeline, and the only one with no command line: the bridge calls it.

    reaches = nearest(table, "jump", state, seconds=0.7)
    reaches[0].entry     the easiest one to get to
    reaches[0].effort    below 1 is reachable, above 1 is asking too much
    reaches[0].binding   which channel is the hard part

Not plain distance, because the channels have very different prices and two of them are
free. Ground position and heading cost nothing, since whoever aims the bridge picks them.
Momentum and posture are capped by what a body can do in the time available. So the
question is not how far a state is, but what rate of change getting there would demand and
whether a robot achieves that rate:

    effort = max over channels of   gap / seconds / achievable rate

achievable comes from the rollouts themselves, so an effort above 1 means the hand-over
needs a faster change than any recorded skill ever performed. See build.achievable.

A necessary condition, not a sufficient one. It says impossible reliably and possible
weakly, because taking the worst channel ignores that changing two channels at once is
harder than either alone. Capturing that needs a model fitted to bridge arrivals, which
needs a sweep of (start, target, duration) outcomes that does not exist yet.

nearest takes any Cost, so another scorer can be passed instead:

    nearest(table, "jump", state, 0.7, cost=MyCost(...))

The one worth trying next is the bridge's own critic, which already estimates "from here,
aiming there, how will I do" in one forward pass and is calibrated to the current bridge.
It is not the default because it would tie this package to a bridge checkpoint, and the
point of the selector is that it has none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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


class Cost(Protocol):
  """How hard it is to get from a state to an entry in a given time.

  here is already canonical, so an implementation never has to strip yaw itself and
  never sees a ground position it might wrongly treat as distance.
  """

  def __call__(self, here: np.ndarray, entry: Entry, seconds: float) -> Reach: ...


@dataclass(frozen=True)
class RateCost:
  """Required rate of change over the rate the corpus achieved. The default."""

  rates: np.ndarray
  """(6,) per second, in CHANNELS order. From EntryTable.rates."""

  def __call__(self, here: np.ndarray, entry: Entry, seconds: float) -> Reach:
    gap = channel_gap(
      torch.from_numpy(here.astype(np.float64))[None],
      torch.from_numpy(entry.state.astype(np.float64))[None],
    )[0].numpy()
    effort = gap / max(seconds, 1e-6) / np.maximum(self.rates, 1e-9)
    worst = int(np.argmax(effort))
    return Reach(
      entry=entry,
      effort=float(effort[worst]),
      binding=CHANNELS[worst],
      detail=tuple(float(value) for value in effort),
    )


def as_canonical(state: np.ndarray) -> np.ndarray:
  """One raw state in the frame entries are stored in. (13 + 2J,) -> (13 + 2J,)."""
  return (
    canonical(torch.from_numpy(np.asarray(state, dtype=np.float64))[None])[0]
    .numpy()
    .astype(np.float64)
  )


def nearest(
  table: EntryTable,
  skill: str,
  state: np.ndarray,
  seconds: float,
  cost: Cost | None = None,
) -> tuple[Reach, ...]:
  """That skill's entries, easiest to reach first.

  state is where the robot is now, as a raw dataset row. It is canonicalized here, so where
  on the floor it stands and which way it faces do not affect the answer.

  seconds is the window the bridge would get. Under RateCost it scales every effort by the
  same factor, so it decides which entries are reachable and never which is easiest: the
  order is the same at 0.3 s and at 1.2 s. A cost that modelled saturation would reorder
  them.

  Returns every entry, not the good ones. Rejecting is filter.py's job and already happened.
  """
  if seconds <= 0.0:
    raise ValueError(f"A window is a positive number of seconds, not {seconds}.")
  measure = cost if cost is not None else RateCost(table.rates)
  here = as_canonical(state)
  return tuple(
    sorted(
      (measure(here, entry, seconds) for entry in table.of(skill)),
      key=lambda reach: reach.effort,
    )
  )


def best(
  table: EntryTable,
  skill: str,
  state: np.ndarray,
  seconds: float,
  cost: Cost | None = None,
) -> Reach:
  """The easiest entry to reach. nearest when the caller does not want the rest."""
  return nearest(table, skill, state, seconds, cost)[0]


def lines(reaches: tuple[Reach, ...]) -> list[str]:
  """The ranking, for printing before a run starts."""
  return [
    f"{len(reaches)} entries, easiest first:",
    *(f"  {r.line()}" for r in reaches),
  ]
