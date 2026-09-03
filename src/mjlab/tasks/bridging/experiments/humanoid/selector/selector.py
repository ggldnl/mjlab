"""The old Selector API, backed by the measured table.

tests/stage.py and tests/handoff.py were written against a hand-written posture table.
They read three things, and this keeps all three working while the states underneath
come from build.py instead:

    Selector.load(skill)     that skill's entry points
    selector.lines()         what is in the table, for printing before a run
    selector.shortlist(env)  the states to aim at, best first

Shortlist also carries `frame` now. That is the control step inside the skill's episode
each entry was recorded at, which is what the entering skill's phase should be set to.
Nothing reads it yet.

Delete this file once those two tests call EntryTable directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.selector.table import (
  TABLE_PATH,
  Entry,
  EntryTable,
)


@dataclass
class Shortlist:
  """States to aim at, best first.

  A list rather than one state because choosing is not this component's job: which
  entry is right depends on where the outgoing skill left the robot, and only the
  bridge has an opinion about that. The caller walks down the list.
  """

  states: np.ndarray
  """(n, 13 + 2J), canonical frame. Ground position is zero and the heading is
  stripped, so whoever places one picks where it goes and which way it faces."""
  postures: tuple[Entry, ...]
  """The table row behind each state."""
  duration_s: np.ndarray
  """(n,) seconds the bridge gets to reach each."""
  frame: np.ndarray
  """(n,) control step inside the skill's episode. The phase to enter at."""

  def __len__(self) -> int:
    return int(self.states.shape[0])

  def __getitem__(self, i: int) -> tuple[np.ndarray, Entry, float]:
    """One entry: its state, its table row, and the time to reach it."""
    i = int(np.clip(i, 0, len(self) - 1))
    return self.states[i : i + 1], self.postures[i], float(self.duration_s[i])


@dataclass
class ScoringCfg:
  """How a hand-over is judged once the entering skill takes over.

  Nothing to do with choosing an entry. It is here because the arena that tests a
  transition has to say whether the hand-over worked, and that verdict needs a horizon,
  a discount and something to compare against.
  """

  window_s: float = 3.0
  """How long the entering skill is scored for."""
  discount: float = 0.99
  """Per-step discount, matching the skills' own PPO gamma.

  A skill that stumbles out of a bad entry and recovers two seconds later earns almost
  nothing back at gamma^100, so the stumble is charged. A flat mean over the window
  forgives exactly the entries a selector exists to avoid."""
  bar: float = 0.8
  """Share of its own baseline a skill has to reach for the hand-over to pass."""

  reach_s: float = 0.6
  """How long the bridge gets to reach an entry.

  One number for every entry, because nothing measures it yet. A crouch reached from a
  walk probably needs longer than a stand does."""


class Selector:
  """Where one skill can be started from."""

  def __init__(
    self,
    skill: str,
    entries: tuple[Entry, ...],
    baseline: float | None = None,
    cfg: ScoringCfg | None = None,
  ) -> None:
    if not entries:
      raise ValueError(f"'{skill}' has no entry points.")
    self.skill = skill
    self.entries = entries
    self.baseline = baseline
    self.cfg = cfg or ScoringCfg()

  @staticmethod
  def load(
    skill: str, cfg: ScoringCfg | None = None, path: Path = TABLE_PATH
  ) -> Selector:
    """This skill's rows out of the table on disk."""
    return Selector(skill, EntryTable.load(path).of(skill), None, cfg)

  ##
  # What the transition arena reads.
  ##

  @property
  def discount(self) -> float:
    return self.cfg.discount

  @property
  def bar(self) -> float:
    return self.cfg.bar

  def scoring_steps(self, fps: float) -> int:
    return max(1, int(round(self.cfg.window_s * fps)))

  def lines(self) -> list[str]:
    """What this selector is offering, for printing before a run starts."""
    out = [f"{len(self.entries)} entry points, measured, best first:"]
    out += [
      f"  {i}. {e.name:<8} progress {e.progress:.2f}  coverage {e.coverage:.2f}  "
      f"spread {e.spread:.2f}  dwell {e.dwell_s:.2f} s"
      for i, e in enumerate(self.entries)
    ]
    if self.baseline is None:
      out.append(
        "  no baseline measured, so a hand-over prints a raw discounted return and its "
        "pass/fail is not meaningful."
      )
    return out

  def shortlist(self, env: ManagerBasedRlEnv | None = None) -> Shortlist:
    """Every entry, as states this robot can be teleported into.

    `env` is unused and kept so the two tests do not have to change. The old table held
    postures in degrees that needed the compiled model to be stood on the floor; these
    are recorded states and are already where they belong.
    """
    del env
    return Shortlist(
      states=np.stack([e.state for e in self.entries]).astype(np.float32),
      postures=self.entries,
      duration_s=np.full(len(self.entries), self.cfg.reach_s, dtype=np.float32),
      frame=np.asarray([e.frame for e in self.entries], dtype=np.int32),
    )
