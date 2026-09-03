"""Entry points: where a skill can be started from, and how reliable each spot is.

Written by build.py, read by the transition tests to pick what the bridge aims at.

Columns
-------

    state      pose to aim at, (13 + 2J), canonical frame (see build.py)
    frame      control step inside the skill's episode, for setting the policy phase
    progress   0 at the episode start, 1 at the end
    coverage   fraction of the skill's rollouts that pass through this spot
    spread     radius of the spot. 1.0 is one arrival tolerance wide
    clearance  metres between the lowest part of the robot and the floor. About zero
               standing, positive in the air. See ground.py
    dwell_s    seconds a rollout stays inside it, median
    hold_s     coverage * dwell_s. Expected seconds a rollout spends here
    share      fraction of the skill's recorded states inside it

Rows are sorted by hold_s, biggest first: the spot a rollout is most likely to be in,
for longest. That is a measurement, not a recommendation. Which entry is best depends
on where the outgoing skill left the robot, and only the bridge knows that.

Run
---

    Build:  uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.build
    Look:   uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.view
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROLLOUTS_PATH = Path("data") / "selector" / "rollouts.npz"
"""What record.py writes and build.py reads. The selector's own, not the bridge's."""

TABLE_PATH = Path("data") / "selector" / "entries.npz"
"""What build.py writes and everything else reads."""

COLUMNS = (
  "progress",
  "coverage",
  "spread",
  "clearance",
  "dwell_s",
  "hold_s",
  "share",
)
"""Scalar columns, in the order they are printed. See the module docstring."""


@dataclass(frozen=True)
class Entry:
  """One place a skill can be entered."""

  skill: str
  name: str
  """Progress through the skill, as a percentage. Unique within a skill."""
  state: np.ndarray = field(compare=False)
  """(13 + 2J,) float32, canonical frame.

  Out of the comparison, or `==` between two entries raises: numpy answers elementwise
  and a dataclass wants one bool."""
  frame: int
  progress: float
  coverage: float
  spread: float
  clearance: float
  dwell_s: float
  hold_s: float
  share: float

  @property
  def why(self) -> str:
    """Why this is an entry point, in one line. The measurement, not a justification."""
    return (
      f"{self.coverage:.0%} of rollouts pass through, {self.dwell_s:.2f} s each, "
      f"{self.progress:.0%} of the way in"
    )

  def row(self) -> str:
    """One markdown table row."""
    return (
      f"| {self.name} | {self.frame} | {self.progress:.2f} | {self.coverage:.2f} "
      f"| {self.spread:.2f} | {self.clearance:+.3f} | {self.dwell_s:.2f} "
      f"| {self.hold_s:.2f} | {self.share:.2f} |"
    )


HEADER = (
  "| entry | frame | progress | coverage | spread | clearance | dwell_s | hold_s "
  "| share |",
  "|---|---|---|---|---|---|---|---|---|",
)


@dataclass
class EntryTable:
  """Every skill's entry points, in one file."""

  entries: tuple[Entry, ...]
  fps: float

  @property
  def skills(self) -> tuple[str, ...]:
    """Skill names, in the order they were built."""
    seen: list[str] = []
    for entry in self.entries:
      if entry.skill not in seen:
        seen.append(entry.skill)
    return tuple(seen)

  def of(self, skill: str) -> tuple[Entry, ...]:
    """One skill's entries, best first. Raises if the skill was never built."""
    found = tuple(e for e in self.entries if e.skill == skill)
    if not found:
      raise SystemExit(
        f"No entry points for '{skill}'. This table holds {', '.join(self.skills)}. "
        f"Build it with `--skills \"('{skill}',)\"`."
      )
    return found

  def lines(self, skill: str | None = None) -> list[str]:
    """The table as markdown. One skill, or all of them."""
    out: list[str] = []
    for name in [skill] if skill else self.skills:
      out += [f"**{name}**", *HEADER]
      out += [e.row() for e in self.of(name)]
      out.append("")
    return out

  ##
  # Disk.
  ##

  @staticmethod
  def load(path: Path = TABLE_PATH) -> EntryTable:
    if not path.exists():
      raise SystemExit(
        f"No entry table at {path}. Build one with `uv run python -m "
        f"mjlab.tasks.bridging.experiments.humanoid.selector.build`."
      )
    raw = np.load(path, allow_pickle=False)
    entries = tuple(
      Entry(
        skill=str(raw["skill"][i]),
        name=str(raw["name"][i]),
        state=raw["states"][i],
        frame=int(raw["frame"][i]),
        progress=float(raw["progress"][i]),
        coverage=float(raw["coverage"][i]),
        spread=float(raw["spread"][i]),
        clearance=float(raw["clearance"][i]),
        dwell_s=float(raw["dwell_s"][i]),
        hold_s=float(raw["hold_s"][i]),
        share=float(raw["share"][i]),
      )
      for i in range(raw["states"].shape[0])
    )
    return EntryTable(entries=entries, fps=float(raw["fps"]))

  def save(self, path: Path = TABLE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, Any] = {
      "states": np.stack([e.state for e in self.entries]).astype(np.float32),
      "skill": np.asarray([e.skill for e in self.entries]),
      "name": np.asarray([e.name for e in self.entries]),
      "frame": np.asarray([e.frame for e in self.entries], dtype=np.int32),
      "fps": np.asarray(self.fps),
    }
    for column in COLUMNS:
      columns[column] = np.asarray(
        [getattr(e, column) for e in self.entries], dtype=np.float32
      )
    np.savez(path, **columns)
    print(f"[selector] wrote {path} ({len(self.entries)} entries)")
    return path
