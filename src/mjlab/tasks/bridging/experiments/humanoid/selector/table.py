"""The entry table: where a skill can be started from, and how reliable each spot is.

    rollouts.npz     what record.py wrote, the input to all of it
    candidates.npz   every cluster build.py found, measured, unjudged
    entries.npz      the ones filter.py accepted

Columns:

    state      pose to aim at, (13 + 2J), canonical frame. See state.py
    command    what the skill was being asked for while it was in this state. Width is
               per skill, empty for a skill that takes none
    frame      control step inside the skill's episode, for setting the policy phase
    progress   0 at the episode start, 1 at the end
    coverage   fraction of the skill's rollouts that pass through this spot
    spread     radius of the spot. 1.0 is one arrival tolerance wide
    clearance  metres between the lowest part of the robot and the floor. About zero
               standing, positive in the air. See ground.py
    dwell_s    seconds a rollout stays inside it, median
    hold_s     coverage * dwell_s. Expected seconds a rollout spends here
    share      fraction of the skill's recorded states inside it

Rows are sorted by hold_s, biggest first: the spot a rollout is most likely to be in, for
longest. That is a measurement, not a recommendation. Which entry is best depends on where
the outgoing skill left the robot, and only the bridge knows that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROLLOUTS_PATH = Path("data") / "selector" / "rollouts.npz"
"""What record.py writes and build.py reads. The selector's own, not the bridge's."""

CANDIDATES_PATH = Path("data") / "selector" / "candidates.npz"
"""What build.py writes and filter.py reads. Every cluster, judged by nothing."""

TABLE_PATH = Path("data") / "selector" / "entries.npz"
"""What filter.py writes and everything else reads. The accepted entries."""

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


def _command(raw, index: int) -> np.ndarray:
  """One entry's command, padding removed. Empty for a file written before the column."""
  if "command" not in raw:
    return np.zeros(0, dtype=np.float32)
  return raw["command"][index][: int(raw["command_dim"][index])]


def _padded(commands: list[np.ndarray]) -> np.ndarray:
  """Commands of different widths as one table, padded to the widest. (N, G).

  Skills take different numbers of command values, and one row per entry is the layout
  everything else indexes by, so the narrow ones are padded rather than kept apart.
  `command_dim` says how much of each row is real.
  """
  width = max((c.size for c in commands), default=0)
  if width == 0:
    return np.zeros((len(commands), 0), dtype=np.float32)
  return np.stack([np.pad(c, (0, width - c.size)) for c in commands]).astype(np.float32)


@dataclass(frozen=True)
class Entry:
  """One place a skill can be entered."""

  skill: str
  name: str
  """Progress through the skill, as a percentage. Unique within a skill."""
  state: np.ndarray = field(compare=False)
  """(13 + 2J,) float32, canonical frame.

  Out of the comparison, or == between two entries raises: numpy answers elementwise
  and a dataclass wants one bool."""
  command: np.ndarray = field(compare=False)
  """(G,) what the skill was being asked for here. Empty for a skill with no command.

  Out of the comparison for the same reason as state. Widths differ per skill, so two
  entries of different skills are not comparable on this and nothing tries."""
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

  def command_text(self, most: int = 6) -> str:
    """The command as one short string. "none" for a skill that takes none.

    Truncated, because a tracking skill's command is mostly the reference it is chasing
    and that is dozens of numbers. Enough to tell two nodes apart at a glance, which is
    what it is for; the whole vector is in `command`.
    """
    if self.command.size == 0:
      return "none"
    shown = " ".join(f"{v:+.2f}" for v in self.command[:most])
    return (
      shown if self.command.size <= most else f"{shown} +{self.command.size - most}"
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
  rates: np.ndarray
  """(6,) fastest per-channel change per second the corpus achieved, in CHANNELS order.

  A property of the robot rather than of any one skill, so it is measured over every
  recorded state and carried by whichever file is being read. query.py divides by it to
  turn a gap into an effort. See build.achievable.
  """

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
        command=_command(raw, i),
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
    return EntryTable(
      entries=entries, fps=float(raw["fps"]), rates=raw["rates"].astype(np.float64)
    )

  def save(self, path: Path = TABLE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, Any] = {
      "states": np.stack([e.state for e in self.entries]).astype(np.float32),
      "skill": np.asarray([e.skill for e in self.entries]),
      "name": np.asarray([e.name for e in self.entries]),
      "frame": np.asarray([e.frame for e in self.entries], dtype=np.int32),
      "command": _padded([e.command for e in self.entries]),
      "command_dim": np.asarray([e.command.size for e in self.entries], dtype=np.int16),
      "fps": np.asarray(self.fps),
      "rates": np.asarray(self.rates, dtype=np.float32),
    }
    for column in COLUMNS:
      columns[column] = np.asarray(
        [getattr(e, column) for e in self.entries], dtype=np.float32
      )
    np.savez(path, **columns)
    print(f"[selector] wrote {path} ({len(self.entries)} entries)")
    return path
