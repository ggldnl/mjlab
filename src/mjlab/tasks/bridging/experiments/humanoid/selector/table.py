"""The entry table: the states each skill can be entered at.

    rollouts.npz   what record.py wrote, the input
    entries.npz    what build.py wrote, what everything else reads

Columns:

    state      pose to aim at, (13 + 2J), canonical frame. See state.py
    command    what the skill was being asked for here. Width is per skill, empty for a
               skill that takes none
    frame      the skill's own clock at this state. What a tracker gets resumed at, and
               step count for a skill with no reference, which resumes nothing
    seconds    frame in seconds. Where this sits in the skill
    coverage   fraction of the skill's rollouts that reach this slice of the window
    spread     how much those rollouts disagree here. 1.0 is one arrival tolerance wide
    clearance  metres between the lowest part of the robot and the floor. About zero
               standing, positive in the air. See ground.py
    previous_action  preceding policy action, (J,)
    reference        reference root pose in the same canonical frame, (7,)
    motion_file      clip filename, empty for a skill without a reference
    motion_scale     recorded clip scale

Rows are in frame order, earliest first, which is the order along the window they were
taken from. Nothing here ranks them: which one to enter at depends on where the outgoing
skill left the robot, and only query.py is told that.

Being in frame order means they are also in place order, and `trail` says by how much: the
ground each entry is further on than the one before it, reconstructed from the velocities
the states carry. That is what lets a whole window be drawn as the sequence a rollout would
pass through rather than as a row of unrelated poses.

coverage, spread and clearance are diagnostics. Nothing reads them to decide anything. A
wide spread means the rollouts disagreed where the window put a state, and a clearance
well above zero means the window reached into a part of the skill the robot spends
airborne. Both are reasons to move the window in selector/__init__.py, not reasons for
code to drop a row.

Read coverage against the clip length rather than on its own. A tracker resets into a
sampled frame of its own clip, so a 455 frame clip spreads its rollouts over 455 starts
and a slice near the opening is reached by a few percent of them, while a 212 frame clip
reaches a third. Low coverage on a long clip says the recording is thin there, not that
the skill avoids the spot.
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

COLUMNS = ("seconds", "coverage", "spread", "clearance")
"""Scalar columns, in the order they are printed. See the module docstring."""

BUILD_HINT = (
  "Build one with `uv run python -m "
  "mjlab.tasks.bridging.experiments.humanoid.selector.build`."
)


def _command(raw, index: int) -> np.ndarray:
  """One entry's command, padding removed."""
  return raw["command"][index][: int(raw["command_dim"][index])]


def _padded(commands: list[np.ndarray]) -> np.ndarray:
  """Commands of different widths as one table, padded to the widest. (N, G).

  Skills take different numbers of command values, and one row per entry is the layout
  everything else indexes by, so the narrow ones are padded rather than kept apart.
  command_dim says how much of each row is real.
  """
  width = max((c.size for c in commands), default=0)
  if width == 0:
    return np.zeros((len(commands), 0), dtype=np.float32)
  return np.stack([np.pad(c, (0, width - c.size)) for c in commands]).astype(np.float32)


@dataclass(frozen=True)
class Entry:
  """One state a skill can be entered at."""

  skill: str
  name: str
  """The frame, zero padded. Unique within a skill: the window is cut into disjoint
  slices and each state comes from one of them."""
  state: np.ndarray = field(compare=False)
  """(13 + 2J,) float32, canonical frame.

  Out of the comparison, or == between two entries raises: numpy answers elementwise and
  a dataclass wants one bool."""
  command: np.ndarray = field(compare=False)
  """(G,) what the skill was being asked for here. Empty for a skill with no command.

  Out of the comparison for the same reason as state. Widths differ per skill, so two
  entries of different skills are not comparable on this and nothing tries."""
  frame: int
  """The skill's own clock here. What a tracker gets resumed at.

  Not the step count for a tracker: training resets into a sampled frame of the clip, so
  a state recorded 53 steps into an episode can be anywhere in the motion."""
  seconds: float
  coverage: float
  spread: float
  clearance: float
  previous_action: np.ndarray | None = field(default=None, compare=False)
  reference: np.ndarray | None = field(default=None, compare=False)
  motion_file: str = ""
  motion_scale: float = 1.0
  segment: np.ndarray | None = field(default=None, compare=False)
  """(K, 13 + 2J) the frames this skill passed through just before this one.

  What the bridge merges onto. The entry is the last of them, so a bridge that rides this
  segment arrives already carrying the motion the skill is about to continue, and the
  hand-over is continuous because the two sides are literally the same motion rather than
  two motions blended at a seam.

  Earliest first, contiguous in one rollout. None where the recording does not reach back
  far enough, which is an entry too near the start of its own clip to have a run-up."""
  segment_bodies: np.ndarray | None = field(default=None, compare=False)
  """(K, B, 3) the same frames as body positions in the root frame. What the merge reward
  is measured in. See bridge.dataset.dataset.bodies."""

  @property
  def why(self) -> str:
    """What this state is, in one line."""
    return (
      f"frame {self.frame}, {self.seconds:.2f} s into the skill, "
      f"{self.coverage:.0%} of rollouts reach it, spread {self.spread:.2f}"
    )

  def command_text(self, most: int = 6) -> str:
    """The command as one short string. "none" for a skill that takes none.

    Truncated, because a tracking skill's command is mostly the reference it is chasing
    and that is dozens of numbers. Enough to tell two entries apart at a glance; the
    whole vector is in command.
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
      f"| {self.name} | {self.frame} | {self.seconds:.2f} | {self.coverage:.2f} "
      f"| {self.spread:.2f} | {self.clearance:+.3f} |"
    )


HEADER = (
  "| entry | frame | seconds | coverage | spread | clearance |",
  "|---|---|---|---|---|---|",
)


@dataclass
class EntryTable:
  """Every skill's entry states, in one file."""

  entries: tuple[Entry, ...]
  fps: float
  rates: np.ndarray
  """(6,) fastest per-channel change per second the corpus achieved, in CHANNELS order.

  A property of the robot rather than of any one skill, so it is measured over every
  recorded state and carried by the file. reach.py divides by it to turn a gap into an
  effort. See build.achievable.
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
    """One skill's entries, earliest frame first. Raises if it has none."""
    found = tuple(e for e in self.entries if e.skill == skill)
    if not found:
      raise SystemExit(
        f"No entry states for '{skill}'. This table holds {', '.join(self.skills)}. "
        f"Give '{skill}' a window in selector/__init__.py, then re-run selector.build."
      )
    return found

  def trail(self, skill: str) -> np.ndarray:
    """Where one skill's entries stand relative to each other, on the ground. (N, 3) metres.

    The entries of a skill are moments of one timeline, so they belong in a line rather than
    side by side: the robot is at the first one and then, a fraction of a second later and
    some distance further on, at the second. This says how much further on.

    Reconstructed rather than recorded, because the recorded ground position is not in a
    state. canonical() drops it, and it would not compose if it were there: two medoids come
    out of two rollouts and the tiles they stood on have nothing to do with each other. What
    does compose is the velocity each one carries, so the gap between two entries is the mean
    of their two root velocities over the time between their frames. That is the same model
    the bridge places its targets with, see stage.crossing.

    Straight, because canonical() drops heading too and every entry comes out pointing along
    +x. Right for a run-up and wrong for a skill that turns through its window, which none of
    these do.

    First entry at the origin, height left alone: that is in the state. A window the skill
    stands still through comes back all zeros, which is the truth about it rather than a
    defect, and the reason selector.view has a gap knob to prise those apart by eye.
    """
    entries = self.of(skill)
    out = np.zeros((len(entries), 3))
    for index in range(1, len(entries)):
      seconds = (entries[index].frame - entries[index - 1].frame) / self.fps
      speed = 0.5 * (entries[index - 1].state[7:9] + entries[index].state[7:9])
      out[index, 0:2] = out[index - 1, 0:2] + speed * seconds
    return out

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
      raise SystemExit(f"No entry table at {path}. {BUILD_HINT}")
    raw = np.load(path, allow_pickle=False)
    missing = [column for column in COLUMNS if column not in raw]
    if missing:
      raise SystemExit(
        f"{path} has no {', '.join(missing)} column, so an older selector wrote it. "
        f"{BUILD_HINT}"
      )
    entries = tuple(
      Entry(
        skill=str(raw["skill"][i]),
        name=str(raw["name"][i]),
        state=raw["states"][i],
        command=_command(raw, i),
        frame=int(raw["frame"][i]),
        seconds=float(raw["seconds"][i]),
        coverage=float(raw["coverage"][i]),
        spread=float(raw["spread"][i]),
        clearance=float(raw["clearance"][i]),
        previous_action=raw["previous_action"][i] if "previous_action" in raw else None,
        reference=raw["reference"][i] if "reference" in raw else None,
        motion_file=str(raw["motion_file"][i]) if "motion_file" in raw else "",
        motion_scale=float(raw["motion_scale"][i]) if "motion_scale" in raw else 1.0,
        segment=raw["segment"][i] if "segment" in raw else None,
        segment_bodies=raw["segment_bodies"][i] if "segment_bodies" in raw else None,
      )
      for i in range(raw["states"].shape[0])
    )
    return EntryTable(
      entries=entries, fps=float(raw["fps"]), rates=raw["rates"].astype(np.float64)
    )

  def save(self, path: Path = TABLE_PATH) -> Path:
    if any(e.previous_action is None or e.reference is None for e in self.entries):
      raise ValueError(
        "Entries require recorded actions and reference context; rerun selector.record"
      )
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
      "previous_action": np.stack(
        [e.previous_action for e in self.entries if e.previous_action is not None]
      ),
      "reference": np.stack(
        [e.reference for e in self.entries if e.reference is not None]
      ),
      "motion_file": np.asarray([e.motion_file for e in self.entries]),
      "motion_scale": np.asarray(
        [e.motion_scale for e in self.entries], dtype=np.float32
      ),
    }
    # All or nothing. A table where some entries carry a merge and others do not would
    # have the bridge silently fall back to the point target on the ones that do not
    if all(e.segment is not None for e in self.entries):
      columns["segment"] = np.stack(
        [e.segment for e in self.entries if e.segment is not None]
      ).astype(np.float32)
      columns["segment_bodies"] = np.stack(
        [e.segment_bodies for e in self.entries if e.segment_bodies is not None]
      ).astype(np.float32)
    for column in COLUMNS:
      columns[column] = np.asarray(
        [getattr(e, column) for e in self.entries], dtype=np.float32
      )
    np.savez(path, **columns)
    print(f"[selector] wrote {path} ({len(self.entries)} entries)")
    return path
