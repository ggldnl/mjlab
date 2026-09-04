"""Accept or reject the candidates build.py found.

Third step of the pipeline: reads the candidate table, writes the entry table everything
else uses. Every reason a spot is not worth aiming at lives here.

    grounded   the robot is touching the floor in this pose
    common     enough rollouts pass through it
    steady     a rollout stays in it long enough to be aimed at
    early      enough of the skill is left to be worth entering

Re-running is instant, since the clustering in build.py does not repeat, so we just
have to tune a threshold.

To add a check, write a function taking a candidate and the config, returning None to
accept or a short reason to reject, and add it to CHECKS. The reason is printed as the
verdict, so make it say the number that failed.

Run

1. Judge the candidates. Prints every one with the check that failed it.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.filter

2. Move a threshold, or hide the rejections.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.filter --min-coverage 0.1
    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.filter --show-rejected False
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.selector.table import (
  CANDIDATES_PATH,
  HEADER,
  TABLE_PATH,
  Entry,
  EntryTable,
)


@dataclass
class FilterCfg:
  """What a candidate has to satisfy."""

  path: Path = CANDIDATES_PATH
  out: Path = TABLE_PATH
  show_rejected: bool = True
  """Print the candidates that failed, and which check failed them."""

  clearance_range: tuple[float, float] = (-0.03, 0.03)
  """Metres the lowest part of the robot may sit above the floor.

  The upper bound rejects mid-flight: a bridge steers a body it is standing on, so an
  airborne target is unreachable however cleanly the skill passes through it. The lower
  bound catches states written into the floor, where a reading well past the sink a solver
  allows is a defect worth seeing rather than a state worth aiming at.

  Grounded and airborne are two separated groups with nothing between them, so anything in
  the gap works as the upper bound.
  """

  min_coverage: float = 0.2
  """Fraction of rollouts that must pass through.

  Low on purpose. A skill that branches, kicking with either leg, sends about half its
  rollouts down each branch, and both branches are real entry points."""

  min_dwell_s: float = 0.05
  """Seconds a rollout must stay in it. Nothing can aim at a target open for one tick.

  Deliberately off the grid. Dwell is a whole number of control steps, so at 50 Hz it
  only ever takes values 0.02, 0.04, 0.06 and so on."""

  max_progress: float = 0.5
  """How far into the skill a spot may sit. Past this there is too little left to be
  worth entering at."""


Check = Callable[[Entry, FilterCfg], str | None]
"""None accepts. A string rejects, and is printed as the verdict."""


def grounded(entry: Entry, cfg: FilterCfg) -> str | None:
  """The robot has to be touching the floor.

  The one check that is about what a bridge can deliver rather than about what the skill
  does. Mid-flight states cluster well and score well, because nearly every jump goes
  through them; none of that helps, because no controller puts a body on a chosen
  ballistic arc.
  """
  low, high = cfg.clearance_range
  if entry.clearance > high:
    return f"airborne {entry.clearance:+.3f}"
  if entry.clearance < low:
    return f"underground {entry.clearance:+.3f}"
  return None


def common(entry: Entry, cfg: FilterCfg) -> str | None:
  """Enough rollouts have to pass through it, or it is not a spot the skill needs."""
  if entry.coverage < cfg.min_coverage:
    return f"rare {entry.coverage:.2f}"
  return None


def steady(entry: Entry, cfg: FilterCfg) -> str | None:
  """A rollout has to stay long enough that arriving does not need an exact tick."""
  if entry.dwell_s < cfg.min_dwell_s:
    return f"fleeting {entry.dwell_s:.2f} s"
  return None


def early(entry: Entry, cfg: FilterCfg) -> str | None:
  """Enough of the skill has to be left that entering here buys anything."""
  if entry.progress > cfg.max_progress:
    return f"late {entry.progress:.2f}"
  return None


CHECKS: tuple[Check, ...] = (grounded, early, common, steady)
"""Run in order. The first failure is the verdict, so the most informative goes first.

Geometry, then relevance, then statistics. A candidate at the very end of a skill is
usually also brief, and being told it is fleeting hides that entering there would have
bought nothing anyway."""


def verdict(entry: Entry, cfg: FilterCfg) -> str | None:
  """Why this candidate was rejected, or None if it was kept."""
  for check in CHECKS:
    reason = check(entry, cfg)
    if reason is not None:
      return reason
  return None


def apply(table: EntryTable, cfg: FilterCfg) -> tuple[EntryTable, dict[str, str]]:
  """The candidates that survive, and why each of the others did not."""
  kept: list[Entry] = []
  rejected: dict[str, str] = {}
  for entry in table.entries:
    reason = verdict(entry, cfg)
    if reason is None:
      kept.append(entry)
    else:
      rejected[f"{entry.skill}/{entry.name}"] = reason
  return EntryTable(tuple(kept), table.fps, table.rates), rejected


def report(table: EntryTable, rejected: dict[str, str]) -> list[str]:
  """Every candidate with its verdict, per skill, in candidate order."""
  out: list[str] = []
  for skill in table.skills:
    rows = [e for e in table.entries if e.skill == skill]
    survivors = sum(1 for e in rows if f"{skill}/{e.name}" not in rejected)
    out += [f"**{skill}**  {survivors} of {len(rows)} kept", ""]
    out.append(HEADER[0] + " verdict |")
    out.append(HEADER[1] + "---|")
    for entry in rows:
      reason = rejected.get(f"{skill}/{entry.name}", "kept")
      out.append(f"{entry.row()} {reason} |")
    out.append("")
  return out


def main(cfg: FilterCfg) -> None:
  candidates = EntryTable.load(cfg.path)
  kept, rejected = apply(candidates, cfg)
  if cfg.show_rejected:
    print("\n".join(report(candidates, rejected)))
  else:
    print("\n".join(kept.lines()))
  if not kept.entries:
    raise SystemExit(
      "Nothing survived. Every candidate is listed above with the check that failed it; "
      "loosen that one, or raise --clusters in build.py if the candidates are too wide."
    )
  kept.save(cfg.out)


if __name__ == "__main__":
  main(tyro.cli(FilterCfg, config=mjlab.TYRO_FLAGS))
