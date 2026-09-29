"""Measure how useful retargeted motion files are for bridge training.

Run:

  uv run python -m mjlab.tasks.bridging.diagnose_motions \
    --datasets "('LAFAN1=data/lafan1_g1/motions/*.npz', \
    'BABEL=data/babel_retargeted/unitree_g1_locomotion_v1/**/*.npz')"
"""

from __future__ import annotations

import glob
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import tyro

import mjlab
from mjlab.tasks.bridging.motion_filter import (
  LOCOMOTION_MOTION_FILTER,
  FilterProfile,
  MotionFilterCfg,
  filter_window_starts,
  motion_filter,
)


@dataclass(frozen=True)
class Summary:
  name: str
  profile: FilterProfile
  clips: int
  unique_sources: int
  frames: int
  total_s: float
  min_s: float
  median_s: float
  mean_s: float
  p90_s: float
  max_s: float
  candidate_windows: int
  kept_windows: int
  unscored_clips: int
  shorter_than_window: int
  robots: tuple[str, ...]
  frame_rates: tuple[float, ...]
  rejected: tuple[tuple[str, int], ...]
  categories: tuple[tuple[str, int], ...]


def diagnose(
  name: str,
  pattern: str,
  window_frames: int,
  filter_cfg: MotionFilterCfg | None = LOCOMOTION_MOTION_FILTER,
  profile: FilterProfile = "locomotion",
) -> Summary:
  """Summarize NPZ clips and score G1 windows with one filter profile."""
  if window_frames < 2:
    raise ValueError("window_frames must exceed one")
  files = sorted(Path(path) for path in glob.glob(pattern, recursive=True))
  if not files:
    raise FileNotFoundError(f"No motion files match {pattern!r}")

  durations: list[float] = []
  frames = candidate_windows = kept_windows = unscored = too_short = 0
  robots: set[str] = set()
  frame_rates: set[float] = set()
  sources: set[str] = set()
  rejected: Counter[str] = Counter()
  categories: Counter[str] = Counter()

  for path in files:
    with np.load(path, allow_pickle=False) as motion:
      fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
      joint_pos = np.asarray(motion["joint_pos"], dtype=np.float32)
      count = len(joint_pos)
      frames += count
      durations.append(max(0, count - 1) / fps)
      frame_rates.add(fps)
      if "robot" in motion:
        robots.add(str(motion["robot"]))
      if "babel_source" in motion:
        sources.add(str(motion["babel_source"]))
      if "babel_categories" in motion:
        categories.update(str(value) for value in motion["babel_categories"])

      windows = max(0, count - window_frames + 1)
      candidate_windows += windows
      if windows == 0:
        too_short += 1
        continue
      if filter_cfg is None:
        kept_windows += windows
        continue

      body_pos = np.asarray(motion["body_pos_w"], dtype=np.float32)
      body_quat = np.asarray(motion["body_quat_w"], dtype=np.float32)
      if body_pos.shape[1:] != (30, 3) or body_quat.shape[1:] != (30, 4):
        unscored += 1
        continue
      state = np.concatenate(
        (
          body_pos[:, 0],
          body_quat[:, 0],
          np.asarray(motion["body_lin_vel_w"], dtype=np.float32)[:, 0],
          np.asarray(motion["body_ang_vel_w"], dtype=np.float32)[:, 0],
          joint_pos,
          np.asarray(motion["joint_vel"], dtype=np.float32),
        ),
        axis=-1,
      )
      starts, reasons = filter_window_starts(
        state,
        body_pos,
        body_quat,
        window_frames,
        fps,
        filter_cfg,
      )
      kept_windows += len(starts)
      rejected.update(reasons)

  values = np.asarray(durations)
  return Summary(
    name=name,
    profile=profile,
    clips=len(files),
    unique_sources=len(sources),
    frames=frames,
    total_s=float(values.sum()),
    min_s=float(values.min()),
    median_s=float(np.median(values)),
    mean_s=float(values.mean()),
    p90_s=float(np.quantile(values, 0.9)),
    max_s=float(values.max()),
    candidate_windows=candidate_windows,
    kept_windows=kept_windows,
    unscored_clips=unscored,
    shorter_than_window=too_short,
    robots=tuple(sorted(robots)),
    frame_rates=tuple(sorted(frame_rates)),
    rejected=tuple(sorted(rejected.items())),
    categories=tuple(categories.most_common()),
  )


def _print(summary: Summary, window_frames: int) -> None:
  scored = summary.candidate_windows if summary.unscored_clips == 0 else None
  keep = (
    f"{summary.kept_windows:,}/{scored:,} ({100 * summary.kept_windows / scored:.1f}%)"
    if scored
    else "not available for every clip"
  )
  print(f"\n{summary.name}  filter={summary.profile}")
  print(
    f"  clips: {summary.clips:,}  frames: {summary.frames:,}  "
    f"duration: {summary.total_s / 60:.1f} min"
  )
  print(
    f"  clip seconds: min={summary.min_s:.2f} median={summary.median_s:.2f} "
    f"mean={summary.mean_s:.2f} p90={summary.p90_s:.2f} max={summary.max_s:.2f}"
  )
  print(
    f"  {window_frames}-frame windows: candidates={summary.candidate_windows:,} "
    f"kept={keep}  short clips={summary.shorter_than_window:,}"
  )
  print(
    f"  robots: {summary.robots or ('unknown',)}  fps: {summary.frame_rates}  "
    f"unscored clips: {summary.unscored_clips:,}"
  )
  if summary.unique_sources:
    print(f"  BABEL source recordings: {summary.unique_sources:,}")
  if summary.rejected:
    print(f"  rejected windows, overlapping: {dict(summary.rejected)}")
  if summary.categories:
    print(f"  BABEL categories: {dict(summary.categories)}")


def main(
  datasets: tuple[str, ...] = ("LAFAN1=data/lafan1_g1/motions/*.npz",),
  profiles: tuple[str, ...] = (),
  window_frames: int = 118,
) -> None:
  """Compare retargeted NPZ corpora using named kinematic filters."""
  selected_profiles: dict[str, FilterProfile] = {}
  for spec in profiles:
    name, separator, profile = spec.partition("=")
    if not separator or not name or profile not in ("quality", "locomotion", "none"):
      raise ValueError(f"Profile must use NAME=quality|locomotion|none: {spec!r}")
    selected_profiles[name] = cast(FilterProfile, profile)
  for spec in datasets:
    name, separator, pattern = spec.partition("=")
    if not separator or not name or not pattern:
      raise ValueError(f"Dataset must use NAME=GLOB syntax: {spec!r}")
    profile = selected_profiles.get(name, "locomotion")
    _print(
      diagnose(name, pattern, window_frames, motion_filter(profile), profile),
      window_frames,
    )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
