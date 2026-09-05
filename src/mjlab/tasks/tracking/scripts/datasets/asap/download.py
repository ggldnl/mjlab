"""Download ASAP's G1-retargeted motion clips.

ASAP (He et al., RSS 2025) publishes its motions already retargeted to
``g1_29dof_anneal_23dof`` as joblib pickles: a root trajectory plus 23 joint
angles per frame at 30 Hz. That is not what the tracking task reads, so a clip
still has to be converted, but this is where it is fetched from.

Only the named files are downloaded rather than the repository being cloned.
ASAP ships a whole framework, its meshes and every other motion, and none of
that is used here.

Run with:
  # Download every clip in the manifest.
  uv run python src/mjlab/tasks/tracking/scripts/datasets/asap/download.py
  # Download one family.
  uv run python src/mjlab/tasks/tracking/scripts/datasets/asap/download.py \\
      --motions "('jump_forward_level3',)"

Reading the pickles needs joblib, which is not a project dependency; the
converters that do so run under ``uv run --with joblib``. Downloading does not.

ASAP is released for research use. Cite ASAP (He et al., RSS 2025) and the
retargeting release in any publication.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import tyro

import mjlab

REPO = "LeCAR-Lab/ASAP"
REF = "main"
MOTION_PATH = "humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles"

# Every file in this release is one motion name wrapped in the same prefix and suffix, so
# the short name is what everything else uses and the long one never leaves this module
FILENAME = "0-motions_raw_tairantestbed_smpl_video_{name}_filter_amass.pkl"

DATASET_URL = (
  f"https://raw.githubusercontent.com/{REPO}/{{ref}}/{MOTION_PATH}/{FILENAME}"
)

DEFAULT_DIR = Path("data") / "asap" / "raw"
"""Where the pickles are cached. Converted motions go elsewhere, one directory per task."""

# The forward jumps, shortest to longest: level1 hops about half a metre, level5 clears
# nearly two. They are the ones a distance can be interpolated across
FORWARD_JUMPS: tuple[str, ...] = tuple(f"jump_forward_level{i}" for i in range(1, 6))

# Jumps that turn, and jumps that go sideways. They widen a goal space rather than filling
# it in: the turning ones make a yaw component mean something, the side ones a lateral one
TURNING_JUMPS: tuple[str, ...] = tuple(f"jump_degree_level{i}" for i in range(1, 6))
SIDE_JUMPS: tuple[str, ...] = tuple(f"side_jump_level{i}" for i in range(1, 5))

MOTIONS: tuple[str, ...] = FORWARD_JUMPS + TURNING_JUMPS + SIDE_JUMPS
"""Clip names this script knows about. Explicit, so a default run is reproducible and a
misspelled name is caught before it becomes a 404."""

CLIP_SETS: dict[str, tuple[str, ...]] = {
  "forward": FORWARD_JUMPS,
  "all": MOTIONS,
}
"""Named subsets, for callers that take a set rather than a list of names."""


def fetch(name: str, output_dir: Path = DEFAULT_DIR, ref: str = REF) -> Path:
  """Download one clip unless it is already cached, and return where it landed.

  Args:
    name: Short clip name, one of ``MOTIONS``.
    output_dir: Directory to cache the pickle in.
    ref: Branch or tag of the ASAP repository to download from.
  """
  destination = output_dir / FILENAME.format(name=name)
  if destination.exists():
    return destination

  output_dir.mkdir(parents=True, exist_ok=True)
  url = DATASET_URL.format(name=name, ref=ref)
  print(f"  downloading {name}")
  # Written to a temporary name first, so an interrupted download cannot leave a truncated
  # file behind that later runs treat as cached
  partial = destination.with_suffix(".pkl.part")
  try:
    urllib.request.urlretrieve(url, partial)
    partial.replace(destination)
  except Exception as exc:
    partial.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {url}: {exc}") from exc
  return destination


def main(
  output_dir: Path = DEFAULT_DIR,
  motions: tuple[str, ...] = MOTIONS,
  ref: str = REF,
) -> None:
  """Download the selected ASAP clips.

  Args:
    output_dir: Directory to cache the joblib pickles.
    motions: Clip names to download. Defaults to every clip in the manifest.
    ref: Branch or tag of the ASAP repository to download from.
  """
  unknown = [motion for motion in motions if motion not in MOTIONS]
  if unknown:
    raise SystemExit(
      f"Unknown motion(s): {', '.join(unknown)}. Available: {', '.join(MOTIONS)}"
    )

  print(f"Downloading {len(motions)} ASAP clip(s) to {output_dir}")
  downloaded = 0
  for motion in motions:
    if (output_dir / FILENAME.format(name=motion)).exists():
      print(f"  cached {motion}")
    else:
      fetch(motion, output_dir, ref)
      downloaded += 1
  print(f"\nDone. Downloaded {downloaded}; {len(motions) - downloaded} already cached.")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
