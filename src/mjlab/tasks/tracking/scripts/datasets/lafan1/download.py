"""Download Unitree G1-retargeted LAFAN1 performances.

The source dataset contains long, continuous 30 Hz CSV performances rather
than isolated skills. Each row is the G1 generalized coordinate used by the
tracking tools: base position, base quaternion in xyzw order, and 29 joint
angles. Download the clips here, then use ``interactive_crop.py`` or
``manual_crop.py`` to extract a single skill and convert it to an NPZ.

Run with:
  # Download every published G1 performance.
  uv run python src/mjlab/tasks/tracking/scripts/datasets/lafan1/download.py
  # Download only clips needed for the default manual crops.
  uv run python src/mjlab/tasks/tracking/scripts/datasets/lafan1/download.py \\
      --motions "('walk1_subject1', 'run2_subject1', 'sprint1_subject2', 'jumps1_subject1')"

LAFAN1 is released under CC BY-NC-ND 4.0. Cite LAFAN1 (Harvey et al.,
SIGGRAPH 2020) and the Unitree retargeting release in any publication.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import tyro

import mjlab

DATASET_URL = (
  "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset"
  "/resolve/main/g1/{name}.csv"
)

# G1 CSVs published by the retargeting release. Keeping the manifest explicit
# makes a default run reproducible and catches misspelled clip names early.
MOTIONS: tuple[str, ...] = (
  "dance1_subject1",
  "dance1_subject2",
  "dance1_subject3",
  "dance2_subject1",
  "dance2_subject2",
  "dance2_subject3",
  "dance2_subject4",
  "dance2_subject5",
  "fallAndGetUp1_subject1",
  "fallAndGetUp1_subject4",
  "fallAndGetUp1_subject5",
  "fallAndGetUp2_subject2",
  "fallAndGetUp2_subject3",
  "fallAndGetUp3_subject1",
  "fight1_subject2",
  "fight1_subject3",
  "fight1_subject5",
  "fightAndSports1_subject1",
  "fightAndSports1_subject4",
  "jumps1_subject1",
  "jumps1_subject2",
  "jumps1_subject5",
  "run1_subject2",
  "run1_subject5",
  "run2_subject1",
  "run2_subject4",
  "sprint1_subject2",
  "sprint1_subject4",
  "walk1_subject1",
  "walk1_subject2",
  "walk1_subject5",
  "walk2_subject1",
  "walk2_subject3",
  "walk2_subject4",
  "walk3_subject1",
  "walk3_subject2",
  "walk3_subject3",
  "walk3_subject4",
  "walk3_subject5",
  "walk4_subject1",
)


def _download(name: str, output_dir: Path) -> bool:
  """Download one performance unless it is already cached.

  Returns whether a download occurred.
  """
  destination = output_dir / f"{name}.csv"
  if destination.exists():
    print(f"  cached {destination.name}")
    return False

  output_dir.mkdir(parents=True, exist_ok=True)
  partial = destination.with_suffix(".csv.part")
  print(f"  downloading {destination.name}")
  try:
    urllib.request.urlretrieve(DATASET_URL.format(name=name), partial)
    partial.replace(destination)
  except Exception as exc:
    partial.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {destination.name}: {exc}") from exc
  return True


def main(
  output_dir: Path = Path("data/lafan1_g1"),
  motions: tuple[str, ...] = MOTIONS,
) -> None:
  """Download the selected LAFAN1 G1 CSV performances.

  Args:
    output_dir: Directory to cache the headerless 36-column G1 CSV files.
    motions: Clip names to download, without the ``.csv`` suffix. Defaults to
      every performance published in the G1 release.
  """
  unknown = [motion for motion in motions if motion not in MOTIONS]
  if unknown:
    raise SystemExit(
      f"Unknown motion(s): {', '.join(unknown)}. Available: {', '.join(MOTIONS)}"
    )

  print(f"Downloading {len(motions)} LAFAN1 G1 performance(s) to {output_dir}")
  downloaded = sum(_download(motion, output_dir) for motion in motions)
  print(f"\nDone. Downloaded {downloaded}; {len(motions) - downloaded} already cached.")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
