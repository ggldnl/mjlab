"""Cut the punch combination out of a LAFAN1 fight performance and convert it.

Run:

    uv run python -m \
      mjlab.tasks.bridging.experiments.humanoid.skills.punch_combo.dataset

    # after moving the window
    uv run python -m ...punch_combo.dataset --crop.start 1580 --crop.end 1710

Source CSVs are cached in data/lafan1_g1, the converted motion lands in
data/lafan1_g1/punch_combo, and both steps skip files already there.

Cutting a strike out of a performance is the same job whatever the strike is, so this is the
frame window and nothing else. The converter lives in the front kick's dataset.py; read its
module docstring for what happens to the clip between the CSV and the npz.
"""

from __future__ import annotations

from pathlib import Path

import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.skills.front_kick.dataset import (
  CLIP_DIR,
  SOURCE_DIR,
  STILL_HOLD_S,
  Crop,
  convert,
)

MOTION_DIR = CLIP_DIR / "punch_combo"

# 131 frames, four and a half seconds at 30 Hz. Longer than the front kick because a
# combination is several strikes, and the window has to reach the end of the last one
CLIP = Crop("fight1_subject2", start=1600, end=1730)


def main(
  output_dir: Path = MOTION_DIR,
  source_dir: Path = SOURCE_DIR,
  crop: Crop = CLIP,
  output_fps: float = 50.0,
  hold_s: float = STILL_HOLD_S,
  device: str = "cuda:0",
) -> None:
  """Convert the punch combination window into an mjlab motion npz file.

  Args:
    output_dir: Where the npz file and the manifest are written.
    source_dir: Where the downloaded LAFAN1 CSVs are kept.
    crop: The frame window to cut. Move it if the default lands on the wrong strike.
    output_fps: Should match the env control rate, 1 / (timestep * decimation).
    hold_s: How long the reference stands still before the combination.
    device: Torch device for the replay.
  """
  convert("punch_combo", crop, output_dir, source_dir, output_fps, hold_s, device)


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
