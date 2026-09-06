"""Fetch and convert the one ASAP clip this task tracks.

The downloader is the tracking task's datasets/asap/download.py, the converter
is the continuous jump's dataset.py. Read that module's docstring. The npz
lands in data/asap/motions next to whatever else has been converted.

Run

    uv run --with joblib python -m \
      mjlab.tasks.bridging.experiments.humanoid.skills.jump.dataset
"""

from __future__ import annotations

from pathlib import Path

import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.dataset import (
  DEFAULT_OUTPUT_DIR,
  convert,
)
from mjlab.tasks.tracking.scripts.datasets.asap import download as asap

CLIP_NAME = "jump_forward_level3"
"""Which ASAP clip this task tracks.

The level 3 forward jump, 1.54 m of displacement over 212 frames, takeoff at frame 120 and
landing at 137. Picked for the distance: level2 is 1.11 m and level4 is 1.82 m, so this is
the one that lands where a 1.5 m jump lands without stretching anything.

It is also the cleanest of the five to reset into. Its sole sits 3.1 cm above the floor at
frame zero and 2.2 cm above it at the load landmark, where every other clip is at or below
zero at one of the two. See JumpCommandCfg.entry_landmark for the table.
"""


def motion_file(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
  """Where the converted clip lands, which is what the env config tracks."""
  return output_dir / f"{CLIP_NAME}.npz"


def main(
  output_dir: Path = DEFAULT_OUTPUT_DIR,
  cache_dir: Path = asap.DEFAULT_DIR,
  output_fps: float = 50.0,
  align: str = "displacement",
  ref: str = asap.REF,
  device: str = "cuda:0",
) -> None:
  """Convert this task's clip into an mjlab motion npz file.

  Args:
    output_dir: Where the npz file and the manifest are written.
    cache_dir: Where the downloaded ASAP pickles are kept.
    output_fps: Should match the env control rate, 1 / (timestep * decimation).
    align: Which yaw to remove, "displacement" or "heading".
    ref: Branch or tag of the ASAP repository to download from.
    device: Torch device for the replay.
  """
  convert((CLIP_NAME,), output_dir, cache_dir, output_fps, align, ref, device)


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
