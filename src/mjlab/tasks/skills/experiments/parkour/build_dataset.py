"""Build a goal-conditioned skill dataset from LAFAN1 command plateaus.

The Unitree-retargeted LAFAN1 clips are long, multi-motion performances: a
single file wanders through several behaviors with natural human transitions
between them (walk-to-run, run-to-jump, jump-to-recover). We turn that into an
asset instead.

The idea:

  1.    Label every frame of a clip with a command: the root velocity expressed
        in the robot's own frame, [v_forward, v_lateral, yaw_rate].
  1.5.  Consider polishing this signal (e.g. smoothing it out).
  2.    Segment the clip on plateaus of that command. A plateau is a stretch
        where the command barely changes, i.e. a near-constant goal. The ramps
        between plateaus are the transitions; we deliberately discard them.

Each surviving plateau is a clean, stationary goal signal, exactly what
a goal-conditioned skill wants to track. This script produces one CSV
slice per plateau (which drops straight into mjlab.scripts.csv_to_npz)
plus a manifest labeling each with its mean command.

Flow:

  1.  download: fetch the selected G1 CSVs from HuggingFace (gated repo).
  2.  label:    per frame, the root velocity in the robot's yaw frame.
  2.5 cleaning: consider cleaning the signal.
  3.  segment:  keep the plateaus of that command, drop the transitions.
  4.  save:     one CSV slice per plateau + a manifest of mean commands.

Run:
  # accept the dataset terms once at
  #   https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset
  # then export your token (the repo is gated, so this is required):
  export HF_TOKEN=hf_...
  uv run python src/mjlab/tasks/skills/experiments/lafan/build_dataset.py

"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import tyro

# LAFAN1 is 30 FPS. The CSV is: base pos (3) + base quat xyzw (4) + 29 joints.
FPS = 30.0

# Gated HuggingFace repo; the G1 CSVs live under the ``g1/`` subfolder.
HF_REPO = "unitreerobotics/LAFAN1_Retargeting_Dataset"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/g1"

# Clips to use. Locomotion only, on purpose: some LAFAN1 clips are dancing or
# fighting, whose root velocity is not a locomotion command we care about.
SELECTED: tuple[str, ...] = (
  "walk1_subject1",
  "run1_subject2",
  "sprint1_subject1",
  "jumps1_subject1",
)

# Plateau detection. A frame is "steady" when the smoothed command changes
# slower than these per-second tolerances on every channel; runs of steady
# frames shorter than the minimum are dropped along with the transitions.
SMOOTH_WINDOW = 15  # frames (~0.5 s) of moving average before measuring change
CHANGE_TOL = np.array([0.5, 0.5, 1.0])  # [m/s, m/s, rad/s] change per second
MIN_PLATEAU = 30  # frames (~1 s); shorter plateaus are too brief to be a skill


def download(clip: str, out_dir: Path) -> Path:
  """Fetch one G1 CSV into ``out_dir`` (skips if already there)."""
  path = out_dir / f"{clip}.csv"
  if path.exists():
    return path
  token = os.environ.get("HF_TOKEN")
  if not token:
    raise SystemExit(
      "The LAFAN1 repo is gated. Accept its terms on HuggingFace and set "
      "HF_TOKEN before running."
    )
  out_dir.mkdir(parents=True, exist_ok=True)
  request = urllib.request.Request(
    f"{HF_BASE}/{clip}.csv", headers={"Authorization": f"Bearer {token}"}
  )
  print(f"Downloading {clip}.csv ...")
  with urllib.request.urlopen(request) as response, open(path, "wb") as f:
    f.write(response.read())
  return path


def local_root_velocity(rows: np.ndarray) -> np.ndarray:
  """Per-frame command: root velocity in the yaw frame [v_fwd, v_lat, yaw_rate]."""
  dt = 1.0 / FPS
  pos = rows[:, 0:3]
  quat_xyzw = rows[:, 3:7]

  # Yaw from the quaternion (xyzw, scalar last).
  x, y, z, w = quat_xyzw.T
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

  # World planar velocity by finite difference, rotated into the yaw frame.
  vel_world = np.gradient(pos, axis=0) / dt
  vx, vy = vel_world[:, 0], vel_world[:, 1]
  cos, sin = np.cos(yaw), np.sin(yaw)
  v_fwd = cos * vx + sin * vy
  v_lat = -sin * vx + cos * vy

  yaw_rate = np.gradient(np.unwrap(yaw)) / dt

  return np.stack([v_fwd, v_lat, yaw_rate], axis=1)


def _smooth(command: np.ndarray) -> np.ndarray:
  """Per-channel moving average."""
  if SMOOTH_WINDOW <= 1:
    return command
  kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
  return np.stack(
    [np.convolve(command[:, i], kernel, mode="same") for i in range(command.shape[1])],
    axis=1,
  )


def find_plateaus(command: np.ndarray) -> list[tuple[int, int]]:
  """Return [start, end) ranges where the command is on a plateau."""
  change = np.abs(np.gradient(_smooth(command), axis=0)) * FPS  # per-second
  steady = np.all(change <= CHANGE_TOL, axis=1)

  plateaus: list[tuple[int, int]] = []
  i, n = 0, len(steady)
  while i < n:
    if not steady[i]:
      i += 1
      continue
    j = i
    while j < n and steady[j]:
      j += 1
    if j - i >= MIN_PLATEAU:
      plateaus.append((i, j))
    i = j
  return plateaus


def main(
  data_dir: Path = Path("data/lafan1_g1"),
  output_dir: Path = Path("data/lafan1_g1/segments"),
  clips: tuple[str, ...] = SELECTED,
) -> None:
  """Download the selected clips, then write one CSV per command plateau."""
  output_dir.mkdir(parents=True, exist_ok=True)
  manifest = []

  for clip in clips:
    path = download(clip, data_dir)
    rows = np.loadtxt(path, delimiter=",")
    command = local_root_velocity(rows)
    plateaus = find_plateaus(command)
    print(f"{clip}: {len(plateaus)} plateau(s) from {len(rows)} frames")

    for k, (start, end) in enumerate(plateaus):
      name = f"{clip}_seg{k:02d}"
      np.savetxt(output_dir / f"{name}.csv", rows[start:end], delimiter=",")
      manifest.append(
        {
          "name": name,
          "source": clip,
          "start": int(start),
          "end": int(end),
          "num_frames": int(end - start),
          "command": command[start:end].mean(axis=0).round(4).tolist(),
        }
      )

  (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
  print(f"\nWrote {len(manifest)} segment(s) to {output_dir}/")


if __name__ == "__main__":
  tyro.cli(main)
