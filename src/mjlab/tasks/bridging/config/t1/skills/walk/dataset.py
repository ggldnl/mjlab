"""Retarget the LAFAN walks onto the T1 and build the style reward's frame library.

The style reward compares the robot to every frame of this library, so the library is
what natural means for the task. A frame holds the joint positions and velocities of one
retargeted LAFAN walking frame, labelled with the velocity the performer was moving at.

The label is the root displacement over a window centred on the frame, in that frame's
heading. The instantaneous root velocity would not work: it swings with every step (lateral
sway, speed dips at foot strike), so a constant command would only match some phases of the
gait cycle and the reward would favour holding them.

The clips go through the shared pipeline, so the bridge corpus can reuse them:

    select_lafan          picks the walk performances, keeps LAFAN's subject split
    retarget_clips        GMR solve, csv_to_npz replay at 50 Hz, floor QA, grounding
    motion_bad_frames     drops tilted, airborne and penetrating frames

Clips land in data/lafan_retargeted/booster_t1/{train,val}/walk, the library in
data/t1_walk_style/frames.npz. Both splits go into the library, since nothing is evaluated
on it. The home pose is appended as a standing frame with zero velocity, so a zero command
has something to match. Head joints are left out: GMR never drives them.

Run

1. Install GMR into the venv, once. Without dependencies, so it cannot move mujoco.

    uv pip install --no-deps -e data/GMR

2. Retarget the walks and build the library. Prints the velocity coverage per clip.

    uv run python -m mjlab.tasks.bridging.config.t1.skills.walk.dataset

3. Skip performances that look wrong, by BVH stem.

    uv run python -m mjlab.tasks.bridging.config.t1.skills.walk.dataset --skip walk3_subject2
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tyro

import mjlab
from mjlab.asset_zoo.robots.booster_t1.t1_constants import HOME_QPOS
from mjlab.datasets.lafan import download as lafan_download
from mjlab.retargeting.gmr.retarget import Clip, retarget_clips
from mjlab.tasks.bridging.bridges.dataset.motion_capture.filters import (
  FilterProfile,
  motion_bad_frames,
  motion_filter,
  select_lafan,
)
from mjlab.tasks.bridging.config.t1.skills.walk.style import LIBRARY_FILE

ROBOT = "booster_t1"
LAFAN_RAW = Path("data/lafan/raw")
RETARGETED_DIR = Path("data/lafan_retargeted") / ROBOT
EXCLUDED_JOINTS = ("AAHead_yaw", "Head_pitch")


def retarget_walks(skip: tuple[str, ...], device: str) -> list[Path]:
  """Retarget every LAFAN walk performance not retargeted yet, return the kept clips."""
  if not any(LAFAN_RAW.glob("*.bvh")):
    lafan_download.main()
  entries = [
    entry
    for entry in select_lafan(LAFAN_RAW, themes=("walk",))
    if Path(entry["source"]).stem not in skip
  ]
  # Same layout and metadata as the bridge corpus build, so it keeps these clips
  clips = [
    Clip(
      Path(entry["source"]),
      Path(entry["split"], entry["category"], Path(entry["source"]).stem + ".npz"),
      metadata={f"lafan_{key}": value for key, value in entry.items()},
    )
    for entry in entries
  ]
  rows, skipped = retarget_clips(clips, ROBOT, LAFAN_RAW, RETARGETED_DIR, device=device)
  for row in rows:
    if row["status"] == "rejected":
      print(f"  floor QA rejected {row['path']}")
  print(f"{RETARGETED_DIR}: {len(rows)} retargeted, {skipped} already there")
  return [
    RETARGETED_DIR / clip.output
    for clip in clips
    if (RETARGETED_DIR / clip.output).is_file()
  ]


def heading_velocity(
  root_pos: np.ndarray, root_quat: np.ndarray, fps: float, window_s: float
) -> np.ndarray:
  """Velocity label per frame: (vx, vy, yaw rate) over a centred window, heading frame."""
  w, x, y, z = root_quat.T
  yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
  count = len(root_pos)
  half = max(1, round(window_s * fps / 2))
  index = np.arange(count)
  lo = np.clip(index - half, 0, count - 1)
  hi = np.clip(index + half, 0, count - 1)
  dt = np.maximum(hi - lo, 1) / fps
  dx, dy = (root_pos[hi, :2] - root_pos[lo, :2]).T
  cos, sin = np.cos(yaw), np.sin(yaw)
  return np.stack(
    [(cos * dx + sin * dy) / dt, (-sin * dx + cos * dy) / dt, (yaw[hi] - yaw[lo]) / dt],
    axis=-1,
  ).astype(np.float32)


def build_library(
  motions: list[Path], profile: FilterProfile, stride: int, window_s: float
) -> dict[str, Any]:
  """Stack the valid frames of every clip, every stride'th, with their velocity labels."""
  cfg = motion_filter(profile)
  joint_names: list[str] | None = None
  joint_pos, joint_vel, velocity, clip_index = [], [], [], []
  print(f"{'clip':28s} {'kept':>6s}   vx p5/p50/p95      vy p5/p95    wz p5/p95")
  for index, path in enumerate(motions):
    with np.load(path, allow_pickle=False) as motion:
      values = {name: motion[name] for name in motion.files}
    names = [str(name) for name in values["joint_names"]]
    columns = [names.index(name) for name in names if name not in EXCLUDED_JOINTS]
    if joint_names is None:
      joint_names = [names[i] for i in columns]
    fps = float(np.asarray(values["fps"]).reshape(-1)[0])

    labels = heading_velocity(
      values["body_pos_w"][:, 0], values["body_quat_w"][:, 0], fps, window_s
    )
    valid = np.ones(len(labels), dtype=bool)
    if cfg is not None:
      state = np.concatenate(
        [values[f"body_{k}_w"][:, 0] for k in ("pos", "quat", "lin_vel", "ang_vel")]
        + [values["joint_pos"], values["joint_vel"]],
        axis=-1,
      )
      bad = motion_bad_frames(
        state, values["body_pos_w"], values["body_quat_w"], fps, cfg, ROBOT
      )
      valid = ~np.logical_or.reduce(tuple(bad.values()))
    keep = np.flatnonzero(valid)[::stride]

    joint_pos.append(values["joint_pos"][keep][:, columns])
    joint_vel.append(values["joint_vel"][keep][:, columns])
    velocity.append(labels[keep])
    clip_index.append(np.full(len(keep), index))
    vx, vy, wz = np.percentile(labels[keep], [5, 50, 95], axis=0).T
    print(
      f"{path.stem:28s} {len(keep):6d}   {vx[0]:+.2f}/{vx[1]:+.2f}/{vx[2]:+.2f}"
      f"   {vy[0]:+.2f}/{vy[2]:+.2f}   {wz[0]:+.2f}/{wz[2]:+.2f}"
    )
  if joint_names is None:
    raise SystemExit("No retargeted walk clips to build the library from")

  # The standing frame, so a zero command matches the home pose at rest
  joint_pos.append(np.array([[HOME_QPOS[name] for name in joint_names]]))
  joint_vel.append(np.zeros((1, len(joint_names))))
  velocity.append(np.zeros((1, 3)))
  clip_index.append(np.array([-1]))

  return {
    "joint_names": np.asarray(joint_names),
    "joint_pos": np.concatenate(joint_pos).astype(np.float32),
    "joint_vel": np.concatenate(joint_vel).astype(np.float32),
    "velocity": np.concatenate(velocity).astype(np.float32),
    "clip_index": np.concatenate(clip_index).astype(np.int32),
    "clips": np.asarray([path.stem for path in motions]),
  }


def main(
  skip: tuple[str, ...] = (),
  profile: FilterProfile = "locomotion",
  stride: int = 2,
  window_s: float = 1.0,
  output: Path = LIBRARY_FILE,
  device: str = "cuda:0",
) -> None:
  """Retarget the LAFAN walks to the T1 and write the style frame library.

  Args:
    skip: BVH stems to leave out, such as walk3_subject2.
    profile: Kinematic filter profile, see motion_capture/filters.py.
    stride: Keep every stride'th valid frame. 2 is 25 Hz, plenty for a nearest frame lookup.
    window_s: Width of the window the velocity label is averaged over, about one gait cycle.
    output: Where the library lands.
    device: Torch device for the csv_to_npz replay.
  """
  motions = retarget_walks(skip, device)
  library = build_library(motions, profile, stride, window_s)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output, **library)

  vx, vy, wz = np.percentile(library["velocity"], [1, 5, 50, 95, 99], axis=0).T
  print(
    f"\n{len(library['joint_pos'])} frames, {len(library['joint_names'])} joints -> {output}"
  )
  print("Coverage p1/p5/p50/p95/p99, compare with COMMAND_RANGES in walk_env_cfg.py")
  for name, column in (("vx", vx), ("vy", vy), ("wz", wz)):
    print(f"  {name}  " + "  ".join(f"{value:+.2f}" for value in column))


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
