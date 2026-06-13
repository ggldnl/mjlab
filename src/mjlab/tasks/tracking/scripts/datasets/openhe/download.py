"""Adapter: openhe G1 retargeted motions -> mjlab tracking pipeline.

The ``openhe/g1-retargeted-motions`` HuggingFace dataset holds *atomic*,
single-skill clips (ACCAD: ``walk``, ``Run``, ``Crouch``, ...) retargeted to
the Unitree G1 -- exactly the clean, fundamental motions the tracking task
wants, unlike the long continuous LAFAN1 performances. The catch is the
format does not match ``mjlab.scripts.csv_to_npz``:

* it is **23-DOF** (ASAP G1 order: 12 legs + 3 waist + 4+4 arms, no wrists),
  whereas ``csv_to_npz`` expects the **29-DOF** G1 (3-DOF waist + 7-DOF arms);
* it is stored as ``.pkl`` (plain pickle for ACCAD, joblib for the LAFAN1
  subset), not the Unitree generalized-coordinate CSV.

This script bridges the gap. For each clip it:

  1. downloads the ``.pkl`` from HuggingFace,
  2. converts it to a Unitree-convention CSV -- base position, base quaternion
     in xyzw, then the 29 joint angles at 30 fps -- mapping the 23 source
     joints onto the 29-DOF layout *by name* and zeroing the 6 wrist DOFs the
     source lacks,
  3. replays that CSV through ``csv_to_npz`` so MuJoCo forward kinematics fill
     in the per-body poses with the correct (MuJoCo) body ordering.

Going through ``csv_to_npz`` (rather than synthesising the NPZ directly) is
deliberate: the NPZ stores body poses indexed by body number, and only the
real G1 model's FK gives the ordering the tracking task expects.

The 23->29 joint remap was verified against the data: on a Stand clip the
quaternion's small components are cols 0,1 (=> xyzw, scalar last), and on a
Walk clip the per-joint variance matches the ASAP 23-DOF order below.

License: the ACCAD source motions come from AMASS (ACCAD subset); the LAFAN1
subset derives from Ubisoft's LAFAN1 (CC BY-NC-ND 4.0, non-commercial). Cite
accordingly.

Run with:
  MUJOCO_GL=egl uv run python \
      src/mjlab/tasks/tracking/scripts/datasets/download.py \
      --output-dir data/openhe_g1 --render True
  # single skill / ad-hoc window:
  MUJOCO_GL=egl uv run python \
      src/mjlab/tasks/tracking/scripts/datasets/download.py \
      --skills jump --frames 300 460 --render True
"""

from __future__ import annotations

import os
import pathlib
import pickle
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import tyro

import mjlab
from mjlab.scripts import csv_to_npz

HF_BASE = "https://huggingface.co/datasets/openhe/g1-retargeted-motions/resolve/main"
SOURCE_FPS = 30

# ASAP G1 23-DOF order (12 legs + 3 waist + 4+4 arms, no wrists). Verified
# empirically: legs 0-11 (ankle_roll barely moves), waist 12-14, the
# shoulder_pitch columns (15, 19) are the big arm-swingers in a walk.
SOURCE_23_JOINTS: tuple[str, ...] = (
  "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
  "left_knee", "left_ankle_pitch", "left_ankle_roll",
  "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
  "right_knee", "right_ankle_pitch", "right_ankle_roll",
  "waist_yaw", "waist_roll", "waist_pitch",
  "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
  "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
)  # fmt: skip

# 29-DOF G1 order. MUST match the joint order ``csv_to_npz`` feeds to
# ``find_joints(..., preserve_order=True)`` -- i.e. CSV column k is joint k
# here. Joints absent from the 23-DOF source (the 6 wrist DOFs) are zeroed.
TARGET_29_JOINTS: tuple[str, ...] = (
  "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
  "left_knee", "left_ankle_pitch", "left_ankle_roll",
  "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
  "right_knee", "right_ankle_pitch", "right_ankle_roll",
  "waist_yaw", "waist_roll", "waist_pitch",
  "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
  "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
  "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
  "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)  # fmt: skip


@dataclass(frozen=True)
class Clip:
  """One openhe source clip to convert."""

  subset: str  # HF folder, e.g. "ACCAD_retargeted".
  filename: str  # ``.pkl`` file name within the subset.
  frames: tuple[int, int] | None = None  # Optional (start, end), 0-indexed.


# Curated atomic skills. ACCAD walk/run are short and clean (no slicing). The
# LAFAN1 ``jumps`` clip is one long performance, so slice a single rep with
# ``--frames`` (or preview first); the default below converts the whole clip.
SKILLS: dict[str, Clip] = {
  "walk": Clip("ACCAD_retargeted", "B3_-_walk1_stageii.pkl"),
  "run": Clip("ACCAD_retargeted", "C3_-_Run_stageii.pkl"),
  "crouch": Clip("ACCAD_retargeted", "A7_-_crouch_stageii.pkl"),
  "jump": Clip("lafan1_retargeted", "jumps1_subject1.pkl"),
}


def _build_dof_index() -> list[int | None]:
  """For each 29-DOF target joint, the source column index, or None (zero)."""
  src = {name: i for i, name in enumerate(SOURCE_23_JOINTS)}
  index: list[int | None] = [src.get(name) for name in TARGET_29_JOINTS]
  mapped = {name for name in TARGET_29_JOINTS if name in src}
  missing = set(SOURCE_23_JOINTS) - mapped
  if missing:
    raise RuntimeError(f"Source joints not mapped into the 29-DOF target: {missing}")
  return index


@contextmanager
def _portable_path_unpickling():
  """Allow unpickling foreign-OS ``Path`` objects.

  openhe pickles keep a ``PosixPath`` as the top-level dict key; instantiating
  a ``PosixPath`` on Windows (or a ``WindowsPath`` on POSIX) raises during
  unpickling. Alias the non-native flavor to the native one for the load -- we
  only ever read the dict's *value*, so the mangled key is harmless.
  """
  saved_posix, saved_windows = pathlib.PosixPath, pathlib.WindowsPath
  foreign = "PosixPath" if os.name == "nt" else "WindowsPath"
  native = pathlib.WindowsPath if os.name == "nt" else pathlib.PosixPath
  setattr(pathlib, foreign, native)
  try:
    yield
  finally:
    pathlib.PosixPath = saved_posix
    pathlib.WindowsPath = saved_windows


def _load_pkl(path: Path) -> dict[str, Any]:
  """Load an openhe clip, falling back to joblib (used by the LAFAN1 subset)."""
  with _portable_path_unpickling():
    return _load_pkl_inner(path)


def _load_pkl_inner(path: Path) -> dict[str, Any]:
  try:
    with open(path, "rb") as f:
      data = pickle.load(f)
  except (ModuleNotFoundError, pickle.UnpicklingError):
    # openhe's LAFAN1 subset is joblib-serialized; plain pickle either can't
    # resolve the joblib globals or chokes on its array framing.
    import importlib

    try:
      joblib = importlib.import_module("joblib")
    except ModuleNotFoundError as err:
      raise SystemExit(
        f"{path.name} is joblib-serialized; install joblib to read it "
        "(`uv add joblib` or `uv run --with joblib ...`)."
      ) from err
    data = joblib.load(path)
  # Top level is a single-entry dict keyed by the source path / clip id.
  inner = next(iter(data.values())) if isinstance(data, dict) else data
  return cast("dict[str, Any]", inner)


def _download(subset: str, filename: str, dest: Path) -> None:
  if dest.exists():
    return
  dest.parent.mkdir(parents=True, exist_ok=True)
  url = f"{HF_BASE}/{subset}/{filename}"
  tmp = dest.with_suffix(dest.suffix + ".part")
  print(f"  downloading {subset}/{filename}")
  urllib.request.urlretrieve(url, tmp)
  tmp.replace(dest)


def _to_unitree_csv(
  inner: dict[str, Any], dof_index: list[int | None], frames: tuple[int, int] | None
) -> np.ndarray:
  """Build the (N, 36) Unitree CSV matrix: pos(3) + quat_xyzw(4) + dof(29)."""
  sl = slice(None) if frames is None else slice(frames[0], frames[1])
  pos = np.asarray(inner["root_trans_offset"], dtype=np.float32)[sl]  # (N, 3)
  quat = np.asarray(inner["root_rot"], dtype=np.float32)[sl]  # (N, 4) xyzw
  src_dof = np.asarray(inner["dof"], dtype=np.float32)[sl]  # (N, 23)
  if src_dof.shape[1] != len(SOURCE_23_JOINTS):
    raise SystemExit(
      f"Expected {len(SOURCE_23_JOINTS)}-DOF source, got {src_dof.shape[1]}."
    )
  n = src_dof.shape[0]
  dof29 = np.zeros((n, len(TARGET_29_JOINTS)), dtype=np.float32)
  for j, src_col in enumerate(dof_index):
    if src_col is not None:
      dof29[:, j] = src_dof[:, src_col]
  return np.concatenate([pos, quat, dof29], axis=1)


def main(
  output_dir: Path = Path("data/openhe_g1"),
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  skills: tuple[str, ...] = (),
  frames: tuple[int, int] | None = None,
  keep_csv: bool = False,
) -> None:
  """Convert curated openhe G1 clips to tracking NPZs.

  Args:
    output_dir: Directory for the NPZs (and intermediate CSVs / videos).
    output_fps: Output frame rate for the NPZ (tracking default is 50).
    device: Torch device for the MuJoCo replay in ``csv_to_npz``.
    render: Also write an MP4 of each converted skill.
    skills: Subset of ``SKILLS`` keys to build. Empty builds all of them.
    frames: ``(start, end)`` frame window (0-indexed) applied to every
      selected skill. Most useful with a single ``--skills`` entry, e.g. to
      carve one jump out of the long LAFAN1 ``jumps`` clip.
    keep_csv: Keep the intermediate Unitree CSVs (otherwise removed).
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA unavailable; falling back to CPU (slow).")
    device = "cpu"

  wanted = skills or tuple(SKILLS)
  unknown = [s for s in wanted if s not in SKILLS]
  if unknown:
    raise SystemExit(
      f"Unknown skill(s): {', '.join(unknown)}. Available: {', '.join(SKILLS)}"
    )

  dof_index = _build_dof_index()
  pkl_dir = output_dir / "pkl"
  csv_dir = output_dir / "csv"
  csv_dir.mkdir(parents=True, exist_ok=True)

  for name in wanted:
    clip = SKILLS[name]
    win = frames if frames is not None else clip.frames
    print(f"\n=== {name}: {clip.subset}/{clip.filename} frames={win or 'all'} ===")

    pkl_path = pkl_dir / clip.subset / clip.filename
    _download(clip.subset, clip.filename, pkl_path)
    matrix = _to_unitree_csv(_load_pkl(pkl_path), dof_index, win)

    csv_path = csv_dir / f"{name}.csv"
    np.savetxt(csv_path, matrix, delimiter=",")
    print(f"  wrote {matrix.shape[0]} frames -> {csv_path}")

    csv_to_npz.main(
      input_file=str(csv_path),
      output_name=name,
      output_dir=output_dir,
      input_fps=SOURCE_FPS,
      output_fps=output_fps,
      device=device,
      render=render,
      upload_to_wandb=False,
      line_range=None,
    )
    if not keep_csv:
      csv_path.unlink(missing_ok=True)

  print(
    f"\nDone. Atomic skill NPZs in {output_dir}/. Train one with:\n"
    "  uv run train Mjlab-Tracking-Flat-Unitree-G1 "
    f"--env.commands.motion.motion-file {output_dir}/walk.npz --env.scene.num-envs 4096"
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
