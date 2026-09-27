from pathlib import Path

import numpy as np

from mjlab.tasks.bridging.diagnose_motions import diagnose
from mjlab.tasks.bridging.motion_filter import (
  QUALITY_MOTION_FILTER,
  filter_window_starts,
)


def test_diagnose_counts_bridge_windows(tmp_path: Path) -> None:
  frames = 8
  body_pos = np.zeros((frames, 30, 3), dtype=np.float32)
  body_pos[:, 0, 2] = 0.75
  body_pos[:, [6, 12], 2] = 0.037
  body_quat = np.zeros((frames, 30, 4), dtype=np.float32)
  body_quat[..., 0] = 1.0
  np.savez(
    tmp_path / "walk.npz",
    fps=np.asarray([50.0]),
    robot=np.asarray("unitree_g1"),
    joint_pos=np.zeros((frames, 29), dtype=np.float32),
    joint_vel=np.zeros((frames, 29), dtype=np.float32),
    body_pos_w=body_pos,
    body_quat_w=body_quat,
    body_lin_vel_w=np.zeros((frames, 30, 3), dtype=np.float32),
    body_ang_vel_w=np.zeros((frames, 30, 3), dtype=np.float32),
    babel_source=np.asarray("ACCAD/walk.npz"),
    babel_categories=np.asarray(["walk"]),
  )

  summary = diagnose("test", str(tmp_path / "*.npz"), window_frames=4)

  assert summary.clips == 1
  assert summary.unique_sources == 1
  assert summary.candidate_windows == 5
  assert summary.kept_windows == 5
  assert summary.categories == (("walk", 1),)


def test_filter_rejects_windows_overlapping_bad_frames() -> None:
  frames = 8
  state = np.zeros((frames, 71), dtype=np.float32)
  state[:, 2] = 0.75
  state[:, 3] = 1.0
  body_pos = np.zeros((frames, 30, 3), dtype=np.float32)
  body_pos[:, 0, 2] = 0.75
  body_pos[:, [6, 12], 2] = 0.037
  body_quat = np.zeros((frames, 30, 4), dtype=np.float32)
  body_quat[..., 0] = 1.0
  state[3, 7] = 3.0

  starts, rejected = filter_window_starts(
    state, body_pos, body_quat, columns=4, fps=50.0
  )

  assert starts.tolist() == [4]
  assert rejected["fast_root"] == 4


def test_quality_filter_keeps_non_locomotion_poses() -> None:
  frames = 8
  state = np.zeros((frames, 71), dtype=np.float32)
  state[:, 2] = 0.3
  state[:, 3] = 1.0
  body_pos = np.zeros((frames, 30, 3), dtype=np.float32)
  body_pos[:, 0, 2] = 0.3
  body_pos[:, [6, 12], 2] = 0.037
  body_quat = np.zeros((frames, 30, 4), dtype=np.float32)
  body_quat[..., 0] = 1.0

  locomotion, rejected = filter_window_starts(
    state, body_pos, body_quat, columns=4, fps=50.0
  )
  quality, quality_rejected = filter_window_starts(
    state,
    body_pos,
    body_quat,
    columns=4,
    fps=50.0,
    cfg=QUALITY_MOTION_FILTER,
  )

  assert locomotion.size == 0
  assert rejected["low_root"] == 5
  assert quality.tolist() == [0, 1, 2, 3, 4]
  assert "low_root" not in quality_rejected
