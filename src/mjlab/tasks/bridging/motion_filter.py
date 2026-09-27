"""Filter retargeted motion windows using only recorded kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

_G1_FOOT_BODY_IDS = (6, 12)
_G1_FOOT_SITE = (0.04, 0.0, -0.037)


@dataclass(frozen=True)
class MotionFilterCfg:
  """Optional limits applied to retargeted G1 kinematics."""

  min_root_up: float | None = None
  min_root_height: float | None = None
  max_root_xy_speed: float | None = None
  max_root_z_speed: float | None = None
  max_root_angular_speed: float | None = None
  max_airborne_foot_height: float | None = None
  max_foot_penetration: float | None = 0.06
  min_airborne_time_s: float = 0.04


QUALITY_MOTION_FILTER = MotionFilterCfg(
  max_root_xy_speed=8.0,
  max_root_z_speed=4.0,
  max_root_angular_speed=12.0,
)
LOCOMOTION_MOTION_FILTER = MotionFilterCfg(
  min_root_up=0.7,
  min_root_height=0.4,
  max_root_xy_speed=2.5,
  max_root_z_speed=1.25,
  max_root_angular_speed=5.0,
  max_airborne_foot_height=0.05,
)
DEFAULT_MOTION_FILTER = LOCOMOTION_MOTION_FILTER
FilterProfile = Literal["quality", "locomotion", "none"]


def motion_filter(profile: FilterProfile) -> MotionFilterCfg | None:
  """Return the named filter profile."""
  if profile == "quality":
    return QUALITY_MOTION_FILTER
  if profile == "locomotion":
    return LOCOMOTION_MOTION_FILTER
  if profile == "none":
    return None
  raise ValueError(f"Unknown motion filter profile: {profile}")


def filter_window_starts(
  state: np.ndarray,
  body_pos: np.ndarray,
  body_quat: np.ndarray,
  columns: int,
  fps: float,
  cfg: MotionFilterCfg = DEFAULT_MOTION_FILTER,
) -> tuple[np.ndarray, dict[str, int]]:
  """Return window starts accepted by the selected kinematic limits."""
  if fps <= 0 or columns < 2 or cfg.min_airborne_time_s <= 0:
    raise ValueError("fps, window length, and airborne time must be positive")
  if body_pos.shape[1:] != (30, 3) or body_quat.shape[1:] != (30, 4):
    raise ValueError("G1 filtering requires all 30 body poses in model order")
  if len(state) != len(body_pos) or len(state) != len(body_quat):
    raise ValueError("state and body poses must have the same frame count")
  if len(state) < columns:
    return np.empty(0, dtype=np.int64), {}

  ankle_pos = body_pos[:, _G1_FOOT_BODY_IDS]
  ankle_quat = body_quat[:, _G1_FOOT_BODY_IDS]
  site = np.asarray(_G1_FOOT_SITE, dtype=np.float32)
  vector = ankle_quat[..., 1:]
  feet = (
    ankle_pos
    + site
    + 2 * np.cross(vector, np.cross(vector, site) + ankle_quat[..., :1] * site)
  )
  if not np.isfinite(feet).all():
    raise ValueError("Foot body poses contain nonfinite values")
  floor = float(np.quantile(feet[..., 2].min(axis=1), 0.02))
  foot_height = feet[..., 2] - floor
  bad_frames = {
    "nonfinite": ~np.isfinite(state).all(axis=1),
  }
  if cfg.min_root_up is not None:
    root_quat = state[:, 3:7]
    root_up = 1 - 2 * (root_quat[:, 1] ** 2 + root_quat[:, 2] ** 2)
    bad_frames["tilted"] = root_up < cfg.min_root_up
  if cfg.min_root_height is not None:
    bad_frames["low_root"] = state[:, 2] - floor < cfg.min_root_height
  if (
    cfg.max_root_xy_speed is not None
    or cfg.max_root_z_speed is not None
    or cfg.max_root_angular_speed is not None
  ):
    fast = np.zeros(len(state), dtype=bool)
    if cfg.max_root_xy_speed is not None:
      fast |= np.linalg.vector_norm(state[:, 7:9], axis=-1) > cfg.max_root_xy_speed
    if cfg.max_root_z_speed is not None:
      fast |= np.abs(state[:, 9]) > cfg.max_root_z_speed
    if cfg.max_root_angular_speed is not None:
      fast |= (
        np.linalg.vector_norm(state[:, 10:13], axis=-1) > cfg.max_root_angular_speed
      )
    bad_frames["fast_root"] = fast
  if cfg.max_airborne_foot_height is not None:
    event_frames = max(1, round(cfg.min_airborne_time_s * fps))
    airborne = foot_height.min(axis=1) > cfg.max_airborne_foot_height
    if event_frames > 1:
      count = np.convolve(
        airborne.astype(np.int32),
        np.ones(event_frames, dtype=np.int32),
        mode="same",
      )
      airborne = count >= event_frames
    bad_frames["airborne"] = airborne
  if cfg.max_foot_penetration is not None:
    bad_frames["penetration"] = foot_height.min(axis=1) < -cfg.max_foot_penetration
  rejected = {}
  for name, bad in bad_frames.items():
    counts = np.concatenate(([0], np.cumsum(bad, dtype=np.int64)))
    rejected[name] = (counts[columns:] - counts[:-columns]) > 0
  valid = ~np.logical_or.reduce(tuple(rejected.values()))
  return np.flatnonzero(valid), {
    name: int(mask.sum()) for name, mask in rejected.items()
  }
