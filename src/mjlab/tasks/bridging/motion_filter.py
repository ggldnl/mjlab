"""Filter retargeted motion windows using only recorded kinematics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_G1_FOOT_BODY_IDS = (6, 12)
_G1_FOOT_SITE = (0.04, 0.0, -0.037)


@dataclass(frozen=True)
class MotionFilterCfg:
  """Limits for grounded, upright, moderate G1 motion."""

  min_root_up: float = 0.7
  min_root_height: float = 0.4
  max_root_xy_speed: float = 2.5
  max_root_z_speed: float = 1.25
  max_root_angular_speed: float = 5.0
  max_airborne_foot_height: float = 0.05
  max_foot_penetration: float = 0.06
  min_airborne_time_s: float = 0.04


DEFAULT_MOTION_FILTER = MotionFilterCfg()


def filter_window_starts(
  state: np.ndarray,
  body_pos: np.ndarray,
  body_quat: np.ndarray,
  columns: int,
  fps: float,
  cfg: MotionFilterCfg = DEFAULT_MOTION_FILTER,
) -> tuple[np.ndarray, dict[str, int]]:
  """Return starts of windows without flight, falls, or fast root motion."""
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
  root_quat = state[:, 3:7]
  root_up = 1 - 2 * (root_quat[:, 1] ** 2 + root_quat[:, 2] ** 2)
  event_frames = max(1, round(cfg.min_airborne_time_s * fps))

  def sustained(event: np.ndarray) -> np.ndarray:
    if event_frames == 1:
      return event
    count = np.convolve(
      event.astype(np.int32), np.ones(event_frames, dtype=np.int32), mode="same"
    )
    return count >= event_frames

  bad_frames = {
    "nonfinite": ~np.isfinite(state).all(axis=1),
    "tilted": root_up < cfg.min_root_up,
    "low_root": state[:, 2] - floor < cfg.min_root_height,
    "fast_root": (np.linalg.vector_norm(state[:, 7:9], axis=-1) > cfg.max_root_xy_speed)
    | (np.abs(state[:, 9]) > cfg.max_root_z_speed)
    | (np.linalg.vector_norm(state[:, 10:13], axis=-1) > cfg.max_root_angular_speed),
    "airborne": sustained(foot_height.min(axis=1) > cfg.max_airborne_foot_height),
    "penetration": foot_height.min(axis=1) < -cfg.max_foot_penetration,
  }
  rejected = {}
  for name, bad in bad_frames.items():
    counts = np.concatenate(([0], np.cumsum(bad, dtype=np.int64)))
    rejected[name] = (counts[columns:] - counts[:-columns]) > 0
  valid = ~np.logical_or.reduce(tuple(rejected.values()))
  return np.flatnonzero(valid), {
    name: int(mask.sum()) for name, mask in rejected.items()
  }
