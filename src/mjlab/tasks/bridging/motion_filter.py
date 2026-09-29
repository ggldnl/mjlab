"""Filter retargeted motion windows using only recorded kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Literal

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.booster_t1.t1_constants import get_spec as get_t1_spec
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec as get_g1_spec

_ROBOT_SPECS = {"unitree_g1": get_g1_spec, "booster_t1": get_t1_spec}


@cache
def _foot_sites(robot: str) -> tuple[tuple[int, int], np.ndarray, int]:
  try:
    model = _ROBOT_SPECS[robot]().compile()
  except KeyError:
    raise ValueError(f"Unsupported robot {robot!r}") from None
  sites = []
  for name in ("left_foot", "right_foot"):
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site < 0:
      raise ValueError(f"{robot} has no {name} site")
    sites.append(site)
  bodies = tuple(int(model.site_bodyid[site]) - 1 for site in sites)
  offsets = np.asarray([model.site_pos[site] for site in sites], dtype=np.float32)
  return (bodies[0], bodies[1]), offsets, model.nbody - 1


def robot_foot_positions(
  body_pos: np.ndarray, body_quat: np.ndarray, robot: str
) -> np.ndarray:
  """Return the selected robot's two sole sites in world coordinates."""
  body_ids, sites, bodies = _foot_sites(robot)
  if body_pos.shape[1:] != (bodies, 3) or body_quat.shape[1:] != (bodies, 4):
    raise ValueError(f"{robot} foot positions require all {bodies} body poses")
  ankle_pos = body_pos[:, body_ids]
  ankle_quat = body_quat[:, body_ids]
  vector = ankle_quat[..., 1:]
  feet = (
    ankle_pos
    + sites
    + 2 * np.cross(vector, np.cross(vector, sites) + ankle_quat[..., :1] * sites)
  )
  if not np.isfinite(feet).all():
    raise ValueError("Foot body poses contain nonfinite values")
  return feet


def g1_foot_positions(body_pos: np.ndarray, body_quat: np.ndarray) -> np.ndarray:
  """Return the two G1 sole sites in world coordinates."""
  return robot_foot_positions(body_pos, body_quat, "unitree_g1")


def align_robot_floor(
  body_pos: np.ndarray,
  body_quat: np.ndarray,
  robot: str,
  floor_quantile: float = 0.1,
  max_offset: float = 0.05,
  max_penetration: float = 0.03,
  contact_height: float = 0.05,
  min_contact_fraction: float = 0.2,
) -> tuple[np.ndarray, dict[str, float | str]]:
  """Align a consistent floor bias and report clips that need rejection."""
  if not 0.0 < floor_quantile < 0.5:
    raise ValueError("floor_quantile must lie between zero and 0.5")
  if min(max_offset, max_penetration, contact_height, min_contact_fraction) < 0:
    raise ValueError("floor QA limits must be nonnegative")

  lower_sole = robot_foot_positions(body_pos, body_quat, robot)[..., 2].min(axis=1)
  if not len(lower_sole):
    raise ValueError("floor alignment requires at least one frame")
  floor = float(np.quantile(lower_sole, floor_quantile))
  offset = max(0.0, -floor)
  aligned = lower_sole + offset
  minimum = float(aligned.min())
  contact_fraction = float(np.mean(aligned <= contact_height))
  reason = ""
  if offset > max_offset:
    reason = "floor_offset_too_large"
  elif minimum < -max_penetration:
    reason = "isolated_foot_penetration"
  elif contact_fraction < min_contact_fraction:
    reason = "insufficient_ground_contact"

  metrics: dict[str, float | str] = {
    "status": "rejected" if reason else "accepted",
    "reason": reason,
    "floor_z": floor,
    "offset": offset,
    "minimum_sole_z": minimum,
    "contact_fraction": contact_fraction,
  }
  if reason or offset == 0.0:
    return body_pos, metrics
  corrected = body_pos.copy()
  corrected[..., 2] += offset
  return corrected, metrics


def align_g1_floor(
  body_pos: np.ndarray,
  body_quat: np.ndarray,
  floor_quantile: float = 0.1,
  max_offset: float = 0.05,
  max_penetration: float = 0.03,
  contact_height: float = 0.05,
  min_contact_fraction: float = 0.2,
) -> tuple[np.ndarray, dict[str, float | str]]:
  """Align a consistent floor bias and report clips that need rejection."""
  return align_robot_floor(
    body_pos,
    body_quat,
    "unitree_g1",
    floor_quantile,
    max_offset,
    max_penetration,
    contact_height,
    min_contact_fraction,
  )


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


def motion_bad_frames(
  state: np.ndarray,
  body_pos: np.ndarray,
  body_quat: np.ndarray,
  fps: float,
  cfg: MotionFilterCfg = DEFAULT_MOTION_FILTER,
  robot: str = "unitree_g1",
) -> dict[str, np.ndarray]:
  """Mark bad frames once so every kinematic consumer uses the same cuts."""
  if fps <= 0 or cfg.min_airborne_time_s <= 0:
    raise ValueError("fps and airborne time must be positive")
  if len(state) != len(body_pos) or len(state) != len(body_quat):
    raise ValueError("state and body poses must have the same frame count")

  feet = robot_foot_positions(body_pos, body_quat, robot)
  floor = float(np.quantile(feet[..., 2].min(axis=1), 0.02))
  foot_height = feet[..., 2] - floor
  bad_frames: dict[str, np.ndarray] = {
    "nonfinite": np.asarray(~np.isfinite(state).all(axis=1), dtype=bool),
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
  return bad_frames


def filter_window_starts(
  state: np.ndarray,
  body_pos: np.ndarray,
  body_quat: np.ndarray,
  columns: int,
  fps: float,
  cfg: MotionFilterCfg = DEFAULT_MOTION_FILTER,
  robot: str = "unitree_g1",
) -> tuple[np.ndarray, dict[str, int]]:
  """Return window starts accepted by the selected kinematic limits."""
  if columns < 2:
    raise ValueError("window length must exceed one")
  if len(state) < columns:
    return np.empty(0, dtype=np.int64), {}

  bad_frames = motion_bad_frames(state, body_pos, body_quat, fps, cfg, robot)
  rejected = {}
  for name, bad in bad_frames.items():
    counts = np.concatenate(([0], np.cumsum(bad, dtype=np.int64)))
    rejected[name] = (counts[columns:] - counts[:-columns]) > 0
  valid = ~np.logical_or.reduce(tuple(rejected.values()))
  return np.flatnonzero(valid), {
    name: int(mask.sum()) for name, mask in rejected.items()
  }
