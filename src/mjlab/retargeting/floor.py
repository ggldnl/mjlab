"""Put retargeted clips on the floor and reject the ones that cannot be.

Retargeting leaves clips hovering by a few centimetres, and the hover drifts within a
clip, so every frame is corrected on its own.
"""

from __future__ import annotations

from functools import cache

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.booster_t1.t1_constants import get_spec as get_t1_spec
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec as get_g1_spec

_ROBOT_SPECS = {"unitree_g1": get_g1_spec, "booster_t1": get_t1_spec}


@cache
def _model(robot: str) -> mujoco.MjModel:
  try:
    return _ROBOT_SPECS[robot]().compile()
  except KeyError:
    raise ValueError(f"Unsupported robot {robot!r}") from None


def _foot_bodies(robot: str) -> tuple[int, int]:
  """Body index of each foot site, without the world body, as stored in motion npz."""
  model = _model(robot)
  sites = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    for name in ("left_foot", "right_foot")
  ]
  if min(sites) < 0:
    raise ValueError(f"{robot} has no left_foot and right_foot sites")
  return int(model.site_bodyid[sites[0]]) - 1, int(model.site_bodyid[sites[1]]) - 1


def _rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
  w, xyz = quat[..., :1], quat[..., 1:]
  return vector + 2 * np.cross(xyz, np.cross(xyz, vector) + w * vector)


def _check_bodies(body_pos: np.ndarray, body_quat: np.ndarray, robot: str) -> None:
  bodies = _model(robot).nbody - 1
  if body_pos.shape[1:] != (bodies, 3) or body_quat.shape[1:] != (bodies, 4):
    raise ValueError(f"{robot} needs all {bodies} body poses")


def robot_foot_positions(
  body_pos: np.ndarray, body_quat: np.ndarray, robot: str
) -> np.ndarray:
  """The two sole sites in world coordinates, shape (frames, 2, 3)."""
  _check_bodies(body_pos, body_quat, robot)
  model = _model(robot)
  bodies = list(_foot_bodies(robot))
  sites = np.asarray(
    [
      model.site_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)]
      for name in ("left_foot", "right_foot")
    ],
    dtype=np.float32,
  )
  feet = body_pos[:, bodies] + _rotate(body_quat[:, bodies], sites)
  if not np.isfinite(feet).all():
    raise ValueError("Foot body poses contain nonfinite values")
  return feet


def g1_foot_positions(body_pos: np.ndarray, body_quat: np.ndarray) -> np.ndarray:
  return robot_foot_positions(body_pos, body_quat, "unitree_g1")


@cache
def _foot_points(robot: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Body, side, body-frame point and radius of every foot collision primitive.

  Spheres and capsules become their centres with their radius, boxes and meshes their
  corners and vertices. The lowest point is min(z) - radius for any orientation.
  """
  model = _model(robot)
  bodies, sides, points, radii = [], [], [], []
  for side, body in enumerate(_foot_bodies(robot)):
    for geom in range(model.ngeom):
      if model.geom_bodyid[geom] != body + 1 or not (
        model.geom_contype[geom] or model.geom_conaffinity[geom]
      ):
        continue
      kind, size = model.geom_type[geom], model.geom_size[geom]
      if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        local, radius = np.zeros((1, 3)), size[0]
      elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        local, radius = np.array([[0, 0, -size[1]], [0, 0, size[1]]]), size[0]
      elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        local, radius = (
          np.array(np.meshgrid(*([[-1, 1]] * 3))).reshape(3, -1).T * size,
          0,
        )
      elif kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        local, radius = model.mesh_vert[start : start + count].astype(np.float64), 0
      else:
        raise ValueError(f"{robot} foot geom type {kind} is not supported")
      rotation = np.zeros(9)
      mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
      local = local @ rotation.reshape(3, 3).T + model.geom_pos[geom]
      bodies.append(np.full(len(local), body))
      sides.append(np.full(len(local), side))
      points.append(local)
      radii.append(np.full(len(local), radius))
  if not points:
    raise ValueError(f"{robot} feet have no collision geometry")
  return (
    np.concatenate(bodies),
    np.concatenate(sides),
    np.concatenate(points),
    np.concatenate(radii),
  )


def robot_foot_heights(
  body_pos: np.ndarray, body_quat: np.ndarray, robot: str
) -> np.ndarray:
  """Lowest collision point of each foot, shape (frames, 2)."""
  _check_bodies(body_pos, body_quat, robot)
  bodies, sides, points, radii = _foot_points(robot)
  z = (body_pos[:, bodies] + _rotate(body_quat[:, bodies], points))[..., 2] - radii
  return np.stack((z[:, sides == 0].min(axis=1), z[:, sides == 1].min(axis=1)), axis=1)


def floor_qa(
  body_pos: np.ndarray,
  body_quat: np.ndarray,
  robot: str,
  floor_quantile: float = 0.1,
  max_offset: float = 0.05,
  max_penetration: float | None = None,
  contact_height: float = 0.05,
  min_contact_fraction: float = 0.2,
) -> dict[str, float | str]:
  """Accept or reject a whole clip from its raw sole heights.

  floor_offset_too_large       the clip sinks more than max_offset below the floor
  isolated_foot_penetration    one frame goes deeper than max_penetration (off by default,
                               a long take should not be lost to one glitch)
  insufficient_ground_contact  soles touch the floor in too few frames
  """
  lower_sole = robot_foot_positions(body_pos, body_quat, robot)[..., 2].min(axis=1)
  if not len(lower_sole):
    raise ValueError("floor QA requires at least one frame")
  floor = float(np.quantile(lower_sole, floor_quantile))
  offset = max(0.0, -floor)
  minimum = float((lower_sole + offset).min())
  contact_fraction = float(np.mean(lower_sole + offset <= contact_height))
  reason = ""
  if offset > max_offset:
    reason = "floor_offset_too_large"
  elif max_penetration is not None and minimum < -max_penetration:
    reason = "isolated_foot_penetration"
  elif contact_fraction < min_contact_fraction:
    reason = "insufficient_ground_contact"
  return {
    "status": "rejected" if reason else "accepted",
    "reason": reason,
    "floor_z": floor,
    "minimum_sole_z": minimum,
    "contact_fraction": contact_fraction,
  }


def ground_motion(
  body_pos: np.ndarray,
  body_quat: np.ndarray,
  body_lin_vel: np.ndarray,
  fps: float,
  robot: str,
  contact_speed: float = 0.25,
  contact_band: float = 0.08,
  smoothing_s: float = 0.1,
  max_correction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Shift every frame vertically so its supporting foot touches z = 0.

  1. Lowest collision point of each foot per frame.
  2. Support feet: slower than contact_speed and within contact_band of the clip floor.
  3. Correction is minus the lowest support foot. Frames without support (flight,
     swing) interpolate between their neighbours. No foot may end below the floor.
  4. The correction is smoothed and added to every body height, its time derivative
     to every body vertical velocity.

  Returns corrected positions, corrected velocities and the per-frame correction.
  """
  if fps <= 0:
    raise ValueError("fps must be positive")
  frames = len(body_pos)
  heights = robot_foot_heights(body_pos, body_quat, robot)
  feet_xy = body_pos[:, list(_foot_bodies(robot)), :2]
  speed = np.linalg.vector_norm(np.gradient(feet_xy, axis=0), axis=-1) * fps
  floor = float(np.quantile(heights.min(axis=1), 0.05))
  support = (speed < contact_speed) & (heights - floor < contact_band)
  lowest = np.where(support, heights, np.inf).min(axis=1)
  known = np.isfinite(lowest)
  frame = np.arange(frames)
  if known.any():
    correction = np.interp(frame, frame[known], -lowest[known])
  else:
    correction = np.full(frames, -floor)
  clearance = -heights.min(axis=1)
  correction = np.maximum(correction, clearance)
  width = max(1, round(smoothing_s * fps))
  if 1 < width < frames:
    padded = np.pad(correction, (width // 2, width - 1 - width // 2), mode="edge")
    correction = np.convolve(padded, np.ones(width) / width, mode="valid")
  correction = np.clip(
    np.maximum(correction, clearance), -max_correction, max_correction
  )
  position = body_pos.copy()
  position[..., 2] += correction[:, None].astype(position.dtype)
  velocity = body_lin_vel.copy()
  rate = np.gradient(correction) * fps if frames > 1 else np.zeros(frames)
  velocity[..., 2] += rate[:, None].astype(velocity.dtype)
  return position, velocity, correction
