"""Collect skill rollouts and stitch compatible pieces into one dataset.

Run

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.skill_rollouts.collect
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset import dataset
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  ROOT_STATE_DIM,
  SKILL_ROLLOUT_DATASET,
  RolloutCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.skills import SKILLS
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


@dataclass
class SkillRolloutCfg(RolloutCfg):
  """Skills to record and how their rollouts may be joined."""

  skills: tuple[str, ...] = tuple(SKILLS)
  checkpoints: tuple[str, ...] = ()
  path: Path = SKILL_ROLLOUT_DATASET
  seed: int = 0
  min_piece_steps: int = 15
  candidates: int = 64
  max_stitches: int = 0
  """Maximum output trajectories. Zero uses every eligible first rollout."""

  position_jitter: float = 0.10
  heading_jitter: float = 0.15
  max_height_gap: float = 0.20
  max_tilt_gap: float = 0.45
  max_linear_velocity_gap: float = 1.5
  max_angular_velocity_gap: float = 2.5
  max_joint_gap: float = 0.55
  max_joint_velocity_gap: float = 3.0


@dataclass(frozen=True)
class Rollout:
  """One contiguous recorded skill rollout."""

  source: int
  states: np.ndarray
  phase: np.ndarray


@dataclass(frozen=True)
class Seam:
  """Intrinsic state gaps at a proposed join."""

  height: float
  tilt: float
  linear_velocity: float
  angular_velocity: float
  joint: float
  joint_velocity: float

  def reachable(self, cfg: SkillRolloutCfg) -> bool:
    return (
      self.height <= cfg.max_height_gap
      and self.tilt <= cfg.max_tilt_gap
      and self.linear_velocity <= cfg.max_linear_velocity_gap
      and self.angular_velocity <= cfg.max_angular_velocity_gap
      and self.joint <= cfg.max_joint_gap
      and self.joint_velocity <= cfg.max_joint_velocity_gap
    )

  def score(self, cfg: SkillRolloutCfg) -> float:
    values = (
      self.height,
      self.tilt,
      self.linear_velocity,
      self.angular_velocity,
      self.joint,
      self.joint_velocity,
    )
    limits = (
      cfg.max_height_gap,
      cfg.max_tilt_gap,
      cfg.max_linear_velocity_gap,
      cfg.max_angular_velocity_gap,
      cfg.max_joint_gap,
      cfg.max_joint_velocity_gap,
    )
    return max(
      value / max(limit, 1e-6) for value, limit in zip(values, limits, strict=True)
    )


def _yaw(quaternion: np.ndarray) -> float:
  w, x, y, z = quaternion
  return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _yaw_quat(angle: float) -> np.ndarray:
  return np.asarray([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


def _quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
  lw, lx, ly, lz = np.moveaxis(left, -1, 0)
  rw, rx, ry, rz = np.moveaxis(right, -1, 0)
  return np.stack(
    (
      lw * rw - lx * rx - ly * ry - lz * rz,
      lw * rx + lx * rw + ly * rz - lz * ry,
      lw * ry - lx * rz + ly * rw + lz * rx,
      lw * rz + lx * ry - ly * rx + lz * rw,
    ),
    axis=-1,
  )


def _rotate_z(values: np.ndarray, angle: float) -> np.ndarray:
  out = values.copy()
  cosine, sine = np.cos(angle), np.sin(angle)
  out[..., 0] = cosine * values[..., 0] - sine * values[..., 1]
  out[..., 1] = sine * values[..., 0] + cosine * values[..., 1]
  return out


def seam(first: np.ndarray, second: np.ndarray) -> Seam:
  """Measure pose and velocity compatibility after matching headings."""
  joints = (first.shape[0] - ROOT_STATE_DIM) // 2
  angle = _yaw(first[3:7]) - _yaw(second[3:7])
  aligned_quat = _quat_mul(_yaw_quat(angle), second[3:7])
  dot = abs(float(np.dot(first[3:7], aligned_quat))) / (
    np.linalg.norm(first[3:7]) * np.linalg.norm(aligned_quat)
  )
  dot = min(dot, 1.0)
  return Seam(
    height=abs(float(first[2] - second[2])),
    tilt=2 * float(np.arccos(dot)),
    linear_velocity=float(np.linalg.norm(first[7:10] - _rotate_z(second[7:10], angle))),
    angular_velocity=float(
      np.linalg.norm(first[10:13] - _rotate_z(second[10:13], angle))
    ),
    joint=float(
      np.sqrt(
        np.mean(
          (
            first[ROOT_STATE_DIM : ROOT_STATE_DIM + joints]
            - second[ROOT_STATE_DIM : ROOT_STATE_DIM + joints]
          )
          ** 2
        )
      )
    ),
    joint_velocity=float(
      np.sqrt(
        np.mean(
          (first[ROOT_STATE_DIM + joints :] - second[ROOT_STATE_DIM + joints :]) ** 2
        )
      )
    ),
  )


def place_suffix(
  states: np.ndarray,
  start: np.ndarray,
  destination: np.ndarray,
  heading_offset: float,
) -> np.ndarray:
  """Move a rollout suffix so its first root lies beside the preceding piece."""
  angle = _yaw(destination[3:7]) - _yaw(start[3:7]) + heading_offset
  out = states.copy()
  out[:, :3] = destination[:3] + _rotate_z(states[:, :3] - start[:3], angle)
  out[:, 3:7] = _quat_mul(_yaw_quat(angle), states[:, 3:7])
  out[:, 7:10] = _rotate_z(states[:, 7:10], angle)
  out[:, 10:13] = _rotate_z(states[:, 10:13], angle)
  return out


def rollouts(
  states: np.ndarray,
  source: np.ndarray,
  trajectory: np.ndarray,
  frame: np.ndarray,
  phase: np.ndarray,
) -> list[Rollout]:
  """Split recorded rows into contiguous, time ordered rollouts."""
  found: list[Rollout] = []
  for source_id in np.unique(source):
    rows_for_source = np.flatnonzero(source == source_id)
    for trajectory_id in np.unique(trajectory[rows_for_source]):
      rows = rows_for_source[trajectory[rows_for_source] == trajectory_id]
      rows = rows[np.argsort(frame[rows])]
      edges = np.flatnonzero(np.diff(frame[rows]) != 1) + 1
      for part in np.split(rows, edges):
        if len(part):
          found.append(Rollout(int(source_id), states[part], phase[part]))
  return found


def stitch(
  recorded: list[Rollout], cfg: SkillRolloutCfg
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Join compatible random rollout pieces into the shared row format."""
  if cfg.min_piece_steps < 1 or cfg.candidates < 1 or cfg.max_stitches < 0:
    raise ValueError(
      "min_piece_steps and candidates must be positive; max_stitches cannot be negative"
    )
  if cfg.position_jitter < 0 or cfg.heading_jitter < 0:
    raise ValueError("position_jitter and heading_jitter cannot be negative")
  rng = np.random.default_rng(cfg.seed)
  usable = [r for r in recorded if len(r.states) >= 2 * cfg.min_piece_steps]
  if len({r.source for r in usable}) < 2:
    raise ValueError("Stitching needs eligible rollouts from at least two skills")
  rng.shuffle(usable)
  first_rollouts = usable
  if cfg.max_stitches:
    first_rollouts = usable[: cfg.max_stitches]

  state_parts: list[np.ndarray] = []
  source_parts: list[np.ndarray] = []
  env_parts: list[np.ndarray] = []
  trajectory_parts: list[np.ndarray] = []
  frame_parts: list[np.ndarray] = []
  phase_parts: list[np.ndarray] = []
  for first in first_rollouts:
    first_at = int(
      rng.integers(cfg.min_piece_steps - 1, len(first.states) - cfg.min_piece_steps)
    )
    candidates = [r for r in usable if r.source != first.source]
    choices: list[tuple[float, Rollout, int]] = []
    for _ in range(cfg.candidates):
      second = candidates[int(rng.integers(len(candidates)))]
      second_at = int(
        rng.integers(cfg.min_piece_steps, len(second.states) - cfg.min_piece_steps + 1)
      )
      gap = seam(first.states[first_at], second.states[second_at])
      if gap.reachable(cfg):
        choices.append((gap.score(cfg), second, second_at))
    if not choices:
      continue
    _, second, second_at = min(choices, key=lambda item: item[0])

    offset = rng.uniform(-cfg.position_jitter, cfg.position_jitter, size=3)
    offset[2] = 0.0
    destination = first.states[first_at].copy()
    destination[:3] += _rotate_z(offset, _yaw(destination[3:7]))
    destination[2] = second.states[second_at, 2]
    suffix = place_suffix(
      second.states[second_at:],
      second.states[second_at],
      destination,
      float(rng.uniform(-cfg.heading_jitter, cfg.heading_jitter)),
    )
    joined = np.concatenate((first.states[: first_at + 1], suffix))
    stitch_id = len(state_parts)
    state_parts.append(joined)
    source_parts.append(
      np.concatenate(
        (
          np.full(first_at + 1, first.source, dtype=np.int16),
          np.full(len(suffix), second.source, dtype=np.int16),
        )
      )
    )
    env_parts.append(np.full(len(joined), stitch_id, dtype=np.int32))
    trajectory_parts.append(np.full(len(joined), stitch_id, dtype=np.int32))
    frame_parts.append(np.arange(len(joined), dtype=np.int32))
    phase_parts.append(
      np.concatenate((first.phase[: first_at + 1], second.phase[second_at:]))
    )

  if not state_parts:
    raise ValueError(
      "No reachable seams found. Relax the gap limits or record more rollouts"
    )
  return (
    np.concatenate(state_parts),
    np.concatenate(source_parts),
    np.concatenate(env_parts),
    np.concatenate(trajectory_parts),
    np.concatenate(frame_parts),
    np.concatenate(phase_parts),
  )


def collect(cfg: SkillRolloutCfg) -> Path:
  """Record configured skills, stitch their rollouts, and write one dataset."""
  unknown = set(cfg.skills) - SKILLS.keys()
  if unknown:
    raise ValueError(f"Unknown skills: {', '.join(sorted(unknown))}")
  if len(cfg.skills) < 2:
    raise ValueError("Stitching needs at least two skills")
  if cfg.checkpoints and len(cfg.checkpoints) != len(cfg.skills):
    raise ValueError("checkpoints must be empty or contain one path per skill")

  states: list[np.ndarray] = []
  sources: list[np.ndarray] = []
  trajectories: list[np.ndarray] = []
  frames: list[np.ndarray] = []
  phases: list[np.ndarray] = []
  fps: float | None = None
  for source, name in enumerate(cfg.skills):
    task = SKILLS[name]
    env_cfg = load_env_cfg(task)
    rate = dataset.control_rate(env_cfg)
    if fps is not None and abs(rate - fps) > 1e-6:
      raise ValueError("All recorded skills must use the same control rate")
    fps = rate
    explicit = cfg.checkpoints[source] if cfg.checkpoints else None
    experiment = load_rl_cfg(task).experiment_name
    checkpoint = dataset.find_checkpoint(
      (experiment,), explicit, hint=f" Train it with `uv run train {task}`."
    )
    rows, _, trajectory, frame, phase, _ = dataset.record(
      task, env_cfg, checkpoint, cfg, name
    )
    states.append(rows)
    sources.append(np.full(len(rows), source, dtype=np.int16))
    trajectories.append(trajectory)
    frames.append(frame)
    phases.append(phase)

  if fps is None:
    raise ValueError("At least two skills are required")
  recorded = rollouts(
    np.concatenate(states),
    np.concatenate(sources),
    np.concatenate(trajectories),
    np.concatenate(frames),
    np.concatenate(phases),
  )
  state, source, env_id, trajectory, frame, phase = stitch(recorded, cfg)
  print(f"[dataset] kept {int(trajectory.max()) + 1} reachable stitched rollouts")
  return dataset.write(
    cfg.path,
    [state],
    [env_id],
    [trajectory],
    [frame],
    [source],
    cfg.skills,
    fps,
    phases=[phase],
    trajectory_ids_global=True,
  )


if __name__ == "__main__":
  collect(tyro.cli(SkillRolloutCfg, config=mjlab.TYRO_FLAGS))
