"""LAFAN1 motion windows for endpoint conditioned inbetweening."""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_from_matrix,
  quat_mul,
  yaw_quat,
)

DEFAULT_MOTIONS = ("data/lafan1_g1/motions/*.npz",)


@dataclass(frozen=True)
class Layout:
  joints: int

  @property
  def width(self) -> int:
    return 15 + 2 * self.joints

  @property
  def root_linear_velocity(self) -> slice:
    return slice(9, 12)

  @property
  def root_angular_velocity(self) -> slice:
    return slice(12, 15)

  @property
  def joint_positions(self) -> slice:
    return slice(15, 15 + self.joints)

  @property
  def joint_velocities(self) -> slice:
    return slice(15 + self.joints, self.width)


def rot6d(quat: torch.Tensor) -> torch.Tensor:
  return matrix_from_quat(quat)[..., :, :2].transpose(-1, -2).flatten(-2)


def quat_from_rot6d(features: torch.Tensor) -> torch.Tensor:
  first = F.normalize(features[..., :3], dim=-1)
  first = torch.where(
    first.square().sum(-1, keepdim=True) < 1e-8,
    first.new_tensor([1.0, 0.0, 0.0]),
    first,
  )
  second = features[..., 3:6]
  second = second - (first * second).sum(-1, keepdim=True) * first
  fallback = torch.where(
    first[..., :1].abs() < 0.9,
    first.new_tensor([1.0, 0.0, 0.0]),
    first.new_tensor([0.0, 1.0, 0.0]),
  )
  fallback = fallback - (first * fallback).sum(-1, keepdim=True) * first
  second = F.normalize(
    torch.where(second.square().sum(-1, keepdim=True) < 1e-8, fallback, second),
    dim=-1,
  )
  return quat_from_matrix(
    torch.stack((first, second, torch.cross(first, second, dim=-1)), dim=-1)
  )


def encode(states: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
  """Encode dynamic states in A's heading frame."""
  if states.ndim != 3 or (states.shape[-1] - 13) % 2:
    raise ValueError("states must have shape (batch, time, 13 + 2 * joints)")
  if anchor.shape != states.shape[:1] + states.shape[2:]:
    raise ValueError("anchor must have shape (batch, state)")
  joints = (states.shape[-1] - 13) // 2
  heading = yaw_quat(anchor[:, 3:7])[:, None].expand(-1, states.shape[1], -1)
  origin = anchor[:, None, :3].clone()
  origin[..., 2] = 0.0
  orientation = quat_mul(quat_conjugate(heading), states[..., 3:7])
  return torch.cat(
    (
      quat_apply_inverse(heading, states[..., :3] - origin),
      rot6d(orientation),
      quat_apply_inverse(heading, states[..., 7:10]),
      quat_apply_inverse(heading, states[..., 10:13]),
      states[..., 13 : 13 + joints],
      states[..., 13 + joints :],
    ),
    dim=-1,
  )


def decode(
  features: torch.Tensor, anchor: torch.Tensor, layout: Layout
) -> torch.Tensor:
  """Decode features to world dynamic states."""
  if features.ndim != 3 or features.shape[-1] != layout.width:
    raise ValueError("features have the wrong shape")
  if anchor.shape != (features.shape[0], 13 + 2 * layout.joints):
    raise ValueError("anchor has the wrong shape")
  heading = yaw_quat(anchor[:, 3:7])[:, None].expand(-1, features.shape[1], -1)
  origin = anchor[:, None, :3].clone()
  origin[..., 2] = 0.0
  return torch.cat(
    (
      origin + quat_apply(heading, features[..., :3]),
      quat_mul(heading, quat_from_rot6d(features[..., 3:9])),
      quat_apply(heading, features[..., layout.root_linear_velocity]),
      quat_apply(heading, features[..., layout.root_angular_velocity]),
      features[..., layout.joint_positions],
      features[..., layout.joint_velocities],
    ),
    dim=-1,
  )


@dataclass
class Normalizer:
  mean: torch.Tensor
  std: torch.Tensor

  @classmethod
  def fit(cls, features: torch.Tensor) -> Normalizer:
    flat = features.flatten(0, 1)
    return cls(flat.mean(0), flat.std(0).clamp_min(1e-3))

  def normalize(self, features: torch.Tensor) -> torch.Tensor:
    return (features - self.mean) / self.std

  def denormalize(self, features: torch.Tensor) -> torch.Tensor:
    return features * self.std + self.mean


@dataclass(frozen=True)
class MotionCorpus:
  """Retargeted kinematic clips kept separate at window boundaries."""

  states: torch.Tensor
  starts: torch.Tensor
  names: tuple[str, ...]
  fps: float
  num_joints: int

  @property
  def num_windows(self) -> int:
    return int(self.starts.numel())


def motion_files(patterns: tuple[str, ...]) -> tuple[Path, ...]:
  found = sorted({Path(name) for pattern in patterns for name in glob.glob(pattern)})
  if not found:
    raise FileNotFoundError(f"No motion files match {patterns}")
  return tuple(found)


def load_motions(
  patterns: tuple[str, ...],
  columns: int,
  device: str,
  split: str = "train",
  holdout: int = 8,
) -> MotionCorpus:
  """Load G1 NPZ clips and split by whole motion files."""
  if split not in ("train", "eval"):
    raise ValueError("split must be train or eval")
  if columns < 2 or holdout < 2:
    raise ValueError("columns must exceed one and holdout must exceed one")
  files = motion_files(patterns)
  selected = [
    path
    for index, path in enumerate(files)
    if (index % holdout == 0) == (split == "eval")
  ]
  if not selected:
    raise ValueError(f"No {split} clips remain after the file split")

  clips: list[torch.Tensor] = []
  starts: list[torch.Tensor] = []
  names: list[str] = []
  fps: float | None = None
  joints: int | None = None
  offset = 0
  for path in selected:
    with np.load(path, allow_pickle=False) as raw:
      clip_fps = float(np.asarray(raw["fps"]).reshape(-1)[0])
      joint_pos = np.asarray(raw["joint_pos"], dtype=np.float32)
      joint_vel = np.asarray(raw["joint_vel"], dtype=np.float32)
      body_pos = np.asarray(raw["body_pos_w"], dtype=np.float32)
      body_quat = np.asarray(raw["body_quat_w"], dtype=np.float32)
      body_lin_vel = np.asarray(raw["body_lin_vel_w"], dtype=np.float32)
      body_ang_vel = np.asarray(raw["body_ang_vel_w"], dtype=np.float32)
    if len(joint_pos) < columns:
      continue
    if fps is not None and abs(clip_fps - fps) > 1e-6:
      raise ValueError("All motion clips must have the same frame rate")
    if joints is not None and joint_pos.shape[1] != joints:
      raise ValueError("All motion clips must use the same robot")
    fps = clip_fps
    joints = joint_pos.shape[1]
    state = np.concatenate(
      (
        body_pos[:, 0],
        body_quat[:, 0],
        body_lin_vel[:, 0],
        body_ang_vel[:, 0],
        joint_pos,
        joint_vel,
      ),
      axis=-1,
    )
    clips.append(torch.from_numpy(state))
    starts.append(torch.arange(offset, offset + len(state) - columns + 1))
    names.append(path.stem)
    offset += len(state)
  if not clips or fps is None or joints is None:
    raise ValueError(f"No {split} clip is at least {columns} frames long")
  return MotionCorpus(
    states=torch.cat(clips).to(device),
    starts=torch.cat(starts).to(device),
    names=tuple(names),
    fps=fps,
    num_joints=joints,
  )


class Windows:
  """Sample A to B windows directly from kinematic clips."""

  def __init__(
    self,
    data: MotionCorpus,
    history: int,
    future: int,
    min_steps: int,
    max_steps: int,
  ) -> None:
    if history < 1 or future < 1 or min_steps < 2 or max_steps < min_steps:
      raise ValueError("invalid boundary or duration bounds")
    self.data = data
    self.history = history
    self.future = future
    self.min_steps = min_steps
    self.max_steps = max_steps
    self.columns = history + max_steps + future - 1
    self.layout = Layout(data.num_joints)
    self.offsets = torch.arange(self.columns, device=data.states.device)

  def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local pose windows and A to B durations."""
    if count < 1:
      raise ValueError("count must be positive")
    device = self.data.states.device
    picked = torch.randint(self.data.starts.numel(), (count,), device=device)
    rows = self.data.starts[picked, None] + self.offsets
    states = self.data.states[rows]
    duration = torch.randint(
      self.min_steps, self.max_steps + 1, (count,), device=device
    )
    features = encode(states, states[:, self.history - 1])
    target_last = self.history - 1 + duration + self.future - 1
    time = torch.arange(self.columns, device=device)[None]
    last = features[torch.arange(count, device=device), target_last]
    features = torch.where(
      (time > target_last[:, None])[..., None], last[:, None], features
    )
    return features, duration

  def states(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full state windows and durations for evaluation."""
    if count < 1:
      raise ValueError("count must be positive")
    device = self.data.states.device
    picked = torch.randint(self.data.starts.numel(), (count,), device=device)
    rows = self.data.starts[picked, None] + self.offsets
    states = self.data.states[rows]
    duration = torch.randint(
      self.min_steps, self.max_steps + 1, (count,), device=device
    )
    return states, duration


def bridge_mask(
  batch: int,
  columns: int,
  layout: Layout,
  history: int,
  future: int,
  duration: torch.Tensor,
) -> torch.Tensor:
  """Condition on pre A history and B's short continuation."""
  if duration.shape != (batch,) or bool(
    ((duration < 2) | (duration > columns - history - future + 1)).any()
  ):
    raise ValueError("duration must fit the prediction horizon")
  mask = torch.zeros(
    batch, columns, layout.width, dtype=torch.bool, device=duration.device
  )
  mask[:, :history] = True
  rows = history - 1 + duration
  offsets = torch.arange(future, device=duration.device)
  indexes = torch.arange(batch, device=duration.device)[:, None]
  mask[indexes, rows[:, None] + offsets] = True
  return mask


def training_mask(
  layout: Layout,
  columns: int,
  history: int,
  future: int,
  duration: torch.Tensor,
  bridge_probability: float = 1.0,
) -> torch.Tensor:
  """Use the deployment mask, with optional generic inpainting examples."""
  if not 0.0 <= bridge_probability <= 1.0:
    raise ValueError("bridge_probability must lie in [0, 1]")
  batch = duration.shape[0]
  device = duration.device
  mask = torch.rand(batch, columns, layout.width, device=device) < 0.25
  temporal = torch.rand(batch, device=device) < 0.5
  if bool(temporal.any()):
    frames = torch.rand(int(temporal.sum()), columns, 1, device=device) < 0.25
    mask[temporal] = frames.expand(-1, -1, layout.width)
  mask[:, 0] = True
  mask[:, -1] = True
  deployment = torch.rand(batch, device=device) < bridge_probability
  if bool(deployment.any()):
    fixed = bridge_mask(batch, columns, layout, history, future, duration)
    mask[deployment] = fixed[deployment]
  return mask
