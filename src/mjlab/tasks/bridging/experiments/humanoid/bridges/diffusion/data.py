"""Position-only motion windows with masked boundary conditioning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import Dataset
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_from_matrix,
  quat_mul,
  yaw_quat,
)


@dataclass(frozen=True)
class Layout:
  joints: int

  @property
  def width(self) -> int:
    return 9 + self.joints

  @property
  def joint_positions(self) -> slice:
    return slice(9, self.width)


def rot6d(quat: torch.Tensor) -> torch.Tensor:
  return matrix_from_quat(quat)[..., :, :2].transpose(-1, -2).flatten(-2)


def quat_from_rot6d(features: torch.Tensor) -> torch.Tensor:
  first = F.normalize(features[..., :3], dim=-1)
  first = torch.where(
    first.square().sum(-1, keepdim=True) < 1e-8,
    torch.tensor([1.0, 0.0, 0.0], device=features.device, dtype=features.dtype),
    first,
  )
  second_raw = features[..., 3:6]
  second_raw = second_raw - (first * second_raw).sum(-1, keepdim=True) * first
  fallback = torch.where(
    first[..., :1].abs() < 0.9,
    torch.tensor([1.0, 0.0, 0.0], device=features.device, dtype=features.dtype),
    torch.tensor([0.0, 1.0, 0.0], device=features.device, dtype=features.dtype),
  )
  fallback = fallback - (first * fallback).sum(-1, keepdim=True) * first
  second = F.normalize(
    torch.where(second_raw.square().sum(-1, keepdim=True) < 1e-8, fallback, second_raw),
    dim=-1,
  )
  third = torch.cross(first, second, dim=-1)
  return quat_from_matrix(torch.stack((first, second, third), dim=-1))


def encode(states: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
  """Encode root pose and joint positions in the anchor yaw frame."""
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
      states[..., 13 : 13 + joints],
    ),
    dim=-1,
  )


def decode(
  features: torch.Tensor, anchor: torch.Tensor, layout: Layout
) -> torch.Tensor:
  """Decode features to world root pose and joint positions."""
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
      features[..., layout.joint_positions],
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


class Windows:
  """Draw contiguous position windows with history and post-target context."""

  def __init__(
    self,
    data: Dataset,
    history: int,
    future: int,
    min_steps: int,
    max_steps: int,
  ):
    if history < 2 or future < 2 or min_steps < 3 or max_steps < min_steps:
      raise ValueError("invalid boundary or duration bounds")
    self.data = data
    self.history = history
    self.future = future
    self.min_steps = min_steps
    self.max_steps = max_steps
    self.columns = history + max_steps + future - 1
    self.layout = Layout(data.num_joints)
    self.segments = data.segments(self.columns - 1, self.columns - 1)
    self.offsets = torch.arange(self.columns, device=data.states.device)

  def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return encoded windows and target ticks after the final history frame."""
    device = self.data.states.device
    picked = torch.randint(self.segments.starts.numel(), (count,), device=device)
    rows = self.segments.order[self.segments.starts[picked, None] + self.offsets]
    states = self.data.states[rows]
    duration = torch.randint(
      self.min_steps, self.max_steps + 1, (count,), device=device
    )
    return encode(states, states[:, self.history - 1]), duration


def bridge_mask(
  batch: int,
  columns: int,
  layout: Layout,
  history: int,
  future: int,
  duration: torch.Tensor,
) -> torch.Tensor:
  """Condition on real frames before A and from B onward."""
  if duration.shape != (batch,) or bool(
    ((duration < 3) | (duration > columns - history - future + 1)).any()
  ):
    raise ValueError("duration must fit the prediction horizon")
  mask = torch.zeros(
    batch, columns, layout.width, dtype=torch.bool, device=duration.device
  )
  mask[:, :history] = True
  rows = history - 1 + duration
  offsets = torch.arange(future, device=duration.device)
  batch_rows = torch.arange(batch, device=duration.device)[:, None]
  mask[batch_rows, rows[:, None] + offsets] = True
  return mask


def training_mask(
  layout: Layout,
  columns: int,
  history: int,
  future: int,
  duration: torch.Tensor,
  bridge_probability: float = 0.5,
) -> torch.Tensor:
  """Mix deployment masks with random temporal and partial-joint masks."""
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
