"""Runtime bridge state machine and arrival measurements."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from mjlab.tasks.bridging.experiments.humanoid.bridges.interface import (
  Bridge,
  BridgeOutput,
)
from mjlab.utils.lab_api.math import quat_error_magnitude

ROOT_STATE_DIM = 13

CHANNELS = (
  "root_pos",
  "root_ori",
  "root_lin_vel",
  "root_ang_vel",
  "joint_pos",
  "joint_vel",
)


@dataclass(frozen=True, kw_only=True)
class Tolerances:
  """Handoff limits in physical units."""

  root_pos: float = 0.05
  root_ori: float = 0.08
  root_lin_vel: float = 0.15
  root_ang_vel: float = 0.30
  joint_pos: float = 0.10
  joint_vel: float = 1.00

  def tensor(self, device: str | torch.device) -> torch.Tensor:
    values = tuple(getattr(self, name) for name in CHANNELS)
    if any(not math.isfinite(value) or value <= 0 for value in values):
      raise ValueError("Tolerances must be finite and positive")
    return torch.tensor(values, device=device, dtype=torch.float32)


def channel_errors(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  """Return the six handoff errors for two batches of dynamic states."""
  if actual.shape != target.shape or actual.ndim != 2:
    raise ValueError("actual and target must have the same (batch, state) shape")
  if actual.shape[1] < ROOT_STATE_DIM or (actual.shape[1] - ROOT_STATE_DIM) % 2:
    raise ValueError(
      "state must contain root state followed by joint position and velocity"
    )
  joints = (actual.shape[1] - ROOT_STATE_DIM) // 2
  q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + joints)
  qd = slice(ROOT_STATE_DIM + joints, ROOT_STATE_DIM + 2 * joints)
  return torch.stack(
    (
      torch.linalg.vector_norm(actual[:, 0:3] - target[:, 0:3], dim=-1),
      quat_error_magnitude(actual[:, 3:7], target[:, 3:7]),
      torch.linalg.vector_norm(actual[:, 7:10] - target[:, 7:10], dim=-1),
      torch.linalg.vector_norm(
        actual[:, 10:ROOT_STATE_DIM] - target[:, 10:ROOT_STATE_DIM], dim=-1
      ),
      (actual[:, q] - target[:, q]).abs().amax(dim=-1),
      (actual[:, qd] - target[:, qd]).abs().amax(dim=-1),
    ),
    dim=-1,
  )


def arrival_score(errors: torch.Tensor, tolerances: torch.Tensor) -> torch.Tensor:
  """Smooth score dominated by the worst normalized channel."""
  reach = torch.log1p(errors / tolerances)
  distance = 0.7 * reach.amax(dim=-1) + 0.3 * reach.mean(dim=-1)
  return 1.0 / (1.0 + distance)


class DockingBridge(Bridge):
  """Switch from an approach policy to docking, then latch and blend the handoff.

  Both policies implement the Bridge input convention and return joint actions. The target
  sequence must have odd length; its middle state is the handoff state.
  """

  def __init__(
    self,
    approach_policy: nn.Module,
    docking_policy: nn.Module,
    action_dim: int,
    *,
    tolerances: Tolerances | None = None,
    capture_scale: float = 3.0,
    blend_steps: int = 5,
  ) -> None:
    super().__init__(action_dim)
    if capture_scale <= 1.0:
      raise ValueError("capture_scale must be greater than one")
    if blend_steps < 1:
      raise ValueError("blend_steps must be positive")
    self.approach_policy = approach_policy
    self.docking_policy = docking_policy
    self.capture_scale = capture_scale
    self.blend_steps = blend_steps
    self.register_buffer("tolerances", (tolerances or Tolerances()).tensor("cpu"))
    self.register_buffer("_docking", torch.empty(0, dtype=torch.bool), persistent=False)
    self.register_buffer(
      "_captured", torch.empty(0, dtype=torch.bool), persistent=False
    )
    self.register_buffer(
      "_blend_age", torch.empty(0, dtype=torch.long), persistent=False
    )

  def _ensure_state(self, batch: int, device: torch.device) -> None:
    if self._docking.shape == (batch,) and self._docking.device == device:
      return
    self._docking = torch.zeros(batch, dtype=torch.bool, device=device)
    self._captured = torch.zeros(batch, dtype=torch.bool, device=device)
    self._blend_age = torch.zeros(batch, dtype=torch.long, device=device)

  def step(
    self,
    history: torch.Tensor,
    target: torch.Tensor,
    time_left: torch.Tensor,
  ) -> BridgeOutput:
    if target.shape[1] % 2 != 1:
      raise ValueError("target trajectory must have odd length")
    batch = history.shape[0]
    self._ensure_state(batch, history.device)
    goal = target[:, target.shape[1] // 2]
    errors = channel_errors(history[:, -1], goal)
    limits = cast(torch.Tensor, self.tolerances).to(history.device)

    self._docking |= (errors <= limits * self.capture_scale).all(dim=-1)
    newly_captured = (errors <= limits).all(dim=-1) & ~self._captured
    self._captured |= newly_captured
    self._blend_age = torch.where(self._captured, self._blend_age + 1, self._blend_age)

    approach = self.approach_policy(history, target, time_left)
    docking = self.docking_policy(history, target, time_left)
    action = torch.where(self._docking.unsqueeze(-1), docking, approach)
    blend = (self._blend_age.float() / self.blend_steps).clamp(max=1.0)
    expired = time_left <= 0.0
    handoff = expired | (self._captured & (blend >= 1.0))
    blend = torch.where(expired, torch.ones_like(blend), blend)
    return BridgeOutput(action, handoff, self._captured.clone(), blend)

  def reset(self, done: torch.Tensor | None = None) -> None:
    if self._docking.numel() == 0:
      return
    if done is None:
      done = torch.ones_like(self._docking)
    if done.shape != self._docking.shape:
      raise ValueError("done must match the current batch")
    self._docking[done] = False
    self._captured[done] = False
    self._blend_age[done] = 0
