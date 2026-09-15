"""Common policy interface for bridge architectures."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class BridgeOutput(NamedTuple):
  """One bridge control step."""

  action: torch.Tensor
  """Joint command with shape (batch, action)."""
  handoff: torch.Tensor
  """Boolean mask saying control should transfer to the entering policy."""
  captured: torch.Tensor
  """Boolean mask saying the target tolerances were met."""
  blend: torch.Tensor
  """Entering-policy weight before handoff, from zero to one."""


class Bridge(nn.Module):
  """Drive a batch of robots toward target state sequences.

  Inputs use shapes ``(batch, time, state)``. ``time_left`` uses seconds and has
  shape ``(batch,)``. Current and target states must use the same coordinate frame.

  The base bridge transfers control immediately without applying an action. Concrete
  architectures override ``step`` and, when they keep per-environment state, ``reset``.
  """

  def __init__(self, action_dim: int) -> None:
    super().__init__()
    if action_dim < 1:
      raise ValueError("action_dim must be positive.")
    self.action_dim = action_dim

  def forward(
    self,
    history: torch.Tensor,
    target: torch.Tensor,
    time_left: torch.Tensor,
  ) -> BridgeOutput:
    """Return one action and the handoff state per batch item."""
    if history.ndim != 3 or target.ndim != 3:
      raise ValueError("history and target must have shape (batch, time, state).")
    if history.shape[0] != target.shape[0] or history.shape[2] != target.shape[2]:
      raise ValueError("history and target batch and state dimensions must match.")
    if time_left.shape != (history.shape[0],):
      raise ValueError("time_left must have shape (batch,).")
    output = self.step(history, target, time_left)
    batch = history.shape[0]
    if output.action.shape != (batch, self.action_dim):
      raise ValueError("action must have shape (batch, action_dim).")
    if any(value.shape != (batch,) for value in output[1:]):
      raise ValueError("handoff, captured and blend must have shape (batch,).")
    return output

  def step(
    self,
    history: torch.Tensor,
    target: torch.Tensor,
    time_left: torch.Tensor,
  ) -> BridgeOutput:
    """Transfer immediately. Concrete bridges override this method."""
    del target, time_left
    batch = history.shape[0]
    return BridgeOutput(
      action=history.new_zeros((batch, self.action_dim)),
      handoff=torch.ones(batch, dtype=torch.bool, device=history.device),
      captured=torch.zeros(batch, dtype=torch.bool, device=history.device),
      blend=history.new_ones(batch),
    )

  def reset(self, done: torch.Tensor | None = None) -> None:
    """Clear state for finished batch items. The base bridge keeps no state."""
    del done
