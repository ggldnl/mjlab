"""Extract discrete touchdown routes from measured contacts."""

from __future__ import annotations

import torch


def touchdown_route(
  contact: torch.Tensor, duration: torch.Tensor, slots: int = 6
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return touchdown count and ordered feet, with zero for unused slots.

  Contact has shape (batch, time, 2). A touchdown needs three consecutive
  contact frames to avoid treating a one-frame sensor bounce as a step.
  """
  if contact.ndim != 3 or contact.shape[-1] != 2 or contact.shape[1] < 3:
    raise ValueError("contact must have shape (batch, at least 3, 2)")
  if duration.shape != contact.shape[:1] or slots < 1:
    raise ValueError("duration or slots has the wrong shape")
  stable = contact.bool().unfold(1, 3, 1).all(dim=-1)
  stable = torch.nn.functional.pad(stable.transpose(1, 2), (2, 0)).transpose(1, 2)
  stable[:, :2] = contact[:, :1].bool()
  rising = stable[:, 1:] & ~stable[:, :-1]
  times = torch.arange(1, contact.shape[1], device=contact.device)
  rising &= times[None, :, None] <= duration[:, None, None]
  count = rising.sum(dim=(1, 2)).clamp(max=slots)
  order = torch.where(
    rising,
    2 * times[None, :, None] + torch.arange(2, device=contact.device)[None, None],
    2 * contact.shape[1],
  ).flatten(1)
  first = order.sort(dim=1).values[:, :slots]
  feet = torch.where(first < 2 * contact.shape[1], first.remainder(2) + 1, 0)
  return count.long(), feet.long()
