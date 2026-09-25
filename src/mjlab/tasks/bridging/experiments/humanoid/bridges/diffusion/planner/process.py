"""Masked DDPM training and DDIM trajectory inpainting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.planner.model import (
  Denoiser,
)


@dataclass(frozen=True)
class ProcessCfg:
  steps: int = 100
  sample_steps: int = 50
  continuity_weight: float = 0.5


def cosine_schedule(steps: int) -> torch.Tensor:
  if steps < 2:
    raise ValueError("steps must be at least two")
  ticks = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
  curve = torch.cos((ticks + 0.008) / 1.008 * math.pi / 2).square()
  return (curve[1:] / curve[0]).clamp(1e-5, 0.9999).float()


class Diffusion(nn.Module):
  def __init__(self, denoiser: Denoiser, cfg: ProcessCfg):
    super().__init__()
    if not 1 <= cfg.sample_steps <= cfg.steps:
      raise ValueError("sample_steps must be between one and steps")
    if cfg.continuity_weight < 0:
      raise ValueError("continuity_weight cannot be negative")
    self.denoiser = denoiser
    self.cfg = cfg
    self.register_buffer("alphas", cosine_schedule(cfg.steps))

  @property
  def schedule(self) -> torch.Tensor:
    assert isinstance(self.alphas, torch.Tensor)
    return self.alphas

  def loss(
    self,
    clean: torch.Tensor,
    known: torch.Tensor,
    valid: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if clean.shape != known.shape:
      raise ValueError("clean window and known mask must match")
    if valid is None:
      valid = torch.ones_like(known)
    elif valid.shape == clean.shape[:2]:
      valid = valid[..., None].expand_as(clean)
    elif valid.shape != clean.shape:
      raise ValueError("valid mask must cover time or every feature")
    step = torch.randint(self.cfg.steps, (clean.shape[0],), device=clean.device)
    alpha = self.schedule[step, None, None]
    noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * torch.randn_like(clean)
    noisy = torch.where(known, clean, noisy)
    predicted = self.denoiser(noisy, known, step)
    unknown = ~known & valid
    reconstruction = (predicted - clean).square()[unknown].mean()
    full = torch.where(known, clean, predicted)
    target_delta = clean[:, 1:] - clean[:, :-1]
    predicted_delta = full[:, 1:] - full[:, :-1]
    active_edges = (unknown[:, 1:] | unknown[:, :-1]) & (valid[:, 1:] & valid[:, :-1])
    continuity = (predicted_delta - target_delta).square()[active_edges].mean()
    return reconstruction + self.cfg.continuity_weight * continuity

  @torch.no_grad()
  def sample(
    self,
    known_values: torch.Tensor,
    known: torch.Tensor,
    trace: list[torch.Tensor] | None = None,
  ) -> torch.Tensor:
    """Fill free channels; copy all conditioned channels at every denoising step.

    A trace list collects one clean prediction per rung of the ladder, for viewers.
    Its last entry is what is returned.
    """
    if known_values.shape != known.shape:
      raise ValueError("known values and mask must match")
    ladder = (
      torch.linspace(
        self.cfg.steps - 1, 0, self.cfg.sample_steps, device=known_values.device
      )
      .round()
      .long()
      .unique_consecutive()
    )
    noisy = torch.where(known, known_values, torch.randn_like(known_values))
    for index, tick in enumerate(ladder):
      step = tick.expand(noisy.shape[0])
      clean = self.denoiser(noisy, known, step)
      clean = torch.where(known, known_values, clean)
      if trace is not None:
        trace.append(clean)
      if index == len(ladder) - 1:
        return clean
      alpha = self.schedule[tick]
      next_alpha = self.schedule[ladder[index + 1]]
      noise = (noisy - alpha.sqrt() * clean) / (1 - alpha).sqrt()
      noisy = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * noise
      noisy = torch.where(known, known_values, noisy)
    raise RuntimeError("empty denoising schedule")
