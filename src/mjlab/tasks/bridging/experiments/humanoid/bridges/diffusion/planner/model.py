"""Temporal denoiser for masked dynamic-state windows."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelCfg:
  width: int = 256
  layers: int = 4
  heads: int = 8


class Denoiser(nn.Module):
  def __init__(self, features: int, columns: int, cfg: ModelCfg):
    super().__init__()
    if (
      features < 1
      or columns < 2
      or cfg.layers < 1
      or cfg.heads < 1
      or cfg.width < 2
      or cfg.width % 2
      or cfg.width % cfg.heads
    ):
      raise ValueError("invalid denoiser shape")
    self.features = features
    self.columns = columns
    self.input = nn.Linear(2 * features, cfg.width)
    self.position = nn.Parameter(torch.zeros(1, columns, cfg.width))
    nn.init.normal_(self.position, std=0.02)
    self.time = nn.Sequential(
      nn.Linear(cfg.width, cfg.width), nn.SiLU(), nn.Linear(cfg.width, cfg.width)
    )
    layer = nn.TransformerEncoderLayer(
      d_model=cfg.width,
      nhead=cfg.heads,
      dim_feedforward=4 * cfg.width,
      dropout=0.0,
      activation="gelu",
      batch_first=True,
      norm_first=True,
    )
    self.blocks = nn.TransformerEncoder(
      layer, num_layers=cfg.layers, enable_nested_tensor=False
    )
    self.output = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, features))

  def forward(
    self, noisy: torch.Tensor, known: torch.Tensor, step: torch.Tensor
  ) -> torch.Tensor:
    if noisy.ndim != 3 or noisy.shape[1:] != (self.columns, self.features):
      raise ValueError("noisy window has the wrong shape")
    if known.shape != noisy.shape or step.shape != noisy.shape[:1]:
      raise ValueError("known mask or diffusion step has the wrong shape")
    half = self.position.shape[-1] // 2
    rates = torch.exp(
      -math.log(10000.0) * torch.arange(half, device=noisy.device) / half
    )
    phase = step.float()[:, None] * rates[None]
    time = self.time(torch.cat((phase.sin(), phase.cos()), dim=-1))[:, None]
    embedded = (
      self.input(torch.cat((noisy, known.float()), dim=-1)) + self.position + time
    )
    return self.output(self.blocks(embedded))
