"""The denoiser: a temporal convolutional U-Net over one window.

Takes a noised window and the diffusion step it was noised to, returns the clean window it
thinks that came from. Convolution runs along time, not along features, so the receptive
field is a stretch of ticks and a filter is a short motion primitive rather than a fixed
coordinate. That is what makes the same weights usable at every column of the window and
what lets a horizon be re-cut without retraining the architecture.

Predicting the clean window rather than the noise it carries is what makes guidance cheap.
Everything guidance wants to say is a statement about a trajectory in physical units, and
with this parameterization the model hands one over at every denoising step, ready to be
corrected, instead of a noise field that has to be converted into one first.

The diffusion step is the only conditioning. What a crossing is asked for never enters the
network: the start is pinned into the window and the target is applied as a cost while
sampling, both from outside. That is the BeyondMimic split, and it is why one trained model
answers questions it was never trained on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelCfg:
  """Shape of the denoiser."""

  dim: int = 128
  """Channels at the finest resolution. The network widens by dim_mults from here."""

  dim_mults: tuple[int, ...] = (1, 2, 4)
  """One entry per resolution. Every entry past the first halves the time axis, so a
  64 tick window becomes 64, 32, 16 and the coarsest filters see the whole crossing."""

  kernel: int = 5
  """Ticks one convolution sees. Five at 50 Hz is 100 ms, about a third of a stride."""

  groups: int = 8
  """Group norm groups. Batch norm is wrong here: a diffusion batch mixes noise levels."""


class Timestep(nn.Module):
  """Sinusoidal embedding of the diffusion step, widened by an MLP."""

  def __init__(self, dim: int) -> None:
    super().__init__()
    self.dim = dim
    self.mlp = nn.Sequential(
      nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim)
    )

  def forward(self, step: torch.Tensor) -> torch.Tensor:
    half = self.dim // 2
    freqs = torch.exp(
      -math.log(10000.0)
      * torch.arange(half, device=step.device, dtype=torch.float32)
      / half
    )
    angles = step.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return self.mlp(torch.cat([angles.sin(), angles.cos()], dim=-1))


class Conv(nn.Module):
  """Conv along time, group norm, Mish."""

  def __init__(self, inp: int, out: int, kernel: int, groups: int) -> None:
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv1d(inp, out, kernel, padding=kernel // 2),
      nn.GroupNorm(min(groups, out), out),
      nn.Mish(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.block(x)


class Residual(nn.Module):
  """Two convolutions with the diffusion step folded in between them.

  The step arrives as a per channel bias rather than an extra input channel, so it reaches
  every position of the window at once instead of having to be carried there by the
  convolution.
  """

  def __init__(self, inp: int, out: int, embed: int, kernel: int, groups: int) -> None:
    super().__init__()
    self.first = Conv(inp, out, kernel, groups)
    self.second = Conv(out, out, kernel, groups)
    self.condition = nn.Sequential(nn.Mish(), nn.Linear(embed, out))
    self.skip = nn.Conv1d(inp, out, 1) if inp != out else nn.Identity()

  def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
    out = self.first(x) + self.condition(embed).unsqueeze(-1)
    return self.second(out) + self.skip(x)


class Denoiser(nn.Module):
  """Window in, window out. (N, T, F) -> (N, T, F).

  Time is the convolution axis, so the tensor is transposed on the way in and back on the
  way out. Every resolution but the last halves T, so a window whose length is not divisible
  by two per downsample would lose ticks in the round trip: the constructor refuses that
  rather than silently returning a shorter window.
  """

  def __init__(self, width: int, columns: int, cfg: ModelCfg) -> None:
    super().__init__()
    levels = len(cfg.dim_mults)
    stride = 2 ** (levels - 1)
    if columns % stride:
      raise ValueError(
        f"A {columns} tick window does not survive {levels - 1} halvings. Pick a "
        f"history plus horizon that is a multiple of {stride}."
      )
    self.cfg = cfg
    self.width = width
    self.columns = columns

    dims = [cfg.dim * m for m in cfg.dim_mults]
    self.time = Timestep(cfg.dim)
    self.inject = nn.Conv1d(width, dims[0], 1)

    self.downs = nn.ModuleList()
    self.reduce = nn.ModuleList()
    for index in range(levels):
      inp = dims[max(index - 1, 0)]
      self.downs.append(Residual(inp, dims[index], cfg.dim, cfg.kernel, cfg.groups))
      last = index == levels - 1
      self.reduce.append(
        nn.Identity()
        if last
        else nn.Conv1d(dims[index], dims[index], 3, stride=2, padding=1)
      )

    self.middle = Residual(dims[-1], dims[-1], cfg.dim, cfg.kernel, cfg.groups)

    self.ups = nn.ModuleList()
    self.expand = nn.ModuleList()
    for index in reversed(range(levels)):
      first = index == levels - 1
      self.expand.append(
        nn.Identity()
        if first
        else nn.ConvTranspose1d(dims[index + 1], dims[index], 4, stride=2, padding=1)
      )
      # Twice the channels in: the skip from the matching level is concatenated
      self.ups.append(
        Residual(dims[index] * 2, dims[index], cfg.dim, cfg.kernel, cfg.groups)
      )

    self.out = nn.Sequential(
      Conv(dims[0], dims[0], cfg.kernel, cfg.groups), nn.Conv1d(dims[0], width, 1)
    )

  def forward(self, window: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    embed = self.time(step)
    x = self.inject(window.transpose(1, 2))

    skips: list[torch.Tensor] = []
    for block, reduce in zip(self.downs, self.reduce, strict=True):
      x = block(x, embed)
      skips.append(x)
      x = reduce(x)

    x = self.middle(x, embed)

    for block, expand in zip(self.ups, self.expand, strict=True):
      x = expand(x)
      x = block(torch.cat([x, skips.pop()], dim=1), embed)

    return self.out(x).transpose(1, 2)
