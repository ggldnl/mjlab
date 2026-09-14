"""The diffusion process: how a window is corrupted during training and rebuilt at inference.

Training corrupts a recorded window to a random noise level and asks the denoiser for the
window it came from. Nothing about a crossing is involved. What comes out is a model of
what the robot does over a second, and that is all it is.

Sampling starts from noise and denoises down, and two things are imposed from outside on
the way:

    pinning    the history columns are overwritten with the ticks that actually happened,
               every step, so the window the model is completing is one that starts where
               the robot is standing
    guidance   a cost pulls the sample toward the target. See guidance.py

That split is the whole reason the model generalizes past its training set. The corpus
never contained the crossing being asked for, and it does not have to: the model supplies
what a second of G1 motion looks like and the constraint supplies where it has to end up.

DDIM rather than the full reverse chain, because a plan is drawn several times a second
inside a control loop. A hundred step ancestral chain at 50 Hz is not a controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.guidance import Cost


@dataclass
class ProcessCfg:
  """The noise schedule, and how many steps to take back down it."""

  steps: int = 100
  """Noise levels used in training. More is a finer ladder and a slower sampler."""

  sample_steps: int = 12
  """Levels actually visited when sampling, spread evenly over the ladder. Twelve is the
  usual trade: the arrival barely improves past it and the control loop pays for every
  one."""

  clip: float = 4.0
  """Clamp on the predicted clean window, in normalized units. Four deviations is well
  outside the corpus, so this only catches a prediction that has diverged. Without it a
  single bad step at high noise propagates all the way down."""


def cosine_alphas(steps: int, offset: float = 0.008) -> torch.Tensor:
  """Nichol and Dhariwal's cosine schedule. (steps,) cumulative alpha.

  Linear betas spend most of their ladder at noise levels where the window is already
  destroyed, which on trajectory data means most of the training budget goes on rungs that
  teach nothing.
  """
  t = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
  f = torch.cos((t + offset) / (1.0 + offset) * math.pi * 0.5) ** 2
  alphas = (f[1:] / f[0]).clamp(min=1e-5, max=0.9999)
  return alphas.float()


class Process(nn.Module):
  """Noise schedule plus the two things done with it: a training loss and a sampler."""

  def __init__(self, denoiser: nn.Module, cfg: ProcessCfg) -> None:
    super().__init__()
    self.denoiser = denoiser
    self.cfg = cfg
    self.register_buffer("alphas", cosine_alphas(cfg.steps))

  @property
  def _alphas(self) -> torch.Tensor:
    buffer = self.alphas
    assert isinstance(buffer, torch.Tensor)
    return buffer

  def corrupt(
    self, window: torch.Tensor, step: torch.Tensor, noise: torch.Tensor
  ) -> torch.Tensor:
    """One window at one noise level. (N, T, F)."""
    alpha = self._alphas[step].view(-1, 1, 1)
    return alpha.sqrt() * window + (1.0 - alpha).sqrt() * noise

  def loss(self, window: torch.Tensor, history: int) -> torch.Tensor:
    """Train on one batch of recorded windows.

    The history columns are pinned to their true values before the denoiser sees them, the
    same way sampling pins them, and then carry no loss. Training on a conditioning pattern
    that inference does not use is the standard way a diffusion planner ends up worse at the
    one thing it is for, so the two are written to match here.
    """
    count = window.shape[0]
    step = torch.randint(0, self.cfg.steps, (count,), device=window.device)
    noise = torch.randn_like(window)
    noised = self.corrupt(window, step, noise)
    noised[:, :history] = window[:, :history]

    predicted = self.denoiser(noised, step)
    return torch.nn.functional.mse_loss(predicted[:, history:], window[:, history:])

  @torch.no_grad()
  def sample(
    self,
    shape: tuple[int, int, int],
    history: torch.Tensor,
    cost: Cost | None,
    device: torch.device | str,
  ) -> torch.Tensor:
    """Draw one window per environment, conditioned on history and pulled by cost.

    Args:
      shape: (N, T, F).
      history: (N, H, F) the ticks that actually happened, pinned into every step.
      cost: what the crossing is asked for, or None for an unguided sample.

    Guidance is applied to the predicted clean window, not to the noisy one, and so needs
    no gradient through the denoiser. That is what keeps this callable from inside the
    inference mode the evaluation harness wraps a policy in, and it is also what makes a
    plan cheap enough to redraw inside a control loop.
    """
    span = history.shape[1]
    ladder = torch.linspace(
      self.cfg.steps - 1, 0, self.cfg.sample_steps, device=device
    ).long()

    x = torch.randn(shape, device=device)
    x[:, :span] = history
    for index, step in enumerate(ladder):
      level = step.expand(shape[0])
      predicted = self.denoiser(x, level).clamp(-self.cfg.clip, self.cfg.clip)
      predicted[:, :span] = history
      if cost is not None:
        predicted = cost.apply(predicted)
        predicted[:, :span] = history

      alpha = self._alphas[step]
      following = self._alphas[ladder[index + 1]] if index + 1 < len(ladder) else None
      if following is None:
        x = predicted
        break
      # Deterministic DDIM: recover the noise this sample implies, then re-noise the
      # corrected clean window to the next level down with that same noise
      noise = (x - alpha.sqrt() * predicted) / (1.0 - alpha).clamp(min=1e-8).sqrt()
      x = following.sqrt() * predicted + (1.0 - following).sqrt() * noise
      x[:, :span] = history

    x[:, :span] = history
    return x
