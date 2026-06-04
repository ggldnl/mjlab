"""Observation terms for the waving (greeting) task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def wave_phase_value(env: ManagerBasedRlEnv, frequency: float) -> torch.Tensor:
  """Wave phase in radians, advancing at ``frequency`` Hz from episode start.

  Shared by the wave-tracking reward and the phase observation so both read the
  same clock. Shape: ``(num_envs,)``.
  """
  t = env.episode_length_buf.float() * env.step_dt
  return 2.0 * math.pi * frequency * t


def wave_phase(env: ManagerBasedRlEnv, frequency: float) -> torch.Tensor:
  """Sine/cosine encoding of the wave phase. Shape: ``(num_envs, 2)``.

  A sin/cos pair gives the policy a smooth, wrap-free view of where it is in the
  wave cycle so it can anticipate the next swing direction.
  """
  phase = wave_phase_value(env, frequency)
  return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)
