"""Reward terms for the crawler velocity task (abstraction variant)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.abstraction.abstraction import Abstraction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def abstraction_signal(
  env: ManagerBasedRlEnv, abstraction_name: str, signal_name: str
) -> torch.Tensor:
  """Surface a named abstraction signal as a reward term, shape ``(num_envs,)``."""
  term = cast(Abstraction, env.abstraction_manager.get_term(abstraction_name))
  return term.get_signal(signal_name)
