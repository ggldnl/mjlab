"""Observation terms for the jump task."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.abstraction.abstraction import Abstraction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def abstraction_obs(
  env: ManagerBasedRlEnv, abstraction_name: str, key: str
) -> torch.Tensor:
  """Expose an abstraction's reference to the policy.

  ``key`` selects which reference tensor to return (e.g. ``"target"`` for the
  landing target in the base frame, ``"takeoff_velocity"``, or ``"phase"``).
  """
  term = cast(Abstraction, env.abstraction_manager.get_term(abstraction_name))
  return term.get_obs(key)
