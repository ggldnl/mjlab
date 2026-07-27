"""Architecture 4: just one bridge.

One bridge to rule them all.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv, VecEnvObs
from mjlab.tasks.skills.meta import MetaPolicy
from mjlab.tasks.skills.skill import SkillPool


class Arch4(MetaPolicy):
  """Meta policy holding the one and only bridge."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    pool: SkillPool,
  ) -> None:

    self.bridge = None

    super().__init__(env, pool)

  @torch.no_grad()
  def bridge_step(
    self,
    obs: VecEnvObs,
    source: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:

    del source  # Source-agnostic: one bridge per target, whatever it came from

    # Homogeneous-batch assumption: the mid-bridge envs all head to the same target
    # (the composition switches every env on the same schedule), so one target's
    # actor/switch serves the batch.
    # TODO Revisit if envs can bridge to different targets at once

    raise ValueError("Not implemented")
