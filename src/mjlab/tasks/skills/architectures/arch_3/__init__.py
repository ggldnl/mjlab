"""Architecture 3: residuals."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv, VecEnvObs
from mjlab.tasks.skills.meta import MetaPolicy
from mjlab.tasks.skills.skill import SkillPool
from mjlab.tasks.skills.view import StateView


class Arch3(MetaPolicy):
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    pool: SkillPool,
    view: StateView | None = None,
  ) -> None:

    self.bridge = None

    super().__init__(env, pool, view)

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
