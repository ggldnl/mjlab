"""Training for arch_4: a bridge per skill, trained as in Masked-Token Prediction.

We use a large corpus of diverse motions. An example could be the LAFAN1 dataset.
We take a clip, we mask a window in the middle (even better if before and after
the window the robot does something different), we train a network to produce
the actions to fill that gap. The network will learn how to stitch two sequences.

At runtime, whenever a switch is signaled, we use the last N frames as first sequence
and the first M frames of a rollout of the next policy as second sequence. We ask the
bridge to fill the gap and produce a sequence of actions that makes the next skill
succeed.

Notice that the "where" could be explored: I talked about a rollout of the next policy,
but we can profile the policy and start from a different point rather than from the
beginning.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv, VecEnvObs
from mjlab.tasks.skills.meta import MetaPolicy
from mjlab.tasks.skills.skill import SkillPool
from mjlab.tasks.skills.view import StateView


class Arch4(MetaPolicy):
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
