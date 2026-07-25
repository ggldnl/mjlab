"""Architecture 0: direct hand-off (the no-bridge baseline).

The thing bridging has to beat. The instant the controller commands a switch, the
target skill takes over from wherever the previous one left the robot, with nothing
in between. This is a naive cut-over from one skill to the next.

Dropping arch_0 in place of a trained architecture measures what that architecture is
worth, with everything else (controller, pool, env, switch timing) held fixed.
"""

from __future__ import annotations

import torch

from mjlab.envs import VecEnvObs
from mjlab.tasks.skills.meta import MetaPolicy
from mjlab.tasks.skills.skill import NO_SKILL


class Arch0(MetaPolicy):
  """Meta policy with no bridge: switches hand over to the target skill at once."""

  def bridge_step(
    self,
    obs: VecEnvObs,
    source: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del source  # Where control came from is exactly what a naive hand-off ignores.
    # Run the target skill on the active envs and declare hand-over immediately, so
    # from the next step control passes to the skill proper via the normal pool path.
    assignment = torch.where(active, target, torch.full_like(target, NO_SKILL))
    return self.pool.act(obs, assignment), active.clone()
