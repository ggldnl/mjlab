"""Architecture 0: the no-bridge baseline. A switch is a direct hand-off.

The target skill takes the controls on the step the switch fires, with nothing in
between. This is the failure the others exist to fix, so it is what they are measured
against. It trains nothing, saves nothing, and has no config.
"""

from __future__ import annotations

import torch

from mjlab.envs import VecEnvObs
from mjlab.tasks.skills.meta import MetaPolicy
from mjlab.tasks.skills.skill import NO_SKILL, SkillPool


class Arch0(MetaPolicy):
  """Hands control over immediately, with no transition of any kind."""

  def begin_switch(
    self, switching: torch.Tensor, source: torch.Tensor, target: torch.Tensor
  ) -> None:
    del source, target
    # The target skill produces this step's action, so it takes control now. The base
    # class engages a skill the step *before* its first action, which never happens here.
    self.engage(switching)

  def involved(self, assignment: torch.Tensor) -> torch.Tensor:
    # The target is driving the switching envs, so it has to be marked in the pool's
    # single tick with those envs set (see skill.py).
    handing_over = self._bridging & (self._target >= 0)
    return super().involved(torch.where(handing_over, self._target, assignment))

  def bridge_step(
    self,
    obs: VecEnvObs,
    skill_actions: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del obs, source  # Where control came from is what a naive hand-off ignores
    # Take the target's row of this step's tick and hand over at once, so from the next
    # step control passes to the skill through the normal path.
    assignment = torch.where(active, target, torch.full_like(target, NO_SKILL))
    return SkillPool.select(skill_actions, assignment), active.clone()
