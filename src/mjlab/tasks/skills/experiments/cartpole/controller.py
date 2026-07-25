"""The scripted controller driving the cartpole demonstration.

It is assumed as given: run spin_up for a fixed number of steps, then switch once
to balance and stay there. A fixed-step switch is enough to exercise the bridges:
t is exactly what makes the naive hand-off fail, since the switch fires whether
the pole is actually up and slow when it does.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.controller import Controller
from mjlab.tasks.skills.skill import SkillPool

SPIN_UP = 0
BALANCE = 1


class CartpoleController(Controller):
  """Runs spin_up, then switches once to balance after swingup_steps, per env."""

  def __init__(self, pool: SkillPool, swingup_steps: int = 400) -> None:
    super().__init__(pool)
    self.swingup_steps = swingup_steps
    self._timer: torch.Tensor | None = None

  def decide(self, env: ManagerBasedRlEnv, target: torch.Tensor) -> torch.Tensor:
    del env
    if self._timer is None:
      self._timer = torch.full_like(target, self.swingup_steps)

    self._timer = self._timer - 1
    # spin_up while the timer is running, balance forever after it expires.
    return torch.where(
      self._timer > 0,
      torch.full_like(target, SPIN_UP),
      torch.full_like(target, BALANCE),
    )

  def reset(self, mask: torch.Tensor) -> None:
    if self._timer is None:
      self._timer = torch.full_like(mask, self.swingup_steps, dtype=torch.long)
    self._timer = torch.where(
      mask, torch.full_like(self._timer, self.swingup_steps), self._timer
    )
