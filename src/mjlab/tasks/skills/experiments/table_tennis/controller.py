"""Controllers for driving the table tennis primitives in a demo.

There is no composite task here (see this package's __init__.py): just the four
primitives (pick, catch, toss, hit) as independent skills in the pool. `CycleController`
is a simple debugging aid that exercises the switching machinery between them.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.controller import Controller
from mjlab.tasks.skills.skill import SkillPool


class CycleController(Controller):
  """Cycles the whole pool on a fixed timer, per env."""

  def __init__(self, pool: SkillPool, steps_per_skill: int = 150) -> None:
    super().__init__(pool)
    self.steps_per_skill = steps_per_skill
    self._timer: torch.Tensor | None = None

  def decide(self, env: ManagerBasedRlEnv, target: torch.Tensor) -> torch.Tensor:
    del env
    if self._timer is None:
      self._timer = torch.full_like(target, self.steps_per_skill)
    current = torch.where(target < 0, torch.zeros_like(target), target)
    self._timer = self._timer - 1
    switch_now = self._timer <= 0
    nxt = torch.where(switch_now, (current + 1) % len(self.pool), current)
    self._timer = torch.where(
      switch_now, torch.full_like(self._timer, self.steps_per_skill), self._timer
    )
    return nxt

  def reset(self, mask: torch.Tensor) -> None:
    if self._timer is None:
      self._timer = torch.full_like(mask, self.steps_per_skill, dtype=torch.long)
    self._timer = torch.where(
      mask, torch.full_like(self._timer, self.steps_per_skill), self._timer
    )
