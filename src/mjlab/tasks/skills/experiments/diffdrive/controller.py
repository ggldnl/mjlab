"""
The scripted controller driving the diffdrive demonstration. It is assumed given: a
fixed-step alternation between the two skills is enough to exercise the bridges.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.controller import Controller
from mjlab.tasks.skills.skill import NO_SKILL, SkillPool

DRIVE_STRAIGHT = 0
TURN = 1


class DiffdriveController(Controller):
  """Alternates drive <-> turn on fixed timers, per env."""

  def __init__(
    self, pool: SkillPool, straight_steps: int = 1000, turn_steps: int = 1000
  ) -> None:
    super().__init__(pool)
    self.straight_steps = straight_steps
    self.turn_steps = turn_steps
    self._timer: torch.Tensor | None = None
    self._pending_restart: torch.Tensor | None = None

  def _phase_length(self, skill_id: torch.Tensor) -> torch.Tensor:
    return torch.where(
      skill_id == TURN,
      torch.full_like(skill_id, self.turn_steps),
      torch.full_like(skill_id, self.straight_steps),
    )

  def decide(self, env: ManagerBasedRlEnv, target: torch.Tensor) -> torch.Tensor:
    del env
    if self._timer is None:
      self._timer = torch.full_like(target, self.straight_steps)
    if self._pending_restart is None:
      self._pending_restart = torch.zeros_like(target, dtype=torch.bool)

    current = torch.where(
      target == NO_SKILL, torch.full_like(target, DRIVE_STRAIGHT), target
    )

    self._timer = self._timer - 1
    switch_now = self._timer <= 0
    # Exactly two skills, so "switch" just means "toggle."
    next_skill = torch.where(switch_now, 1 - current, current)
    self._timer = torch.where(switch_now, self._phase_length(next_skill), self._timer)

    # Pending restarts (from `reset`) are applied last, overriding whatever the
    # timer logic above just computed, otherwise a restart landing on the same
    # call the old timer happens to expire would get immediately toggled away
    # again instead of actually restarting at DRIVE_STRAIGHT.
    next_skill = torch.where(
      self._pending_restart, torch.full_like(next_skill, DRIVE_STRAIGHT), next_skill
    )
    self._timer = torch.where(
      self._pending_restart,
      torch.full_like(self._timer, self.straight_steps),
      self._timer,
    )
    self._pending_restart = torch.zeros_like(self._pending_restart)
    return next_skill

  def reset(self, mask: torch.Tensor) -> None:
    if self._timer is None:
      self._timer = torch.full_like(mask, self.straight_steps, dtype=torch.long)
    if self._pending_restart is None:
      self._pending_restart = torch.zeros_like(mask)
    self._timer = torch.where(
      mask, torch.full_like(self._timer, self.straight_steps), self._timer
    )
    self._pending_restart = self._pending_restart | mask
