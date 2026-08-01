"""The scripted controller driving the cartpole demonstration.

It is assumed as given: run spin_up, then switch once to balance and stay there.
Only the moment of the switch changes between the two rules, and either one is
enough to exercise the bridges, since neither asks whether the pole is actually
up and slow when the hand-over fires.

- the timed rule switches after a fixed number of steps, so where the pole is at
  that instant is whatever the swing happens to be doing;
- the position rule waits for the same number of steps and then fires the first
  time the pole swings back through the bottom, which is the worst instant there
  is: the pole is as far from upright as it can be and moving at its fastest.
"""

from __future__ import annotations

import math

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.controller import Controller
from mjlab.tasks.skills.skill import SkillPool

SPIN_UP = 0
BALANCE = 1

# Scene entity holding the hinge the position rule reads.
ENTITY_NAME = "cartpole"

# Hinge joint, measured from upright: 0 is up, +-pi is hanging down.
HINGE_JOINT = "hinge_1"


class CartpoleController(Controller):
  """Runs spin_up, then switches once to balance after swingup_steps, per env."""

  def __init__(
    self,
    pool: SkillPool,
    swingup_steps: int = 400,
    position_based: bool = True,
    down_band: float = 0.35,
  ) -> None:
    """
    Initialize the controller.

    If `position_based` is true, the controller will transition from `swing_up` to `balance`
    based on the position (at one of the worst possible moments i.e. when the pole is pointing
    down), otherwise it will transition from `swing_up` to `balance` based on a timer.

    Either way `swingup_steps` is how long spin_up gets: for the timed rule it is the switch
    itself, for the position rule it is how long the swing is left alone before the next pass
    through the bottom is taken. `down_band` is how close to hanging counts as pointing down
    (radians away from +-pi), and is unused by the timed rule.
    """
    super().__init__(pool)
    self.position_based = position_based
    self.swingup_steps = swingup_steps
    self.down_band = down_band
    self._timer: torch.Tensor | None = None
    # The position rule latches: the switch fires once and balance keeps control even
    # as the pole it was handed carries on past the bottom.
    self._switched: torch.Tensor | None = None
    # Resolved on the first decide, once the scene exists.
    self._hinge_id: int | None = None

  def decide_timer(self, env: ManagerBasedRlEnv, target: torch.Tensor) -> torch.Tensor:
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

  def _hinge_angle(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """The hinge angle wrapped to [-pi, pi], shaped (num_envs,)."""
    entity = env.scene[ENTITY_NAME]
    if self._hinge_id is None:
      joint_ids, _ = entity.find_joints(HINGE_JOINT)
      self._hinge_id = joint_ids[0]
    theta = entity.data.joint_pos[:, self._hinge_id]
    # The hinge is free to keep turning, so the raw joint position accumulates over
    # revolutions; only the wrapped angle says where the pole is pointing.
    return torch.atan2(torch.sin(theta), torch.cos(theta))

  def decide_position(
    self, env: ManagerBasedRlEnv, target: torch.Tensor
  ) -> torch.Tensor:
    if self._timer is None:
      self._timer = torch.full_like(target, self.swingup_steps)
    if self._switched is None:
      self._switched = torch.zeros_like(target, dtype=torch.bool)

    self._timer = self._timer - 1

    # Waiting out the timer first matters: the episode starts with the pole hanging, so
    # without it the switch would fire on the very first step and spin_up would never run.
    theta = self._hinge_angle(env)
    pointing_down = theta.abs() > math.pi - self.down_band
    self._switched = self._switched | ((self._timer <= 0) & pointing_down)

    return torch.where(
      self._switched,
      torch.full_like(target, BALANCE),
      torch.full_like(target, SPIN_UP),
    )

  def decide(self, env: ManagerBasedRlEnv, target: torch.Tensor) -> torch.Tensor:
    if self.position_based:
      return self.decide_position(env, target)
    else:
      return self.decide_timer(env, target)

  def reset(self, mask: torch.Tensor) -> None:
    if self._timer is None:
      self._timer = torch.full_like(mask, self.swingup_steps, dtype=torch.long)
    if self._switched is None:
      self._switched = torch.zeros_like(mask)
    self._timer = torch.where(
      mask, torch.full_like(self._timer, self.swingup_steps), self._timer
    )
    self._switched = self._switched & ~mask
