"""Bridges: what to command *between* skills, plus the interface they share.

Shared vocabulary used across the whole skeleton (kept here because this is the
lowest-level shared module):

    State   = np.ndarray   the robot's reduced state (layout is system-specific).
    Command = np.ndarray   the action a skill emits and that is applied to the robot
                           (system-specific; wheel torques for diffdrive).
    Skill   = State -> Command.

A bridge is engaged by the controller when a switch is requested. It is ``reset``
with the skill being left and the skill being entered, then ``step``-ed each tick;
it returns the command to apply now and whether the next skill may take over.

``InstantBridge`` is the basic baseline: it performs no transition at all -- it
defers to the next skill immediately. Real, system-specific bridges live in the
per-experiment folders and implement this same ``Bridge`` interface.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

State = np.ndarray
Command = np.ndarray
Skill = Callable[[State], Command]


class Bridge(Protocol):
  """Interface every bridge implements so the controller can drive it."""

  def reset(self, from_skill: Skill, to_skill: Skill) -> None:
    """Begin a transition from ``from_skill`` to ``to_skill``."""
    ...

  def step(self, state: State) -> tuple[Command, bool]:
    """Return ``(command, done)``; ``done`` means the next skill may take over."""
    ...


class InstantBridge:
  """Basic baseline: no transition -- hand straight over to the next skill.

  ``step`` reports ``done`` on the first tick, so the controller activates the next
  skill from the current state. Simple, and the reference for "no bridging".
  """

  def reset(self, from_skill: Skill, to_skill: Skill) -> None:
    self._to_skill = to_skill

  def step(self, state: State) -> tuple[Command, bool]:
    return self._to_skill(state), True
