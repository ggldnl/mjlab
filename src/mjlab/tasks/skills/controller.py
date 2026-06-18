"""Controllers: orchestrate which skill is active and run a bridge across switches.

A controller owns the named skills and one bridge. It does NOT decide *when* or *to
which* skill to switch -- that is external (an experiment's scenario / supervisor,
later a planner or a learned high-level policy). It only carries out a requested
switch, using the bridge to get there.

``Controller`` is the shared interface; ``FSMController`` is the basic finite-state
machine every experiment can reuse. Custom controllers implement ``Controller``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from mjlab.tasks.skills.bridge import Bridge, Command, Skill, State


class Controller(Protocol):
  """Interface a controller exposes to an experiment."""

  def reset(self, start: str) -> None:
    """Reset to ``start`` as the active skill, no switch in progress."""
    ...

  def switch_to(self, name: str) -> None:
    """Request a switch to skill ``name`` (carried out via the bridge)."""
    ...

  def step(self, state: State) -> Command:
    """Return the command to apply this tick."""
    ...


class FSMController:
  """Basic skill-switching controller (a two-state finite-state machine).

  States:
    * RUNNING   -- emit the active skill's command.
    * SWITCHING -- a switch was requested; emit the bridge's command until the
                   bridge reports ``done``, then make the target the active skill.
  """

  def __init__(self, skills: Mapping[str, Skill], bridge: Bridge, start: str) -> None:
    self._skills = dict(skills)
    self._bridge = bridge
    self.reset(start)

  def reset(self, start: str) -> None:
    self.active = start  # name of the running skill
    self._target: str | None = None  # skill we are switching to, else None

  @property
  def switching(self) -> bool:
    return self._target is not None

  @property
  def mode(self) -> str:
    """Human-readable state, for logging / overlays."""
    return f"{self.active} -> {self._target}" if self.switching else self.active

  def switch_to(self, name: str) -> None:
    if self.switching or name == self.active:
      return  # ignore redundant / repeated requests
    self._target = name
    self._bridge.reset(self._skills[self.active], self._skills[name])

  def step(self, state: State) -> Command:
    if not self.switching:
      return self._skills[self.active](state)  # RUNNING
    command, done = self._bridge.step(state)  # SWITCHING
    if done:
      # The bridge delivered us into the next skill; hand control over.
      assert self._target is not None
      self.active = self._target
      self._target = None
    return command
