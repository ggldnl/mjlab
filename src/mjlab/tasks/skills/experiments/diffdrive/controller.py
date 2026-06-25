"""
A controller owns the skills and the bridge, decides which skill should be active,
and uses the bridge to transition there.

For diffdrive-in-corridors the decision comes for free, positionally: follow the active
corridor's skill and, as soon as the robot reaches the boundary to the next corridor,
switch to that corridor's skill  invoking the bridge to carry it across (a side entry
the skill itself can't make).
In other experiments, the Controller could be implemented using a finite state machine,
a neural network, a planning algorithm, ...
"""

from __future__ import annotations

from collections.abc import Mapping

from mjlab.tasks.skills.experiments.diffdrive.gridworld import GridWorld
from mjlab.tasks.skills.experiments.diffdrive.robot import X, Y
from mjlab.tasks.skills.interfaces import Bridge, Command, Controller, Skill, State


class CorridorController(Controller):
  """Positional controller for the corridor world.

  Follows the active corridor's skill; at each junction it begins a bridge transition to
  the next corridor's skill and hands control over once the bridge reports `done`. The
  switching decision is carried out by the _decide() method; the switching mechanism
  (run the bridge, then activate the target) is generic.
  """

  def __init__(
    self, world: GridWorld, skills: Mapping[int, Skill], bridge: Bridge
  ) -> None:

    # Skills and bridge
    self.skills = dict(skills)
    self.bridge = bridge

    # World related stuff
    self.world = world
    self._junctions = world.junction_map()

    # Current and next skill (set by reset)
    self.current: int = 1
    self.target: int | None = None
    self.reset()

  def reset(self) -> None:
    self.current = min(self.skills)  # first corridor in the sequence
    self.target = None  # corridor we are switching to, else None
    for skill in self.skills.values():  # re-arm latched skills (e.g. CorridorSkill)
      reset = getattr(skill, "reset", None)
      if callable(reset):
        reset()

  @property
  def switching(self) -> bool:
    return self.target is not None

  @property
  def mode(self) -> str:
    """Human-readable state, for status overlays: "2" or "2 -> 3"."""
    return f"{self.current} -> {self.target}" if self.switching else str(self.current)

  def _decide(self, state: State) -> int | None:
    """
    Fires at the junction corner of the active corridor.
    This is the only experiment-specific part of the controller.
    """
    target = self._junctions.get(self.current)
    if target is None:
      return None
    cell, nxt = target
    r, c = self.world.world_to_cell(float(state[X]), float(state[Y]))
    return nxt if (int(r), int(c)) == cell else None

  def step(self, state: State) -> Command:
    self.bridge.observe(
      state
    )  # feed the bridge every tick, so it can buffer the approach
    if not self.switching:
      nxt = self._decide(state)
      if nxt is not None:  # reached a junction -> start bridging to the next skill
        self.target = nxt
        rearm = getattr(self.skills[nxt], "reset", None)  # re-check its initiation set
        if callable(rearm):
          rearm()
        self.bridge.reset(self.skills[self.current], self.skills[nxt])
    if self.switching:
      command, done = self.bridge.step(state)
      if done:  # the bridge delivered us into the next corridor; hand control over
        assert self.target is not None
        self.current, self.target = self.target, None
      return command
    return self.skills[self.current](state)
