"""
The do-nothing baseline bridge.

InstantBridge performs no transition and defers to the next skill immediately. It
is the reference for "no bridging": since the robot arrives at a junction carrying
cross-axis momentum that the next skill cannot steer out, this bridge won't work,
and that failure is the whole motivation for a real bridge.
"""

from __future__ import annotations

from mjlab.tasks.skills.interfaces import Bridge, Command, Skill, State


class InstantBridge(Bridge):
  """Basic baseline: no transition, hand straight over to the next skill.

  `step` reports `done` on the first tick, so the controller activates the next
  skill from the current state.
  """

  def __init__(self):
    self._to_skill = lambda x: x  # dummy skill

  def reset(self, from_skill: Skill, to_skill: Skill) -> None:
    self._to_skill = to_skill

  def step(self, state: State) -> tuple[Command, bool]:
    return self._to_skill(state), True
