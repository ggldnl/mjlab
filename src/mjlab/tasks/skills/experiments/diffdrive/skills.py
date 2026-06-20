"""Plain per-corridor skills.

A skill is just a state -> command function. The robot layer turns the command
-- a target body twist (v*, omega*) -- into wheel torques, so a skill only has
to say what motion it wants, never how to actuate it.

cruise(speed) is the single base skill: drive forward at speed with zero turn rate.
It holds its heading and never steers, so it stays on a corridor only when handed
a state already on the centerline and aligned. Crucially it does not turn:
turning/recovering onto a corridor is the bridge's job, not a skill's (a skill
that steered toward the axis would be doing part of the bridging).

One corridor = one cruise skill, differing only in the desired speed:

    from mjlab.tasks.skills.experiments.diffdrive.experiment import corridor_speeds
    skills = corridor_skills(corridor_speeds(world))   # {cid: state -> command}
    controller = CorridorController(world, skills, bridge)
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from mjlab.tasks.skills.interfaces import Command, Skill, State


def cruise(speed: float) -> Skill:
  """The narrow corridor skill: drive straight at speed, holding heading.

  Returns a state -> command policy whose command is the constant target twist
  (v*, omega*) = (speed, 0). With omega* = 0 it never steers (no recovery); it
  does not even read the state.
  """

  def skill(state: State) -> Command:
    return np.array([speed, 0.0])

  return skill


def corridor_skills(speeds: Mapping[int, float]) -> dict[int, Skill]:
  """One cruise skill per corridor, each at that corridor's target speed."""
  return {cid: cruise(speed) for cid, speed in speeds.items()}
