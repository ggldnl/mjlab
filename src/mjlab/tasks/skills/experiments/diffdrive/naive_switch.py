"""Naive baseline experiment: drive straight, then instantly switch to turn_left.

Wires the diff-drive skills with the shared ``FSMController`` + ``InstantBridge``
(i.e. no real bridging) and scripts one switch: cruise forward, and once past a
fixed distance request ``turn_left``. The instant bridge engages the turn while the
robot is still moving, so it cannot spin in place. This is the reference point a
real bridge / controller is compared against -- not the final experiment.

Run: ``uv run python -m mjlab.tasks.skills.experiments.diffdrive.naive_switch``
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from mjlab.tasks.skills import play
from mjlab.tasks.skills.bridge import InstantBridge
from mjlab.tasks.skills.config.diffdrive import skills
from mjlab.tasks.skills.config.diffdrive.dynamics import state_from_mjdata
from mjlab.tasks.skills.controller import FSMController

_XML = Path(__file__).parents[2] / "config" / "diffdrive" / "diffdrive.xml"
_SWITCH_AFTER_X = 2.0  # forward travel (m) before requesting the turn


def build_policy() -> tuple[play.Policy, play.ResetHook]:
  """Build the closed-loop policy and its reset hook for ``play.run``."""
  controller = FSMController(skills.SKILLS, InstantBridge(), start="drive_straight")
  switched = {"done": False}

  def on_reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    controller.reset("drive_straight")
    switched["done"] = False

  def policy(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    state = state_from_mjdata(data)
    # Supervisor: request the turn once we have driven far enough.
    if not switched["done"] and state[0] >= _SWITCH_AFTER_X:
      controller.switch_to("turn_left")
      switched["done"] = True
    return controller.step(state)  # a skill/bridge emits the action directly

  return policy, on_reset


def main() -> None:
  policy, on_reset = build_policy()
  play.run(_XML, policy, on_reset=on_reset)


if __name__ == "__main__":
  main()
