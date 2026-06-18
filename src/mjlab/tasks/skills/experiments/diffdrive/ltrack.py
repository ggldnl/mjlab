"""L-track experiment: drive corridor A, turn left at the junction, drive corridor B.

World: an L of two corridors (half-width ``W``) -- A along +x (centerline y=0) up to
the junction J=(jx, 0), then B along +y (centerline x=jx). The robot must stay
within ``W`` of the active corridor's centerline; leaving it is a crash.

Supervisor (high-level controller): drive A -> at J request turn_left -> when faced
+y, drive B. It hands the switch to the bridge but cannot itself decelerate.

This baseline uses ``InstantBridge`` (no real bridging): the turn is engaged from
high speed, so the robot cannot bleed momentum (turn has no forward authority), it
skids past J, and lands off corridor B's centerline -- "non fa in tempo a girare".
A learned bridge must (1) decelerate into J's low-speed turn set, and (2) if handed
off late, recover the centerline. Run:

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.ltrack
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from mjlab.tasks.skills import play
from mjlab.tasks.skills.bridge import InstantBridge
from mjlab.tasks.skills.config.diffdrive import skills
from mjlab.tasks.skills.config.diffdrive.dynamics import THETA, X, Y, state_from_mjdata
from mjlab.tasks.skills.controller import FSMController

_XML = Path(__file__).parents[2] / "config" / "diffdrive" / "diffdrive.xml"


@dataclass(frozen=True)
class LTrack:
  """L corridor: A along +x to J=(jx,0), then B along +y (centerline x=jx)."""

  jx: float = 3.0  # junction x (length of corridor A's centerline)
  jy: float = 3.0  # length of corridor B
  half_width: float = 0.4  # W: corridor half-width (|offset| > W is a crash)


def build_world(track: LTrack) -> mujoco.MjModel:
  """diffdrive model + visual (non-colliding) walls drawing the L corridor."""
  spec = mujoco.MjSpec.from_file(str(_XML))
  w, jx, jy, half_h, thick = track.half_width, track.jx, track.jy, 0.075, 0.02

  def wall(cx: float, cy: float, sx: float, sy: float) -> None:
    g = spec.worldbody.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos = np.array([cx, cy, half_h])
    g.size = np.array([sx, sy, half_h])
    g.rgba = np.array([0.55, 0.55, 0.62, 1.0])
    g.contype, g.conaffinity = 0, 0  # visual only; crashes are scored analytically

  wall((jx + w) / 2, -w, (jx + w) / 2, thick)  # A outer (bottom), x: 0 .. jx+w
  wall((jx - w) / 2, +w, (jx - w) / 2, thick)  # A inner (top),    x: 0 .. jx-w
  wall(jx + w, (jy - w) / 2, thick, (jy + w) / 2)  # B outer (right), y: -w .. jy
  wall(jx - w, (jy + w) / 2, thick, (jy - w) / 2)  # B inner (left),  y: +w .. jy
  return spec.compile()


def build_policy(
  track: LTrack, handoff_dist: float = 0.0, verbose: bool = True
) -> tuple[play.Policy, play.ResetHook, dict]:
  """Closed-loop policy for the L-track. Returns (policy, on_reset, info).

  ``info`` is updated live with the current phase, peak lateral offset and crash
  flag, so a headless caller can read the outcome. ``handoff_dist`` is how far
  before J the turn is requested (the difficulty knob).
  """
  controller = FSMController(skills.SKILLS, InstantBridge(), start="drive_straight")
  info = {"phase": "A", "crashed": False, "peak_offset": 0.0}

  def set_phase(phase: str) -> None:
    if verbose and info["phase"] != phase:
      print(f"[ltrack] phase {info['phase']} -> {phase}")
    info["phase"] = phase

  def on_reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    controller.reset("drive_straight")
    info.update(phase="A", crashed=False, peak_offset=0.0)

  def policy(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    s = state_from_mjdata(data)

    # Supervisor: phase transitions (decides *when* to switch; controller carries it
    # out via the bridge).
    if info["phase"] == "A" and s[X] >= track.jx - handoff_dist:
      controller.switch_to("turn_left")
      set_phase("TURN")
    elif info["phase"] == "TURN" and s[THETA] >= math.pi / 2 - 0.05:
      controller.switch_to("drive_straight")
      set_phase("B")
    elif info["phase"] == "B" and s[Y] >= track.jy:
      set_phase("DONE")

    # Score: lateral offset from the active corridor's centerline.
    offset = abs(s[Y]) if info["phase"] in ("A", "TURN") else abs(s[X] - track.jx)
    info["peak_offset"] = max(info["peak_offset"], offset)
    if offset > track.half_width and not info["crashed"]:
      info["crashed"] = True
      if verbose:
        print(
          f"[ltrack] CRASH: left corridor (offset {offset:.2f} > W {track.half_width})"
        )

    return controller.step(s)

  return policy, on_reset, info


def main() -> None:
  track = LTrack()
  model = build_world(track)
  policy, on_reset, _ = build_policy(track)
  play.run(model, policy, on_reset=on_reset)


if __name__ == "__main__":
  main()
