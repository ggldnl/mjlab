"""View an analytic policy on a MuJoCo model -- the dummy-agent analog of play.py.

``mjlab/scripts/play.py`` drives a trained checkpoint through a ManagerBasedRlEnv.
The controllers here are plain Python (skills, FSMs, bridges) with no network and
no manager env, so this runs any callable ``policy(model, data) -> ctrl`` directly
on a raw MuJoCo model in a viser viewer.

Library use: ``run(model_path, policy, on_reset=...)`` (what experiment.py calls).
Standalone: ``uv run python -m mjlab.tasks.skills.play --model <path> --agent zero``
to eyeball a bare model with a zero / random policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

Policy = Callable[[mujoco.MjModel, mujoco.MjData], np.ndarray]
ResetHook = Callable[[mujoco.MjModel, mujoco.MjData], None]
# A status callback returns label -> value rows; values may contain inline HTML.
StatusFn = Callable[[], Mapping[str, str]]

# Playback speed multipliers stepped through by the Slower / 1x / Faster buttons.
_SPEEDS: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def run(
  model: mujoco.MjModel | str | Path,
  policy: Policy,
  *,
  decimation: int = 4,
  on_reset: ResetHook | None = None,
  status: StatusFn | None = None,
) -> None:
  """Run ``policy`` on ``model`` (an MjModel or an XML path) in a viser viewer.

  The policy is evaluated once per control step; the simulation then advances
  ``decimation`` physics steps with that control held. ``on_reset`` runs after every
  ``mj_resetData`` (e.g. to reset a controller's internal state or the start pose).

  Side-panel controls mirror ``scripts/play.py``: a Play/Pause toggle, Step (advance
  one control step while paused), Reset Environment (keeps the play/pause state), and
  Slower / 1x / Faster speed buttons. If ``status`` is given, its ``label -> value``
  rows are shown live in an Info box. Loops forever.
  """
  # Local imports: keep the viewer's heavy deps out of the import path of callers
  # that only use this module as a type/lib reference.
  import time

  import viser
  from mjviser import ViserMujocoScene

  if not isinstance(model, mujoco.MjModel):
    model = mujoco.MjModel.from_xml_path(str(model))
  data = mujoco.MjData(model)

  server = viser.ViserServer()
  scene = ViserMujocoScene(server, model, num_envs=1)
  scene.create_scene_gui()

  ui = {"paused": False, "pending": 0, "speed_i": _SPEEDS.index(1.0)}

  info = None
  if status is not None:
    with server.gui.add_folder("Info"):
      info = server.gui.add_html("")

  with server.gui.add_folder("Simulation"):
    pause_btn = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
    step_btn = server.gui.add_button("Step", icon=viser.Icon.PLAYER_TRACK_NEXT)
    reset_btn = server.gui.add_button("Reset Environment", icon=viser.Icon.REFRESH)
    speed_btns = server.gui.add_button_group("Speed", ("Slower", "1x", "Faster"))

  def on_pause(_) -> None:
    ui["paused"] = not ui["paused"]
    pause_btn.label = "Play" if ui["paused"] else "Pause"
    pause_btn.icon = viser.Icon.PLAYER_PLAY if ui["paused"] else viser.Icon.PLAYER_PAUSE

  def on_step(_) -> None:
    ui["pending"] += 1

  def on_speed(event) -> None:
    label = event.target.value
    if label == "Slower":
      ui["speed_i"] = max(0, ui["speed_i"] - 1)
    elif label == "Faster":
      ui["speed_i"] = min(len(_SPEEDS) - 1, ui["speed_i"] + 1)
    else:
      ui["speed_i"] = _SPEEDS.index(1.0)

  def reset() -> None:  # keeps the current play/pause state
    mujoco.mj_resetData(model, data)
    if on_reset is not None:
      on_reset(model, data)
    mujoco.mj_forward(model, data)
    ui["pending"] = 0

  pause_btn.on_click(on_pause)
  step_btn.on_click(on_step)
  reset_btn.on_click(lambda _: reset())
  speed_btns.on_click(on_speed)
  reset()

  def render_status() -> None:
    if status is None or info is None:
      return
    rows = "".join(f"<strong>{k}:</strong> {v}<br/>" for k, v in status().items())
    info.content = (
      f'<div style="font-size:0.85em;line-height:1.4;padding:0 0.5em 0.4em;">'
      f"{rows}</div>"
    )

  def advance() -> None:
    ctrl = np.asarray(policy(model, data), dtype=float)
    data.ctrl[: ctrl.size] = ctrl
    for _ in range(decimation):
      mujoco.mj_step(model, data)
    scene.update_from_mjdata(data)

  while True:
    if not ui["paused"]:
      advance()
      ui["pending"] = 0  # running consumes the queue; don't replay it on next pause
    elif ui["pending"] > 0:
      advance()
      ui["pending"] -= 1
    render_status()
    time.sleep(model.opt.timestep * decimation / _SPEEDS[ui["speed_i"]])


def main() -> None:
  """Quick viewer for a bare model with a dummy policy (zero or random control)."""
  from dataclasses import dataclass

  import tyro

  @dataclass
  class Args:
    model: str
    agent: Literal["zero", "random"] = "zero"

  args = tyro.cli(Args)

  def policy(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    if args.agent == "random":
      lo, hi = model.actuator_ctrlrange.T
      return np.random.uniform(lo, hi)
    return np.zeros(model.nu)

  run(args.model, policy)


if __name__ == "__main__":
  main()
