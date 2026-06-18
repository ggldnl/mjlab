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

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

Policy = Callable[[mujoco.MjModel, mujoco.MjData], np.ndarray]
ResetHook = Callable[[mujoco.MjModel, mujoco.MjData], None]


def run(
  model: mujoco.MjModel | str | Path,
  policy: Policy,
  *,
  decimation: int = 4,
  on_reset: ResetHook | None = None,
) -> None:
  """Run ``policy`` on ``model`` (an MjModel or an XML path) in a viser viewer.

  The policy is evaluated once per control step; the simulation then advances
  ``decimation`` physics steps with that control held. ``on_reset`` runs after every
  ``mj_resetData`` (e.g. to reset a controller's internal state or the start pose).
  Loops forever.
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
  reset_btn = server.gui.add_button("Reset")

  def reset() -> None:
    mujoco.mj_resetData(model, data)
    if on_reset is not None:
      on_reset(model, data)
    mujoco.mj_forward(model, data)

  reset_btn.on_click(lambda _: reset())
  reset()

  while True:
    ctrl = np.asarray(policy(model, data), dtype=float)
    data.ctrl[: ctrl.size] = ctrl
    for _ in range(decimation):
      mujoco.mj_step(model, data)
    scene.update_from_mjdata(data)
    time.sleep(model.opt.timestep * decimation)


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
