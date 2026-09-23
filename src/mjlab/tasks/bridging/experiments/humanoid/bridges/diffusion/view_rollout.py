"""Inspect a recorded walk2kick diffusion rollout in Viser.

Record with walk2kick --bridge diffusion --viewer none --diagnostic-path FILE,
then run this module with --path FILE.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro
import viser
from mjviser import ViserMujocoScene

import mjlab
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.view import (
  show,
  slot,
  tint,
)


@dataclass
class ViewCfg:
  path: Path
  port: int = 8080


def serve(cfg: ViewCfg) -> None:
  with np.load(cfg.path) as saved:
    actual = saved["actual"].astype(np.float64)
    planned = saved["planned"].astype(np.float64)
    phase = saved["phase"]
    kick_errors = saved["kick_errors"]
    target = saved["target"]
    fps = float(saved["fps"])
  if len(actual) == 0 or actual.shape != planned.shape:
    raise ValueError("Diagnostic has no matching actual and planned states")

  world = mujoco.MjSpec()
  world.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[20.0, 20.0, 0.1],
    rgba=[0.3, 0.3, 0.32, 1.0],
  )
  for name in ("actual", "planned"):
    world.attach(get_spec(), prefix=f"{name}/", frame=world.worldbody.add_frame())
  model = world.compile()
  tint(model, "actual/", (0.2, 0.7, 1.0, 0.95))
  tint(model, "planned/", (1.0, 0.55, 0.2, 0.5))
  data = mujoco.MjData(model)
  slots = (slot(model, "actual/"), slot(model, "planned/"))

  server = viser.ViserServer(port=cfg.port, label="diffusion bridge rollout")
  scene = ViserMujocoScene(server, model, num_envs=1)
  scene.camera_tracking_enabled = False
  slider = server.gui.add_slider(
    "Control tick", min=0, max=len(actual) - 1, step=1, initial_value=0
  )
  readout = server.gui.add_markdown("")
  bridge = phase == 1
  if bridge.any():
    server.scene.add_spline_catmull_rom(
      "/planned root", points=planned[bridge, :3], color=(255, 140, 50)
    )
    server.scene.add_spline_catmull_rom(
      "/actual root", points=actual[bridge, :3], color=(50, 180, 255)
    )
  server.scene.add_frame(
    "/target", position=target[:3], wxyz=target[3:7], axes_length=0.3
  )

  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    center = 0.5 * (actual[:, :3].min(0) + actual[:, :3].max(0))
    client.camera.position = center + np.array([1.5, -3.0, 1.5])
    client.camera.look_at = center

  def draw() -> None:
    index = int(slider.value)
    show(data.qpos, slots[0], actual[index], np.zeros(3))
    show(data.qpos, slots[1], planned[index], np.zeros(3))
    mujoco.mj_kinematics(model, data)
    scene.update_from_mjdata(data)
    if phase[index] == 1:
      root = np.linalg.norm(actual[index, :3] - planned[index, :3])
      joint = np.sqrt(np.mean((actual[index, 13:42] - planned[index, 13:42]) ** 2))
      detail = f"Root path error: {root:.3f} m; joint path RMS: {joint:.3f} rad"
    else:
      names = ("anchor m", "body m", "joint rad", "joint rad/s")
      detail = ", ".join(
        f"{name}: {value:.3f}"
        for name, value in zip(names, kick_errors[index], strict=True)
      )
    readout.content = (
      f"**{'bridge' if phase[index] == 1 else 'kick'}**, "
      f"tick {index}, {index / fps:.2f} s  \n"
      f"Blue: simulated robot. Orange: bridge plan or kick reference.  \n{detail}"
    )

  slider.on_update(lambda _: draw())
  draw()
  print(f"[diffusion] serving {cfg.path} on http://localhost:{cfg.port}")
  while True:
    time.sleep(0.1)


if __name__ == "__main__":
  serve(tyro.cli(ViewCfg, config=mjlab.TYRO_FLAGS))
