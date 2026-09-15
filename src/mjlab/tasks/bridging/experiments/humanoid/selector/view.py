"""Show selected states at their recorded positions along the medoid rollout.

Run:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.view
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
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  ROOT_STATE_DIM,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.view import (
  Slot,
  show,
  slot,
  tint,
)
from mjlab.tasks.bridging.experiments.humanoid.selector import STATES_PATH
from mjlab.tasks.bridging.experiments.humanoid.selector.table import (
  Entry,
  EntryTable,
)

COLOR = (0.35, 0.6, 1.0, 0.9)
PARKED = np.array([0.0, 0.0, -50.0])


@dataclass
class ViewCfg:
  path: Path = STATES_PATH
  skill: str = ""
  port: int = 8080


def model_for(count: int) -> tuple[mujoco.MjModel, list[Slot]]:
  """Build a floor and one tinted robot per selected state."""
  world = mujoco.MjSpec()
  world.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[20.0, 20.0, 0.1],
    rgba=[0.3, 0.3, 0.32, 1.0],
  )
  prefixes = [f"entry{index}/" for index in range(count)]
  for prefix in prefixes:
    world.attach(get_spec(), prefix=prefix, frame=world.worldbody.add_frame())
  model = world.compile()
  for prefix in prefixes:
    tint(model, prefix, COLOR)
  return model, [slot(model, prefix) for prefix in prefixes]


def markdown(entries: tuple[Entry, ...]) -> str:
  """Selected phases and their recorded ground displacement."""
  position = np.stack([entry.state[:2] for entry in entries])
  steps = np.zeros(len(entries))
  steps[1:] = np.linalg.norm(np.diff(position, axis=0), axis=1)
  return "\n".join(
    [
      "| entry | phase | seconds | step m |",
      "|---|---|---|---|",
      *[
        f"| {entry.name} | {entry.frame} | {entry.seconds:.2f} | {step:.2f} |"
        for entry, step in zip(entries, steps, strict=True)
      ],
    ]
  )


def serve(cfg: ViewCfg) -> None:
  table = EntryTable.load(cfg.path)
  opening = cfg.skill or table.skills[0]
  if opening not in table.skills:
    raise ValueError(f"Unknown skill {opening}. Available: {', '.join(table.skills)}")

  model, slots = model_for(max(len(table.of(skill)) for skill in table.skills))
  data = mujoco.MjData(model)
  joints = (table.entries[0].state.size - ROOT_STATE_DIM) // 2
  if slots[0].joints.size != joints:
    raise ValueError("Selector states were recorded with a different robot")
  parked = np.zeros(ROOT_STATE_DIM + 2 * joints)
  parked[3] = 1.0

  server = viser.ViserServer(port=cfg.port)
  scene = ViserMujocoScene(server, model, num_envs=1)
  scene.camera_tracking_enabled = False
  picker = server.gui.add_dropdown("Skill", list(table.skills), initial_value=opening)
  readout = server.gui.add_markdown("")
  labels: list = []

  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    client.camera.position = (1.0, -5.0, 2.0)
    client.camera.look_at = (0.0, 0.0, 0.8)

  def draw() -> None:
    entries = table.of(picker.value)
    positions = np.stack([entry.state[:2] for entry in entries])
    center = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    shift = np.array([-center[0], -center[1], 0.0])
    for label in labels:
      label.remove()
    labels.clear()
    for index, where in enumerate(slots):
      if index >= len(entries):
        show(data.qpos, where, parked, PARKED)
        continue
      entry = entries[index]
      show(data.qpos, where, entry.state.astype(np.float64), shift)
      labels.append(
        server.scene.add_label(
          f"/entry{index}",
          text=entry.name,
          position=entry.state[:3] + shift + np.array([0.0, 0.0, 0.6]),
        )
      )
    mujoco.mj_kinematics(model, data)
    scene.update_from_mjdata(data)
    readout.content = markdown(entries)

  picker.on_update(lambda _: draw())
  draw()
  print(f"[selector] serving on http://localhost:{cfg.port}")
  while True:
    time.sleep(0.1)


if __name__ == "__main__":
  serve(tyro.cli(ViewCfg, config=mjlab.TYRO_FLAGS))
