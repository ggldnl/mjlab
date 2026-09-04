"""Look at the entry points of a skill.

Every entry of one skill, side by side, in table order along +y, with the numbers behind
them. The dropdown switches skills. Reads whichever table it is pointed at, so passing the
candidate file is how you look at what filter.py rejected.

No physics. These states were recorded under physics already; this writes qpos and runs
forward kinematics, so a foot through the floor here is a defect in the recording, not in
the playback.

Run

1. Draw the accepted entries, then open the printed address.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.view

2. Open on one skill, with more room between entries.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.view --skill jump --spacing 1.2

3. Draw the candidates instead, rejected ones included.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.view --path data/selector/candidates.npz
"""

# TODO we have nothing to control in the viewer. The nodes depend on the conditioning signal
#   so it would be nice to show how the nodes change with respect to that signal
# TODO the markdown should go, it shows nothing important except for names. What do the names
#   even mean? They are something like p50, p03, ... are those percentages, percentiles, ...?

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
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.dataset import (
  ROOT_STATE_DIM,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.ground import (
  Slot,
  build,
  show,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.table import (
  HEADER,
  TABLE_PATH,
  Entry,
  EntryTable,
)

UNDERGROUND = np.array([0.0, 0.0, -50.0])
"""Where a copy with nothing to show goes."""

COLOR = (0.35, 0.6, 1.0, 0.9)


@dataclass
class ViewCfg:
  path: Path = TABLE_PATH
  skill: str = ""
  """Which skill to open on. Empty means the first in the table."""
  spacing: float = 1.0
  """Metres between entries."""
  port: int = 8080


def coloured(count: int) -> tuple[mujoco.MjModel, list[Slot]]:
  """count copies of the G1 on a floor, painted so they can be seen through."""
  model, slots = build(count)
  for index in range(count):
    tint(model, f"e{index}/")
  return model, slots


def tint(model: mujoco.MjModel, prefix: str) -> None:
  """Paint one copy's visual geoms and hide its collision ones.

  Collision geoms are the crude convex stand-ins the solver works with, and drawing
  them puts a robot made of boxes inside the robot.

  The material has to be dropped, not merely recoloured. A geom's colour resolves as
  material first and geom_rgba only as a fallback, so a G1 mesh still carrying its
  material comes out the robot's own colour whatever is written here.
  """
  for geom in range(model.ngeom):
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom])
    if body is None or not body.startswith(prefix):
      continue
    if model.geom_contype[geom] or model.geom_conaffinity[geom]:
      model.geom_rgba[geom, 3] = 0.0
      continue
    model.geom_matid[geom] = -1
    model.geom_rgba[geom] = COLOR


def markdown(skill: str, entries: tuple[Entry, ...]) -> str:
  """The skill's rows as they stand on screen, with what each one was asked for.

  The command column is the point of looking. A node is a moment the skill goes through
  whatever it was told to do, so one pose appearing five times over with five different
  commands is one node the clustering split by command, and --clusters is too high. Two
  nodes with genuinely different commands, a left and a right kick, are two nodes and
  should stay two.
  """
  # TODO this markdown is awful. It takes a lot of space and shows nothing important
  return "\n".join(
    [
      f"**{skill}**, left to right along +y",
      "",
      HEADER[0] + " command |",
      HEADER[1] + "---|",
      *[f"{e.row()} {e.command_text()} |" for e in entries],
    ]
  )


def serve(cfg: ViewCfg) -> None:
  table = EntryTable.load(cfg.path)
  if cfg.skill and cfg.skill not in table.skills:
    raise SystemExit(f"This table holds {', '.join(table.skills)}, not '{cfg.skill}'.")
  opening = cfg.skill or table.skills[0]

  slots_needed = max(len(table.of(name)) for name in table.skills)
  model, slots = coloured(slots_needed)
  mj_data = mujoco.MjData(model)

  num_joints = (table.entries[0].state.shape[0] - ROOT_STATE_DIM) // 2
  if slots[0].joints.size != num_joints:
    raise SystemExit(
      f"The table holds {num_joints}-joint states and this G1 has "
      f"{slots[0].joints.size}, so it was built against a different robot."
    )

  # A copy with nothing to show is parked, and it still needs a valid pose to run
  # kinematics on. A zero quaternion is not one
  parked = np.zeros(ROOT_STATE_DIM + 2 * num_joints)
  parked[3] = 1.0

  server = viser.ViserServer(port=cfg.port)
  scene = ViserMujocoScene(server, model, num_envs=1)
  # Off, or the parked copies take the whole scene with them. Camera tracking translates
  # everything by minus the position of the first body that has a joint
  scene.camera_tracking_enabled = False

  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    client.camera.position = (4.0, 0.0, 2.0)
    client.camera.look_at = (0.0, 0.0, 0.8)

  picker = server.gui.add_dropdown("Skill", list(table.skills), initial_value=opening)
  readout = server.gui.add_markdown("")
  labels: list = []

  def draw() -> None:
    entries = table.of(picker.value)
    middle = (len(entries) - 1) / 2.0
    for handle in labels:
      handle.remove()
    labels.clear()
    for index, where in enumerate(slots):
      if index >= len(entries):
        show(mj_data.qpos, where, parked, UNDERGROUND)
        continue
      state = entries[index].state.astype(np.float64)
      shift = np.array([0.0, (index - middle) * cfg.spacing, 0.0])
      show(mj_data.qpos, where, state, shift)
      labels.append(
        server.scene.add_label(
          f"/entry{index}",
          text=entries[index].name,
          position=(0.0, shift[1], state[2] + 0.6),
        )
      )
    mujoco.mj_kinematics(model, mj_data)
    scene.update_from_mjdata(mj_data)
    readout.content = markdown(picker.value, entries)

  @picker.on_update
  def _(_) -> None:
    draw()

  draw()
  print(f"[selector] serving on http://localhost:{cfg.port}")
  while True:
    time.sleep(0.1)


if __name__ == "__main__":
  serve(tyro.cli(ViewCfg, config=mjlab.TYRO_FLAGS))
