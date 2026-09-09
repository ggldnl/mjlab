"""Look at the entry states of a skill, laid out the way the skill passes through them.

Every entry of one skill in one line along its own direction of travel, earliest first, each
one as far along as the skill really got in the time between their frames. So a window reads
as a stretch of rollout rather than as a row of unrelated poses: the robot stands here, and
a third of a second and half a metre later it is crouching there. The dropdown switches
skills, and each skill has its own sequence.

The spacing is measured, not chosen. It comes from EntryTable.trail, which integrates the
root velocities the states themselves carry over the frames between them, because the
recorded ground position is not in a state and would not compose across two medoids if it
were. Nothing here spaces entries evenly, and that is the point: even spacing says every
entry is the same distance on from the last, which is false for every skill that accelerates
through its window.

Which leaves one thing to watch for. A window the skill stands still through, and the jump
opens with thirty frames of exactly that, comes back with several entries on the same tile.
They are drawn on top of each other because that is where they are. --gap prises them apart
by eye, and the table keeps the real distances.

This is how a window gets checked. A window is a guess about which part of a skill is worth
entering, and the way to find out it was wrong is to see a robot mid-flight, or with a hand
where a box should be, or six copies of the same standing pose in the same place.

No physics. These states were recorded under physics already; this writes qpos and runs
forward kinematics, so a foot through the floor here is a defect in the recording, not in
the playback.

Run

1. Draw the entry states, then open the printed address.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.view

2. Open on one skill, with a metre of daylight inserted between entries.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.view --skill jump --gap 1.0

3. Move a window in selector/__init__.py, re-run selector.build, reload the page.
"""

# TODO we have nothing to control in the viewer. The entry states of a conditioned skill
#   depend on the conditioning signal, so it would be nice to show how they move with it

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
  gap: float = 0.0
  """Extra metres inserted between consecutive entries, on top of the measured trail.

  Zero is the truth and the default. Anything else is a legibility knob for a window the
  skill barely moves through, where the real answer is several robots standing in the same
  place. The table reports the measured distances whatever this is set to."""
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


def markdown(skill: str, entries: tuple[Entry, ...], trail: np.ndarray) -> str:
  """One line per entry, in the order they stand on screen.

  Frame and seconds say where in the skill each robot is, which is what a window is being
  judged on. step is how far the skill travelled since the entry before it, so a run of
  zeros is a stretch the robot stood still through and a growing column is a skill
  accelerating. spread says how much the rollouts disagreed there: a large one next to a
  pose that looks odd means the medoid came out of a cloud with no middle.
  """
  steps = np.zeros(len(entries))
  steps[1:] = np.linalg.norm(np.diff(trail[:, 0:2], axis=0), axis=-1)
  return "\n".join(
    [
      f"**{skill}**, earliest frame first along +x, {steps.sum():.2f} m end to end",
      "",
      "| entry | frame | s | step m | spread | clearance |",
      "|---|---|---|---|---|---|",
      *[
        f"| {e.name} | {e.frame} | {e.seconds:.2f} | {step:.2f} | {e.spread:.2f} "
        f"| {e.clearance:+.3f} |"
        for e, step in zip(entries, steps, strict=True)
      ],
    ]
  )


def placed(count: int, trail: np.ndarray, gap: float) -> np.ndarray:
  """The trail as ground shifts to draw at, padded and centred. (N, 3).

  Centred on the middle of its own extent rather than on the first entry, so the camera sits
  in front of the whole sequence whichever skill is picked and however long its window is.
  """
  out = trail.copy()
  out[:, 0] += gap * np.arange(count)
  out[:, 0:2] -= 0.5 * (out[:, 0:2].min(axis=0) + out[:, 0:2].max(axis=0))
  return out


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
    # Off to the side, because the sequence runs along +x now. Seen from in front, every
    # robot but the nearest is hidden behind the one ahead of it
    client.camera.position = (1.0, -5.0, 2.0)
    client.camera.look_at = (0.0, 0.0, 0.8)

  picker = server.gui.add_dropdown("Skill", list(table.skills), initial_value=opening)
  readout = server.gui.add_markdown("")
  labels: list = []

  def draw() -> None:
    entries = table.of(picker.value)
    trail = table.trail(picker.value)
    shifts = placed(len(entries), trail, cfg.gap)
    for handle in labels:
      handle.remove()
    labels.clear()
    for index, where in enumerate(slots):
      if index >= len(entries):
        show(mj_data.qpos, where, parked, UNDERGROUND)
        continue
      state = entries[index].state.astype(np.float64)
      show(mj_data.qpos, where, state, shifts[index])
      labels.append(
        server.scene.add_label(
          f"/entry{index}",
          text=entries[index].name,
          position=(shifts[index][0], shifts[index][1], state[2] + 0.6),
        )
      )
    mujoco.mj_kinematics(model, mj_data)
    scene.update_from_mjdata(mj_data)
    readout.content = markdown(picker.value, entries, trail)

  @picker.on_update
  def _(_) -> None:
    draw()

  draw()
  print(f"[selector] serving on http://localhost:{cfg.port}")
  while True:
    time.sleep(0.1)


if __name__ == "__main__":
  serve(tyro.cli(ViewCfg, config=mjlab.TYRO_FLAGS))
