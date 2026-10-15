"""Throwaway: inspect one diffusion bridge from any skill to any other skill.

A comes from a recorded skill rollout, B is a selector entry of the target skill,
placed in front of A. Or both come from a parkour capture.

The bridge is shown two ways, like the gap check:
  raw       the model's steps added up, untouched
  smeared   what ships, with the leftover gap to B spread over the bridge
With a capture, also:
  plan      the path parkour actually shipped
  executed  what the robot did while following it

Foot marks: green planted, red planted but sliding, grey in the air.

Run:

  1. uv run python -m mjlab.tasks.bridging.config.g1.skills.transition_view
  2. Open http://localhost:8080

  Parkour capture instead of rollouts:

  uv run python -m mjlab.tasks.bridging.config.g1.skills.transition_view \
    --capture data/bridge_debug/parkour_capture.npz
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import tyro
import viser

import mjlab
from mjlab.tasks.bridging.bridges.dataset.view import show, slot, tint
from mjlab.tasks.bridging.bridges.diffusion.evaluation.gap_check import (
  foot_marks,
  path_metrics,
  raw_and_smeared,
)
from mjlab.tasks.bridging.bridges.diffusion.execution.runtime import (
  latest_planner_checkpoint,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.bridge import DiffusionBridge
from mjlab.tasks.bridging.bridges.diffusion.planner.process import (
  RobotFootKinematics,
)
from mjlab.tasks.bridging.bridges.mixed.bridge import load_bridge
from mjlab.tasks.bridging.config import get_robot
from mjlab.tasks.bridging.config.g1 import selector as resume
from mjlab.tasks.bridging.selector import paths as selector_paths
from mjlab.tasks.bridging.selector.table import EntryTable
from mjlab.utils.lab_api.math import quat_from_euler_xyz
from mjlab.viewer.viser.scene import MjlabViserScene

ROLLOUTS_PATH, STATES_PATH = selector_paths("g1")
ROBOT = "robot/"
LAYERS = ("raw", "smeared", "plan", "executed")
COLORS = {
  "A": (0.25, 0.9, 0.4, 0.25),
  "B": (1.0, 0.3, 0.25, 0.25),
  "context": (0.85, 0.85, 0.85, 0.5),
  "raw": (0.1, 0.45, 1.0, 0.5),
  "smeared": (1.0, 0.6, 0.1, 0.5),
  "plan": (0.7, 0.3, 1.0, 0.5),
  "executed": (0.95, 0.95, 0.2, 0.5),
}
SPLINES = {
  "context": (200, 200, 200),
  "raw": (25, 115, 255),
  "smeared": (255, 150, 25),
  "plan": (180, 80, 255),
  "executed": (240, 240, 50),
}


@dataclass
class ViewCfg:
  checkpoint: Path | None = None
  """Plain or footstep diffusion planner checkpoint, newest plain one when empty"""
  capture: Path | None = None
  """Parkour capture file, replaces the rollout and selector sources"""
  source: str = "walk"
  target: str = "jump"
  entry: int = 4
  """Index among the target skill's selector entries, 4 is parkour's jump f096"""
  duration_s: float = 0.8
  forward: float = 0.5
  lateral: float = 0.0
  yaw_deg: float = 0.0
  context_s: float = 1.0
  rollouts_path: Path = ROLLOUTS_PATH
  selector_path: Path = STATES_PATH
  sample_steps: int | None = None
  floor_height: float | None = 0.0
  """Lowest sole height the planner may produce, None shows the unprojected model"""
  device: str = "cuda:0"
  seed: int = 0
  contact_height: float = 0.05
  slip_speed: float = 0.2
  port: int = 8080


@dataclass
class Transition:
  history: np.ndarray
  """States the planner sees, A is the last one"""
  target: np.ndarray
  """B and the states after it the planner needs"""
  before: np.ndarray
  """Context ending at A"""
  after: np.ndarray
  """Context starting at B"""
  label: str
  extras: dict[str, np.ndarray] = field(default_factory=dict)
  ticks: int | None = None


class Rollouts:
  """Recorded skill rollouts, one trajectory at a time"""

  def __init__(self, path: Path) -> None:
    raw = np.load(path)
    self.states = raw["states"]
    self.skill = raw["skill"]
    self.trajectory = raw["trajectory"]
    self.frame = raw["frame"]
    self.names = tuple(str(name) for name in raw["skill_names"])
    self.trajectories = {
      name: np.unique(self.trajectory[self.skill == index])
      for index, name in enumerate(self.names)
    }

  def rows(self, trajectory: int) -> np.ndarray:
    rows = np.nonzero(self.trajectory == trajectory)[0]
    return rows[np.argsort(self.frame[rows], kind="stable")]

  def slowest(self, skill: str, after: int) -> tuple[int, int]:
    """Trajectory and frame where the skill moves its root the least"""
    rows = np.nonzero(self.skill == self.names.index(skill))[0]
    rows = rows[self.frame[rows] >= after]
    speed = np.linalg.norm(self.states[rows, 7:9], axis=-1)
    row = int(rows[np.argmin(speed)])
    trajectory = int(self.trajectory[row])
    index = int(np.nonzero(self.rows(trajectory) == row)[0][0])
    return int(np.searchsorted(self.trajectories[skill], trajectory)), index


def _yaw(quat: np.ndarray) -> float:
  w, x, y, z = quat
  return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def from_rollouts(
  rollouts: Rollouts,
  table: EntryTable,
  source: str,
  trajectory: int,
  frame: int,
  target: str,
  entry_index: int,
  forward: float,
  lateral: float,
  yaw_deg: float,
  history: int,
  future: int,
  context: int,
) -> tuple[Transition, int]:
  """A from a source rollout, B from a selector entry placed in front of A"""
  rows = rollouts.rows(int(rollouts.trajectories[source][trajectory]))
  frame = min(max(frame, history - 1), len(rows) - 1)
  states = rollouts.states
  before = states[rows[max(frame - context, 0) : frame + 1]]
  window = states[rows[frame - history + 1 : frame + 1]]

  entry = table.of(target)[entry_index]
  target_rows = rollouts.rows(int(rollouts.trajectory[entry.source_row]))
  start = int(np.nonzero(target_rows == entry.source_row)[0][0])
  after = states[target_rows[start : start + context + 1]]
  if not np.allclose(after[0], entry.state, atol=1e-4):
    raise ValueError("Selector states and rollouts are out of sync, rebuild them")

  a = window[-1]
  heading = _yaw(a[3:7])
  cos, sin = math.cos(heading), math.sin(heading)
  xy = a[:2] + np.array([cos * forward - sin * lateral, sin * forward + cos * lateral])
  yaw = torch.tensor([heading + math.radians(yaw_deg)])
  zero = torch.zeros(1)
  placed = torch.cat(
    (
      torch.tensor([[*xy, entry.state[2]]], dtype=torch.float32),
      quat_from_euler_xyz(zero, zero, yaw).float(),
    ),
    dim=-1,
  )
  reference = torch.as_tensor(entry.state[:7])[None]
  moved = resume.place(
    torch.as_tensor(after),
    reference.expand(len(after), -1),
    placed.expand(len(after), -1),
  ).numpy()
  label = (
    f"{source} traj {trajectory} frame {frame} to {target} f{entry.frame:03d} "
    f"({forward:.2f} m ahead, {lateral:.2f} m left, {yaw_deg:.0f} deg)"
  )
  return Transition(window, moved[:future], before, moved, label), frame


def from_capture(path: Path, index: int, history: int) -> Transition:
  raw = np.load(path)
  key = f"b{index}_"
  return Transition(
    history=raw[key + "history"][-history:],
    target=raw[key + "target"][None],
    before=raw[key + "before"],
    after=np.concatenate((raw[key + "target"][None], raw[key + "after"])),
    label=(
      f"parkour bridge {index}: obstacle {int(raw[key + 'obstacle'])} "
      f"to {raw[key + 'skill']} f{int(raw[key + 'entry_frame']):03d}"
    ),
    extras={"plan": raw[key + "plan"], "executed": raw[key + "executed"]},
    ticks=int(raw[key + "ticks"]),
  )


@dataclass
class Shown:
  transition: Transition
  ticks: int
  layers: dict[str, np.ndarray]
  feet: dict[str, np.ndarray]
  readout: str


@torch.no_grad()
def build(
  bridge: DiffusionBridge,
  feet: RobotFootKinematics,
  transition: Transition,
  ticks: int,
  seed: int,
  cfg: ViewCfg,
) -> Shown:
  device = cfg.device
  history = torch.as_tensor(transition.history, device=device)[None]
  target = torch.as_tensor(transition.target, device=device)[None]
  duration = torch.tensor([ticks], device=device)
  torch.manual_seed(seed)
  raw, smeared, gap = raw_and_smeared(bridge, history, target, duration)
  layers = {
    "raw": raw[0, : ticks + 1].cpu().numpy(),
    "smeared": smeared[0, : ticks + 1].cpu().numpy(),
    **transition.extras,
  }

  rows = []
  soles: dict[str, np.ndarray] = {}
  for name, states in layers.items():
    path = torch.as_tensor(states, device=device, dtype=torch.float32)[None]
    soles[name] = feet(path)[0].cpu().numpy()
    values = path_metrics(
      path,
      torch.tensor([len(states) - 1], device=device),
      bridge.fps,
      feet,
      cfg.contact_height,
      cfg.slip_speed,
    )
    rows.append(
      f"| {name} | {float(values['slip_mps']):.2f} "
      f"| {100 * float(values['slip_rate']):.0f} "
      f"| {float(values['steps_per_s']):.1f} "
      f"| {float(values['off_blend_deg']):.1f} |"
    )

  a, b = transition.history[-1], transition.target[0]
  ends = feet(torch.as_tensor(np.stack((a, b)), device=device)[None])[0].cpu().numpy()

  def planted(sole: np.ndarray) -> str:
    sides = [
      name for name, z in zip("LR", sole[:, 2], strict=True) if z <= cfg.contact_height
    ]
    return "+".join(sides) or "none"

  distance = float(np.linalg.norm(b[:2] - a[:2]))
  seconds = ticks / bridge.fps
  lines = [
    transition.label,
    "",
    f"A speed {np.linalg.norm(a[7:9]):.2f} m/s, feet down {planted(ends[0])}",
    f"B speed {np.linalg.norm(b[7:9]):.2f} m/s, feet down {planted(ends[1])}, "
    f"root height {b[2]:.2f} m",
    f"A to B {distance:.2f} m in {seconds:.2f} s, "
    f"needs {distance / seconds:.2f} m/s on average",
    f"gap at B {100 * float(torch.linalg.vector_norm(gap[0, :2])):.1f} cm",
    "",
    "| path | slip m/s | slip % | steps/s | off blend deg |",
    "|---|---|---|---|---|",
    *rows,
  ]
  if transition.ticks is not None and transition.ticks != ticks:
    lines += ["", f"plan and executed are at the captured {transition.ticks} ticks"]
  return Shown(transition, ticks, layers, soles, "\n".join(lines))


def serve(cfg: ViewCfg) -> None:
  checkpoint = cfg.checkpoint or latest_planner_checkpoint()
  print(f"planner {checkpoint}")
  bridge = load_bridge(checkpoint, cfg.device, cfg.sample_steps)
  bridge.floor_height = cfg.floor_height
  feet = RobotFootKinematics(bridge.robot).to(cfg.device)
  context = max(round(cfg.context_s * bridge.fps), 1)

  rollouts: Rollouts | None = None
  table: EntryTable | None = None
  captured = 0
  if cfg.capture is None:
    print("loading rollouts")
    rollouts = Rollouts(cfg.rollouts_path)
    table = EntryTable.load(cfg.selector_path)
  else:
    captured = int(np.load(cfg.capture)["count"])
    if captured == 0:
      raise ValueError(f"{cfg.capture} holds no bridges")

  world = mujoco.MjSpec()
  world.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[20.0, 20.0, 0.1],
    rgba=[0.3, 0.3, 0.32, 1.0],
  )
  world.attach(
    get_robot(bridge.robot).get_spec(),
    prefix=ROBOT,
    frame=world.worldbody.add_frame(),
  )
  model = world.compile()
  where = slot(model, ROBOT)
  ghosts = {}
  for name, color in COLORS.items():
    ghost = copy.deepcopy(model)
    tint(ghost, ROBOT, color)
    ghosts[name] = ghost
  tint(model, ROBOT, (0.0, 0.0, 0.0, 0.0))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  def qpos(state: np.ndarray, shift: np.ndarray) -> np.ndarray:
    value = np.array(model.qpos0, dtype=np.float64)
    show(value, where, state.astype(np.float64), -shift)
    return value

  server = viser.ViserServer(port=cfg.port, label="transition view")
  scene = MjlabViserScene(server, model, num_envs=1)
  scene.camera_tracking_enabled = False
  scene.debug_visualization_enabled = True

  low = bridge.min_steps / bridge.fps
  high = bridge.max_steps / bridge.fps
  with server.gui.add_folder("Planner"):
    duration = server.gui.add_slider(
      "Duration s",
      min=low,
      max=high,
      step=1.0 / bridge.fps,
      initial_value=min(max(cfg.duration_s, low), high),
    )
    seed = server.gui.add_number("Seed", initial_value=cfg.seed, step=1)
    resample = server.gui.add_button("Resample")
  controls: dict[str, Any] = {}
  buttons: dict[str, viser.GuiButtonHandle] = {}
  if rollouts is not None and table is not None:
    with server.gui.add_folder("A, end of a skill rollout"):
      controls["source"] = server.gui.add_dropdown(
        "Skill", rollouts.names, initial_value=cfg.source
      )
      controls["trajectory"] = server.gui.add_slider(
        "Rollout",
        min=0,
        max=len(rollouts.trajectories[cfg.source]) - 1,
        step=1,
        initial_value=0,
      )
      controls["frame"] = server.gui.add_slider(
        "Frame", min=0, max=10_000, step=1, initial_value=context
      )
      buttons["slowest"] = server.gui.add_button("Slowest frame")
    with server.gui.add_folder("B, selector entry"):
      controls["target"] = server.gui.add_dropdown(
        "Skill", table.skills, initial_value=cfg.target
      )
      controls["entry"] = server.gui.add_slider(
        "Entry",
        min=0,
        max=max(len(table.of(cfg.target)) - 1, 1),
        step=1,
        initial_value=cfg.entry,
      )
      controls["forward"] = server.gui.add_slider(
        "Ahead m", min=-1.0, max=3.0, step=0.05, initial_value=cfg.forward
      )
      controls["lateral"] = server.gui.add_slider(
        "Left m", min=-1.0, max=1.0, step=0.05, initial_value=cfg.lateral
      )
      controls["yaw"] = server.gui.add_slider(
        "Turn deg", min=-180.0, max=180.0, step=5.0, initial_value=cfg.yaw_deg
      )
      buttons["auto"] = server.gui.add_button("Distance from speeds")
  else:
    with server.gui.add_folder("Parkour capture"):
      controls["bridge"] = server.gui.add_slider(
        "Bridge", min=0, max=max(captured - 1, 1), step=1, initial_value=0
      )
  with server.gui.add_folder("Show"):
    visible = {
      name: server.gui.add_checkbox(name, initial_value=name != "raw")
      for name in ("context", *LAYERS)
    }
    marks = server.gui.add_dropdown("Foot marks", LAYERS, initial_value="smeared")
  with server.gui.add_folder("Playback"):
    playing = server.gui.add_checkbox("Play", initial_value=True)
    cursor = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    speed = server.gui.add_slider(
      "Speed", min=0.1, max=2.0, step=0.1, initial_value=0.5
    )
  readout = server.gui.add_markdown("")

  pressed: set[str] = set()
  for name, button in {**buttons, "resample": resample}.items():
    button.on_click(lambda _, name=name: pressed.add(name))

  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    client.camera.position = (2.5, -3.0, 1.6)
    client.camera.look_at = (0.6, 0.0, 0.6)

  def make_transition() -> Transition:
    if rollouts is None or table is None:
      assert cfg.capture is not None
      index = min(int(controls["bridge"].value), captured - 1)
      return from_capture(cfg.capture, index, bridge.history)
    source = str(controls["source"].value)
    target = str(controls["target"].value)
    controls["trajectory"].max = len(rollouts.trajectories[source]) - 1
    controls["entry"].max = max(len(table.of(target)) - 1, 1)
    trajectory = min(
      int(controls["trajectory"].value), len(rollouts.trajectories[source]) - 1
    )
    entry = min(int(controls["entry"].value), len(table.of(target)) - 1)
    transition, frame = from_rollouts(
      rollouts,
      table,
      source,
      trajectory,
      int(controls["frame"].value),
      target,
      entry,
      float(controls["forward"].value),
      float(controls["lateral"].value),
      float(controls["yaw"].value),
      bridge.history,
      bridge.future,
      context,
    )
    length = len(rollouts.rows(int(rollouts.trajectories[source][trajectory])))
    controls["frame"].max = length - 1
    controls["frame"].value = frame
    return transition

  shown: Shown | None = None
  handles: list = []
  last_key: tuple = ()
  last_drawn: tuple = ()
  next_frame = time.monotonic()
  while True:
    if "resample" in pressed:
      seed.value = int(seed.value) + 1
    if "slowest" in pressed and rollouts is not None:
      trajectory, frame = rollouts.slowest(str(controls["source"].value), context)
      controls["trajectory"].value = trajectory
      controls["frame"].value = frame
    if "auto" in pressed and shown is not None:
      a, b = shown.transition.history[-1], shown.transition.target[0]
      average = 0.5 * (np.linalg.norm(a[7:9]) + np.linalg.norm(b[7:9]))
      controls["forward"].value = round(average * float(duration.value) / 0.05) * 0.05
    pressed.clear()

    key = (
      float(duration.value),
      int(seed.value),
      *(control.value for control in controls.values()),
    )
    if key != last_key:
      try:
        transition = make_transition()
        ticks = round(float(duration.value) * bridge.fps)
        shown = build(bridge, feet, transition, ticks, int(seed.value), cfg)
        readout.content = shown.readout
      except ValueError as error:
        readout.content = f"error: {error}"
      last_key = (
        float(duration.value),
        int(seed.value),
        *(control.value for control in controls.values()),
      )
    if shown is None:
      time.sleep(0.1)
      continue

    origin = np.array([*shown.transition.history[-1][:2], 0.0])
    start = len(shown.transition.before) - 1
    total = start + shown.ticks + len(shown.transition.after)
    drawn = (id(shown), marks.value, *(box.value for box in visible.values()))
    if drawn != last_drawn:
      for handle in handles:
        handle.remove()
      handles = []
      paths = dict(shown.layers)
      paths["context"] = np.concatenate(
        (shown.transition.before, shown.transition.after)
      )
      for name, states in paths.items():
        if visible[name].value and len(states) > 1:
          handles.append(
            server.scene.add_spline_catmull_rom(
              f"/root/{name}",
              points=(states[:, :3] - origin).astype(np.float32),
              color=SPLINES[name],
            )
          )
      if marks.value in shown.feet:
        soles = shown.feet[marks.value]
        handles.append(
          server.scene.add_point_cloud(
            "/feet",
            points=(soles - origin).reshape(-1, 3).astype(np.float32),
            colors=foot_marks(
              soles, bridge.fps, cfg.contact_height, cfg.slip_speed
            ).reshape(-1, 3),
            point_size=0.02,
          )
        )
      cursor.max = total - 1
      last_drawn = drawn

    now = time.monotonic()
    if playing.value and now >= next_frame:
      cursor.value = (int(cursor.value) + 1) % total
      next_frame = now + 1.0 / (bridge.fps * max(float(speed.value), 1e-3))
    frame = min(int(cursor.value), total - 1)

    scene.clear()
    scene.add_ghost_mesh(
      qpos(shown.transition.history[-1], origin), ghosts["A"], alpha=COLORS["A"][3]
    )
    scene.add_ghost_mesh(
      qpos(shown.transition.target[0], origin), ghosts["B"], alpha=COLORS["B"][3]
    )
    inside = start <= frame <= start + shown.ticks
    if visible["context"].value and not inside:
      states = (
        shown.transition.before[frame]
        if frame < start
        else shown.transition.after[frame - start - shown.ticks]
      )
      scene.add_ghost_mesh(
        qpos(states, origin), ghosts["context"], alpha=COLORS["context"][3]
      )
    if inside:
      for name, states in shown.layers.items():
        if visible[name].value:
          scene.add_ghost_mesh(
            qpos(states[min(frame - start, len(states) - 1)], origin),
            ghosts[name],
            alpha=COLORS[name][3],
          )
    scene.update_from_mjdata(data)
    time.sleep(1.0 / 60.0)


if __name__ == "__main__":
  serve(tyro.cli(ViewCfg, config=mjlab.TYRO_FLAGS))
