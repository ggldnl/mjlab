"""Inspect held out footstep diffusion plans in Viser.

Each window shows A in green, B in red and the plan in blue. Footsteps are drawn as
discs: cyan for the left foot, orange for the right. The generated sole paths are
drawn as lines in the same colors, so a sole that misses its disc is easy to spot.

The footstep source switches between the heuristic planner, the footsteps read
from the clip itself, and none. Comparing none with the other two shows whether
the model actually uses its footstep channels.

Run

1. Train a planner, see bridges.mixed.train.

2. View held out windows.

    uv run python -m mjlab.tasks.bridging.bridges.mixed.view --checkpoint logs/rsl_rl/g1_footstep_diffusion_planner/<run>/model_30000.pt
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro
import viser

import mjlab
from mjlab.tasks.bridging.bridges.dataset.view import Slot, show, slot, tint
from mjlab.tasks.bridging.bridges.diffusion.config import motion_patterns
from mjlab.tasks.bridging.bridges.diffusion.dataset.motions import (
  Windows,
  load_motions,
)
from mjlab.tasks.bridging.bridges.mixed.bridge import FootstepBridge, FootstepPlan
from mjlab.tasks.bridging.config import get_robot
from mjlab.viewer.viser.scene import MjlabViserScene

ROBOT = "robot/"
START_COLOR = (0.25, 0.9, 0.4, 0.25)
END_COLOR = (1.0, 0.3, 0.25, 0.25)
PLAN_COLOR = (0.1, 0.45, 1.0, 0.45)
FOOT_COLORS = ((40, 200, 240), (245, 150, 40))
SOURCES = ("planner", "clip", "none")


@dataclass
class ViewCfg:
  checkpoint: Path
  motions: tuple[str, ...] = ()
  holdout: int = 8
  sample_steps: int | None = None
  device: str = "cuda:0"
  seed: int = 0
  speed: float = 1.0
  port: int = 8080


@dataclass
class Plan:
  window: torch.Tensor
  duration: int
  source: str
  states: np.ndarray
  soles: np.ndarray
  footsteps: list[tuple[int, np.ndarray, float]]
  """Foot, world position and yaw of every footstep between A and B."""
  error: float
  """Mean sole distance to the clip's own footsteps."""
  planned_error: float
  """Mean sole distance to the footsteps the model was given, NaN for none."""


def footstep_plan(
  bridge: FootstepBridge,
  window: torch.Tensor,
  duration: torch.Tensor,
  source: str,
) -> FootstepPlan:
  history = window[:, : bridge.history]
  rows = bridge.history - 1 + int(duration[0])
  target = window[:, rows : rows + bridge.future]
  if source == "clip":
    return bridge.clip_plan(window)
  plan = bridge.plan(history, target, duration)
  if source == "none":
    plan.known = torch.zeros_like(plan.known)
  return plan


@torch.no_grad()
def generate(
  bridge: FootstepBridge, window: torch.Tensor, duration: int, source: str, seed: int
) -> Plan:
  states = window.to(bridge.normalizer.mean.device)[None]
  ticks = torch.tensor([duration], device=states.device)
  plan = footstep_plan(bridge, states, ticks, source)
  history = states[:, : bridge.history]
  rows = bridge.history - 1 + duration
  target = states[:, rows : rows + bridge.future]
  devices = [states.device] if states.is_cuda else []
  with torch.random.fork_rng(devices=devices):
    torch.manual_seed(seed)
    path = bridge.generate(history, target, ticks, plan=plan)
  anchor = history[:, -1]
  error = float(bridge.plant_error(path, bridge.clip_plan(states), anchor)[0])
  planned_error = (
    float(bridge.plant_error(path, plan, anchor)[0]) if source != "none" else math.nan
  )

  generated = path.states[0, : duration + 1].double().cpu().numpy()
  soles = bridge.feet(path.states[:, : duration + 1])[0].double().cpu().numpy()
  center = 0.5 * (generated[0, :2] + generated[-1, :2])
  generated[:, :2] -= center
  soles[..., :2] -= center

  w, x, y, z = anchor[0, 3:7].tolist()
  heading = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  turn = np.array(
    ((math.cos(heading), -math.sin(heading)), (math.sin(heading), math.cos(heading)))
  )
  origin = anchor[0, :2].double().cpu().numpy() - center
  start = bridge.history - 1
  contact = plan.contact[0, start : start + duration + 1].cpu().numpy()
  known = plan.known[0, start : start + duration + 1].cpu().numpy()
  channels = plan.channels[0, start : start + duration + 1].double().cpu().numpy()
  footsteps = []
  for foot in range(2):
    planted = contact[:, foot] & known[:, foot]
    for tick in np.flatnonzero(planted & ~np.r_[False, planted[:-1]]):
      local = channels[tick, foot]
      position = np.r_[origin + turn @ local[1:3], local[3]]
      yaw = heading + math.atan2(local[5], local[4])
      footsteps.append((foot, position, yaw))
  return Plan(
    window.cpu(),
    duration,
    source,
    generated,
    soles,
    footsteps,
    error,
    planned_error,
  )


def build_scene(
  robot: str,
) -> tuple[mujoco.MjModel, Slot, mujoco.MjModel, mujoco.MjModel, mujoco.MjModel]:
  world = mujoco.MjSpec()
  world.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[20.0, 20.0, 0.1],
    rgba=[0.3, 0.3, 0.32, 1.0],
  )
  world.attach(
    get_robot(robot).get_spec(), prefix=ROBOT, frame=world.worldbody.add_frame()
  )
  model = world.compile()
  where = slot(model, ROBOT)
  ghosts = []
  for color in (START_COLOR, END_COLOR, PLAN_COLOR):
    ghost = copy.deepcopy(model)
    tint(ghost, ROBOT, color)
    ghosts.append(ghost)
  tint(model, ROBOT, (0.0, 0.0, 0.0, 0.0))
  return model, where, ghosts[0], ghosts[1], ghosts[2]


def ghost_qpos(model: mujoco.MjModel, where: Slot, state: np.ndarray) -> np.ndarray:
  qpos = np.array(model.qpos0, dtype=np.float64)
  show(qpos, where, state, np.zeros(3))
  return qpos


def serve(cfg: ViewCfg) -> None:
  bridge = FootstepBridge.load(cfg.checkpoint, cfg.device, cfg.sample_steps)
  corpus = load_motions(
    cfg.motions or motion_patterns(bridge.robot, "val"),
    bridge.process.denoiser.columns,
    cfg.device,
    "all",
    cfg.holdout,
    robot=bridge.robot,
  )
  if corpus.num_joints != bridge.layout.joints or corpus.fps != bridge.fps:
    raise ValueError("Motion data and checkpoint use different robot layouts or rates")
  windows = Windows(
    corpus, bridge.history, bridge.future, bridge.min_steps, bridge.max_steps
  )
  torch.manual_seed(cfg.seed)

  def draw(source: str, seed: int) -> Plan:
    states, duration = windows.states(1)
    return generate(bridge, states[0], int(duration.item()), source, seed)

  plans = [draw(SOURCES[0], cfg.seed)]
  at = 0
  plan = plans[at]
  model, where, start_ghost, end_ghost, plan_ghost = build_scene(bridge.robot)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  server = viser.ViserServer(port=cfg.port, label="footstep diffusion")
  scene = MjlabViserScene(server, model, num_envs=1)
  scene.camera_tracking_enabled = False
  scene.debug_visualization_enabled = True

  with server.gui.add_folder("Evaluation windows"):
    previous = server.gui.add_button("Previous")
    next_window = server.gui.add_button("Next")
    readout = server.gui.add_markdown("")
  with server.gui.add_folder("Footsteps"):
    source = server.gui.add_dropdown("Source", SOURCES, initial_value=SOURCES[0])
    seed = server.gui.add_number("Seed", initial_value=cfg.seed, step=1)
    regenerate = server.gui.add_button("Regenerate current")
  with server.gui.add_folder("Playback"):
    playing = server.gui.add_checkbox("Play", initial_value=True)
    reset = server.gui.add_button("Reset")
    cursor = server.gui.add_slider(
      "Plan frame", min=0, max=len(plan.states) - 1, step=1, initial_value=0
    )
    speed = server.gui.add_slider(
      "Speed", min=0.1, max=2.0, step=0.1, initial_value=cfg.speed
    )

  requested = ""

  @previous.on_click
  def _(_) -> None:
    nonlocal requested
    requested = "previous"

  @next_window.on_click
  def _(_) -> None:
    nonlocal requested
    requested = "next"

  @regenerate.on_click
  def _(_) -> None:
    nonlocal requested
    requested = "regenerate"

  @source.on_update
  def _(_) -> None:
    nonlocal requested
    requested = "regenerate"

  @reset.on_click
  def _(_) -> None:
    cursor.value = 0

  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    client.camera.position = (3.0, -3.0, 1.8)
    client.camera.look_at = (0.0, 0.0, 0.8)

  handles: list = []

  def use_plan() -> None:
    nonlocal plan, handles
    plan = plans[at]
    for handle in handles:
      handle.remove()
    handles = []
    for foot in range(2):
      handles.append(
        server.scene.add_spline_catmull_rom(
          f"/soles/{foot}", points=plan.soles[:, foot], color=FOOT_COLORS[foot]
        )
      )
    for index, (foot, position, yaw) in enumerate(plan.footsteps):
      handles.append(
        server.scene.add_box(
          f"/footsteps/{index}",
          color=FOOT_COLORS[foot],
          dimensions=(0.2, 0.08, 0.005),
          position=tuple(position),
          wxyz=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
        )
      )
    cursor.max = len(plan.states) - 1
    cursor.value = 0
    previous.disabled = at == 0
    readout.content = (
      f"Window {at + 1}/{len(plans)} &nbsp; {plan.duration / bridge.fps:.2f} s "
      f"&nbsp; footsteps from {plan.source} &nbsp; "
      f"sole error to given footsteps {100 * plan.planned_error:.1f} cm, "
      f"to clip footsteps {100 * plan.error:.1f} cm"
    )

  use_plan()
  next_frame = time.monotonic()
  while True:
    if requested:
      if requested == "previous" and at > 0:
        at -= 1
      elif requested == "next":
        if at + 1 == len(plans):
          plans.append(draw(str(source.value), int(seed.value)))
        at += 1
      elif requested == "regenerate":
        current = plans[at]
        plans[at] = generate(
          bridge, current.window, current.duration, str(source.value), int(seed.value)
        )
      requested = ""
      use_plan()

    now = time.monotonic()
    if bool(playing.value) and now >= next_frame:
      cursor.value = (int(cursor.value) + 1) % len(plan.states)
      next_frame = now + 1.0 / (bridge.fps * max(float(speed.value), 1e-3))
    frame = min(int(cursor.value), len(plan.states) - 1)

    scene.clear()
    scene.add_ghost_mesh(
      ghost_qpos(model, where, plan.states[0]),
      start_ghost,
      alpha=START_COLOR[3],
      label="start",
    )
    scene.add_ghost_mesh(
      ghost_qpos(model, where, plan.states[-1]),
      end_ghost,
      alpha=END_COLOR[3],
      label="end",
    )
    scene.add_ghost_mesh(
      ghost_qpos(model, where, plan.states[frame]),
      plan_ghost,
      alpha=PLAN_COLOR[3],
      label="plan",
    )
    scene.update_from_mjdata(data)
    time.sleep(1.0 / 60.0)


if __name__ == "__main__":
  serve(tyro.cli(ViewCfg, config=mjlab.TYRO_FLAGS))
