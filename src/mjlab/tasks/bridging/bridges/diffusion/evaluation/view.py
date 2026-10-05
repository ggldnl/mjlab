"""Inspect held out kinematic diffusion plans in Viser.

Run:

    uv run python -m \
      mjlab.tasks.bridging.bridges.diffusion.evaluation.view \
      --checkpoint logs/rsl_rl/g1_diffusion_kinematic_planner/<run>/model_30000.pt
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro
import viser

import mjlab
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from mjlab.tasks.bridging.bridges.dataset.view import (
  Slot,
  show,
  slot,
  tint,
)
from mjlab.tasks.bridging.bridges.diffusion.dataset.motions import (
  BABEL_EVAL_MOTIONS,
  Windows,
  load_motions,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.bridge import (
  DiffusionBridge,
  GeneratedPath,
)
from mjlab.viewer.viser.scene import MjlabViserScene

ROBOT = "robot/"
START_COLOR = (0.25, 0.9, 0.4, 0.25)
END_COLOR = (1.0, 0.3, 0.25, 0.25)
PLAN_COLOR = (0.1, 0.45, 1.0, 0.45)


@dataclass
class ViewCfg:
  checkpoint: Path
  motions: tuple[str, ...] = BABEL_EVAL_MOTIONS
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
  rungs: list[np.ndarray]
  fps: float


def boundary_sequences(
  window: torch.Tensor,
  history: int,
  future: int,
  duration: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Extract the conditioning sequences around A and B."""
  target = history - 1 + duration
  return window[:history][None], window[target : target + future][None]


@torch.no_grad()
def generate(
  bridge: DiffusionBridge,
  window: torch.Tensor,
  duration: int,
  sample_steps: int,
  seed: int,
) -> Plan:
  """Generate every clean estimate from one held out A to B pair."""
  if not bridge.min_steps <= duration <= bridge.max_steps:
    raise ValueError("Window duration is outside the trained range")
  if not 1 <= sample_steps <= bridge.process.cfg.steps:
    raise ValueError("Sample steps are outside the diffusion schedule")

  states = window.to(bridge.normalizer.mean.device)
  history, target = boundary_sequences(states, bridge.history, bridge.future, duration)
  bridge.process.cfg = replace(bridge.process.cfg, sample_steps=sample_steps)
  trace: list[GeneratedPath] = []
  devices = [states.device] if states.is_cuda else []
  with torch.random.fork_rng(devices=devices):
    torch.manual_seed(seed)
    bridge.generate(
      history,
      target,
      torch.tensor([duration], device=states.device),
      trace,
    )

  rungs = [
    rung.states[0, : duration + 1].cpu().numpy().astype(np.float64) for rung in trace
  ]
  center = 0.5 * (rungs[-1][0, :2] + rungs[-1][-1, :2])
  for states in rungs:
    states[:, :2] -= center
  return Plan(window.cpu(), duration, rungs, bridge.fps)


def build_scene() -> tuple[
  mujoco.MjModel,
  Slot,
  mujoco.MjModel,
  mujoco.MjModel,
  mujoco.MjModel,
]:
  """Build one hidden G1 and three transparent ghost appearances."""
  world = mujoco.MjSpec()
  world.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[20.0, 20.0, 0.1],
    rgba=[0.3, 0.3, 0.32, 1.0],
  )
  world.attach(get_spec(), prefix=ROBOT, frame=world.worldbody.add_frame())
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
  bridge = DiffusionBridge.load(cfg.checkpoint, cfg.device, cfg.sample_steps)
  corpus = load_motions(
    cfg.motions,
    bridge.process.denoiser.columns,
    cfg.device,
    "all",
    cfg.holdout,
  )
  if corpus.num_joints != bridge.layout.joints or corpus.fps != bridge.fps:
    raise ValueError("Motion data and checkpoint use different robot layouts or rates")
  windows = Windows(
    corpus,
    bridge.history,
    bridge.future,
    bridge.min_steps,
    bridge.max_steps,
  )
  torch.manual_seed(cfg.seed)
  initial_sample_steps = bridge.process.cfg.sample_steps

  def draw(sample_steps: int, seed: int) -> Plan:
    states, duration = windows.states(1)
    return generate(bridge, states[0], int(duration.item()), sample_steps, seed)

  plans = [draw(initial_sample_steps, cfg.seed)]
  at = 0
  plan = plans[at]

  model, where, start_ghost, end_ghost, plan_ghost = build_scene()
  if where.joints.size != corpus.num_joints:
    raise ValueError("Motion data and G1 model have different joint counts")
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  server = viser.ViserServer(port=cfg.port, label="kinematic diffusion")
  scene = MjlabViserScene(server, model, num_envs=1)
  scene.camera_tracking_enabled = False
  scene.debug_visualization_enabled = True

  with server.gui.add_folder("Evaluation windows"):
    previous = server.gui.add_button("Previous")
    next_window = server.gui.add_button("Next")
    readout = server.gui.add_markdown("")
  with server.gui.add_folder("Diffusion"):
    seed = server.gui.add_number("Seed", initial_value=cfg.seed, step=1)
    sample_steps = server.gui.add_slider(
      "Sampling passes",
      min=1,
      max=bridge.process.cfg.steps,
      step=1,
      initial_value=initial_sample_steps,
    )
    regenerate = server.gui.add_button("Regenerate current")
    rung = server.gui.add_slider(
      "Denoising",
      min=1,
      max=len(plan.rungs),
      step=1,
      initial_value=len(plan.rungs),
    )
  with server.gui.add_folder("Playback"):
    playing = server.gui.add_checkbox("Play", initial_value=True)
    reset = server.gui.add_button("Reset")
    cursor = server.gui.add_slider(
      "Plan frame",
      min=0,
      max=len(plan.rungs[-1]) - 1,
      step=1,
      initial_value=0,
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

  @reset.on_click
  def _(_) -> None:
    cursor.value = 0

  root = None
  drawn_rung = -1

  def describe() -> None:
    readout.content = (
      f"Window {at + 1}/{len(plans)} &nbsp; "
      f"{plan.duration / plan.fps:.2f} s &nbsp; "
      f"diffusion {int(rung.value)}/{len(plan.rungs)}"
    )

  def use_plan() -> None:
    nonlocal plan, drawn_rung
    plan = plans[at]
    rung.max = len(plan.rungs)
    rung.value = len(plan.rungs)
    cursor.max = len(plan.rungs[-1]) - 1
    cursor.value = 0
    previous.disabled = at == 0
    drawn_rung = -1
    # describe()

  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    client.camera.position = (3.0, -3.0, 1.8)
    client.camera.look_at = (0.0, 0.0, 0.8)

  use_plan()
  next_frame = time.monotonic()
  while True:
    if requested:
      if requested == "previous" and at > 0:
        at -= 1
      elif requested == "next":
        if at + 1 == len(plans):
          plans.append(draw(int(sample_steps.value), int(seed.value)))
        at += 1
      elif requested == "regenerate":
        current = plans[at]
        plans[at] = generate(
          bridge,
          current.window,
          current.duration,
          int(sample_steps.value),
          int(seed.value),
        )
      requested = ""
      use_plan()

    rung_index = min(int(rung.value) - 1, len(plan.rungs) - 1)
    states = plan.rungs[rung_index]
    if rung_index != drawn_rung:
      if root is not None:
        root.remove()
      root = server.scene.add_spline_catmull_rom(
        "/plan/root", points=states[:, :3], color=(25, 115, 255)
      )
      drawn_rung = rung_index
      # describe()

    now = time.monotonic()
    if bool(playing.value) and now >= next_frame:
      cursor.value = (int(cursor.value) + 1) % len(states)
      next_frame = now + 1.0 / (plan.fps * max(float(speed.value), 1e-3))
    frame = min(int(cursor.value), len(states) - 1)

    scene.clear()
    scene.add_ghost_mesh(
      ghost_qpos(model, where, states[0]),
      start_ghost,
      alpha=START_COLOR[3],
      label="start",
    )
    scene.add_ghost_mesh(
      ghost_qpos(model, where, states[-1]),
      end_ghost,
      alpha=END_COLOR[3],
      label="end",
    )
    scene.add_ghost_mesh(
      ghost_qpos(model, where, states[frame]),
      plan_ghost,
      alpha=PLAN_COLOR[3],
      label="plan",
    )
    scene.update_from_mjdata(data)
    time.sleep(1.0 / 60.0)


if __name__ == "__main__":
  serve(tyro.cli(ViewCfg, config=mjlab.TYRO_FLAGS))
