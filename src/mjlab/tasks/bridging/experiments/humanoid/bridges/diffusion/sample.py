"""Generate a held-out bridge and animate its kinematic ghost in Viser.

Run:

    uv run python -m \
      mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.sample \
      --checkpoint <model.pt>
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro
import viser
from mjviser import ViserMujocoScene

import mjlab
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DEFAULT_DATASET,
  Dataset,
  load_dataset,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.view import (
  show,
  slot,
  tint,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.bridge import (
  DiffusionBridge,
)

GHOST = "ghost/"
START = "start/"
END = "end/"
GHOST_ALPHA = 0.25


@dataclass
class SampleCfg:
  checkpoint: Path
  dataset: Path = DEFAULT_DATASET
  duration: int = 40
  sample_steps: int = 50
  device: str = "cuda:0"
  seed: int = 0
  speed: float = 1.0
  port: int = 8080


@torch.no_grad()
def _generate(
  bridge: DiffusionBridge, data: Dataset, duration_steps: int
) -> np.ndarray:
  length = bridge.history + duration_steps + bridge.future - 1
  segments = data.segments(length - 1, length - 1)
  start = segments.starts[
    torch.randint(segments.starts.numel(), (1,), device=data.states.device)
  ]
  rows = segments.order[
    start[:, None] + torch.arange(length, device=data.states.device)
  ]
  history = data.states[rows[:, : bridge.history]]
  target_start = bridge.history - 1 + duration_steps
  target = data.states[rows[:, target_start : target_start + bridge.future]]
  duration = torch.tensor([duration_steps], device=data.states.device)
  path = bridge.generate(history, target, duration).states[
    0, : duration_steps + bridge.future
  ]
  if not torch.equal(path[0], history[0, -1]) or not torch.equal(
    path[duration_steps:], target[0]
  ):
    raise AssertionError("generated path missed an exact boundary")
  return path.cpu().numpy().astype(np.float64)


def _center(states: np.ndarray, duration: int) -> np.ndarray:
  """Place the A to B midpoint at the viewer origin."""
  states = states.copy()
  states[:, :2] -= 0.5 * (states[0, :2] + states[duration, :2])
  return states


@torch.no_grad()
def generate(cfg: SampleCfg) -> tuple[np.ndarray, float, int]:
  """Generate one bridge from a held-out physical rollout."""
  torch.manual_seed(cfg.seed)
  bridge = DiffusionBridge.load(cfg.checkpoint, cfg.device, cfg.sample_steps)
  data = load_dataset(cfg.dataset, cfg.device, "eval")
  if data.fps != bridge.fps or data.num_joints != bridge.layout.joints:
    raise ValueError("checkpoint and dataset use different robots or control rates")
  if not bridge.min_steps <= cfg.duration <= bridge.max_steps:
    raise ValueError("duration must lie inside the checkpoint's trained range")
  return _generate(bridge, data, cfg.duration), bridge.fps, bridge.future


def serve(cfg: SampleCfg) -> None:
  """Generate a path, then play it as a kinematic G1 ghost."""
  torch.manual_seed(cfg.seed)
  bridge = DiffusionBridge.load(cfg.checkpoint, cfg.device, cfg.sample_steps)
  dataset = load_dataset(cfg.dataset, cfg.device, "eval")
  if dataset.fps != bridge.fps or dataset.num_joints != bridge.layout.joints:
    raise ValueError("checkpoint and dataset use different robots or control rates")
  if not bridge.min_steps <= cfg.duration <= bridge.max_steps:
    raise ValueError("duration must lie inside the checkpoint's trained range")
  states = _center(_generate(bridge, dataset, cfg.duration), cfg.duration)
  fps = bridge.fps
  world = mujoco.MjSpec()
  world.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[20.0, 20.0, 0.1],
    rgba=[0.3, 0.3, 0.32, 1.0],
  )
  for prefix in (GHOST, START, END):
    world.attach(get_spec(), prefix=prefix, frame=world.worldbody.add_frame())
  model = world.compile()
  tint(model, GHOST, (0.2, 0.6, 1.0, GHOST_ALPHA))
  tint(model, START, (0.25, 0.9, 0.4, GHOST_ALPHA))
  tint(model, END, (1.0, 0.3, 0.3, GHOST_ALPHA))
  ghost = slot(model, GHOST)
  start = slot(model, START)
  end = slot(model, END)
  if ghost.joints.size != (states.shape[1] - 13) // 2:
    raise ValueError("generated path and G1 model have different joint counts")
  data = mujoco.MjData(model)

  server = viser.ViserServer(port=cfg.port, label="diffusion bridge sample")
  scene = ViserMujocoScene(server, model, num_envs=1)
  scene.camera_tracking_enabled = False
  play = server.gui.add_button("Play")
  stop = server.gui.add_button("Stop")
  reset = server.gui.add_button("Reset")
  next_pair = server.gui.add_button("Next pair")
  cursor = server.gui.add_slider(
    "Time",
    min=0.0,
    max=(len(states) - 1) / fps,
    step=1.0 / fps,
    initial_value=0.0,
  )
  speed = server.gui.add_slider(
    "Speed", min=0.1, max=2.0, step=0.1, initial_value=cfg.speed
  )
  root = server.scene.add_spline_catmull_rom(
    "/root", points=states[:, :3], color=(50, 150, 255)
  )

  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    center = 0.5 * (states[:, :3].min(0) + states[:, :3].max(0))
    client.camera.position = center + np.array([2.0, -3.0, 1.5])
    client.camera.look_at = center + np.array([0.0, 0.0, 0.5])

  playing = True
  redraw = False

  @play.on_click
  def _(_) -> None:
    nonlocal playing
    playing = True

  @stop.on_click
  def _(_) -> None:
    nonlocal playing
    playing = False

  @reset.on_click
  def _(_) -> None:
    cursor.value = 0.0

  @next_pair.on_click
  def _(_) -> None:
    nonlocal redraw
    redraw = True

  while True:
    if redraw:
      states = _center(_generate(bridge, dataset, cfg.duration), cfg.duration)
      cursor.max = (len(states) - 1) / fps
      cursor.value = 0.0
      root.remove()
      root = server.scene.add_spline_catmull_rom(
        "/root", points=states[:, :3], color=(50, 150, 255)
      )
      redraw = False
    if playing:
      cursor.value = (float(cursor.value) + 1.0 / fps) % (len(states) / fps)
    index = min(round(float(cursor.value) * fps), len(states) - 1)
    show(data.qpos, ghost, states[index], np.zeros(3))
    show(data.qpos, start, states[0], np.zeros(3))
    show(data.qpos, end, states[cfg.duration], np.zeros(3))
    mujoco.mj_kinematics(model, data)
    scene.update_from_mjdata(data)
    time.sleep(1.0 / (fps * max(float(speed.value), 1e-3)))


if __name__ == "__main__":
  serve(tyro.cli(SampleCfg, config=mjlab.TYRO_FLAGS))
