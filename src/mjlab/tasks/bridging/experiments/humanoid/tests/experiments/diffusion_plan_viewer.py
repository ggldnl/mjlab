"""Show the generated diffusion plan between walk and kick states.

Run:

    uv run python -m \
      mjlab.tasks.bridging.experiments.humanoid.tests.experiments.diffusion_plan_viewer \
      --checkpoint <diffusion-model.pt>
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro
import viser
from mjviser import ViserMujocoScene

import mjlab
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.bridges import BRIDGES
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.view import (
  UNDERGROUND,
  show,
  slot,
  tint,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.execution.runtime import (
  DiffusionRuntime,
)
from mjlab.tasks.bridging.experiments.humanoid.selector import STATES_PATH, resume
from mjlab.tasks.bridging.experiments.humanoid.selector.table import EntryTable
from mjlab.tasks.bridging.experiments.humanoid.skills.kick import KICK_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.walk import WALK_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import (
  arena,
  load_policy,
  state,
)
from mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2kick import (
  Config as Walk2KickCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2kick import Run
from mjlab.tasks.registry import load_rl_cfg

WALK = 0
BRIDGE = 1
KICK = 2
PREFIXES = ("walk/", "bridge/", "kick/", "start/", "end/")


@dataclass
class Config:
  checkpoint: Path
  walk_checkpoint: Path | None = None
  selector_path: Path = STATES_PATH
  entry: int = 0
  duration_s: float = 0.8
  trigger_distance: float = 0.5
  walk_speed: float = 1.0
  ball_distance: float = 3.0
  context_s: float = 1.0
  sample_steps: int = 50
  max_walk_steps: int = 1000
  device: str = "cuda:0"
  seed: int = 0
  speed: float = 1.0
  port: int = 8080


@dataclass
class Plan:
  states: np.ndarray
  phase: np.ndarray
  fps: float
  start: int
  end: int


@torch.no_grad()
def generate(cfg: Config) -> Plan:
  """Roll the walk to A and generate the surrounding kinematic sequence."""
  torch.manual_seed(cfg.seed)
  spec = BRIDGES["diffusion"]
  env = ManagerBasedRlEnv(arena(spec.task, WALK_TASK_ID, KICK_TASK_ID), cfg.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(spec.task).clip_actions)
  try:
    walk = load_policy(
      WALK_TASK_ID, wrapped, "leaving", cfg.device, cfg.walk_checkpoint
    )
    entries = EntryTable.load(cfg.selector_path).of("kick")
    if not 0 <= cfg.entry < len(entries):
      raise ValueError(f"Kick has {len(entries)} selector states, not {cfg.entry}")
    handoff_cfg = Walk2KickCfg(
      bridge="diffusion",
      bridge_duration_s=cfg.duration_s,
      trigger_distance=cfg.trigger_distance,
      automatic=False,
      walk_speed=cfg.walk_speed,
      ball_distance=cfg.ball_distance,
      bridge_checkpoint=cfg.checkpoint,
      walk_checkpoint=cfg.walk_checkpoint,
      selector_path=cfg.selector_path,
      entry=cfg.entry,
      viewer="none",
      device=cfg.device,
      seed=cfg.seed,
      diffusion_sample_steps=cfg.sample_steps,
    )
    env.reset()
    run = Run(env, {"walk": walk}, entries, handoff_cfg, {"diffusion": spec})
    runtime = run.runtime_bridges["diffusion"]
    if not isinstance(runtime, DiffusionRuntime):
      raise TypeError("Diffusion bridge has no diffusion runtime")
    bridge = runtime.load(cfg.device)

    obs = wrapped.get_observations()
    walk_states: list[torch.Tensor] = []
    for _ in range(cfg.max_walk_steps):
      walk_states.append(state(env)[0].clone())
      action = run(obs)
      if run.distance <= cfg.trigger_distance:
        break
      obs, _, _, _ = wrapped.step(action)
    else:
      raise RuntimeError("Walk policy did not reach the bridge trigger")
    assert run.history is not None

    entry = entries[cfg.entry]
    needed = bridge.future - 1
    if needed and (
      entry.future_states is None
      or entry.future_mask is None
      or len(entry.future_states) < needed
      or not entry.future_mask[:needed].all()
    ):
      raise ValueError(f"Selector entry needs {needed} valid post-B states")
    future = (
      np.empty((0, len(entry.state)), dtype=entry.state.dtype)
      if entry.future_states is None
      else entry.future_states[:needed]
    )
    raw = torch.as_tensor(
      np.concatenate((entry.state[None], future)),
      device=cfg.device,
    )
    reference = torch.as_tensor(entry.state[:7], device=cfg.device).expand(
      bridge.future, -1
    )
    placed = run.target[0, :7].expand(bridge.future, -1)
    target = resume.place(raw, reference, placed)[None]
    duration = round(cfg.duration_s * bridge.fps)
    bridge_states = bridge.generate(
      run.history[:, -bridge.history :],
      target,
      torch.tensor([duration], device=cfg.device),
    ).states[0, : duration + 1]

    context = max(round(cfg.context_s * bridge.fps), 1)
    walk_context = torch.stack(walk_states[-(context + 1) :])
    kick_states: list[torch.Tensor] = []
    for offset in range(context + 1):
      run._hold_motion(offset)
      kick_states.append(
        torch.cat(
          (
            run.motion.body_pos_w[:, 0],
            run.motion.body_quat_w[:, 0],
            run.motion.body_lin_vel_w[:, 0],
            run.motion.body_ang_vel_w[:, 0],
            run.motion.joint_pos,
            run.motion.joint_vel,
          ),
          dim=-1,
        )[0].clone()
      )
    kick_context = torch.stack(kick_states)

    start = len(walk_context) - 1
    end = start + duration
    states = torch.cat((walk_context[:-1], bridge_states, kick_context[1:]))
    phase = np.concatenate(
      (
        np.full(start, WALK),
        np.full(duration + 1, BRIDGE),
        np.full(len(kick_context) - 1, KICK),
      )
    )
    result = states.cpu().numpy().astype(np.float64)
    result[:, :2] -= 0.5 * (result[start, :2] + result[end, :2])
    return Plan(result, phase, bridge.fps, start, end)
  finally:
    wrapped.close()


def serve(cfg: Config) -> None:
  plan = generate(cfg)
  entries = EntryTable.load(cfg.selector_path).of("kick")
  saved = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
  fps = float(saved["fps"])
  min_duration = int(saved["min_steps"]) / fps
  max_duration = int(saved["max_steps"]) / fps

  world = mujoco.MjSpec()
  world.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[20.0, 20.0, 0.1],
    rgba=[0.3, 0.3, 0.32, 1.0],
  )
  for prefix in PREFIXES:
    world.attach(get_spec(), prefix=prefix, frame=world.worldbody.add_frame())
  model = world.compile()
  colors = (
    (0.25, 0.9, 0.4, 0.25),
    (0.2, 0.6, 1.0, 0.25),
    (1.0, 0.55, 0.2, 0.25),
    (0.25, 0.9, 0.4, 0.18),
    (1.0, 0.3, 0.3, 0.18),
  )
  for prefix, color in zip(PREFIXES, colors, strict=True):
    tint(model, prefix, color)
  slots = tuple(slot(model, prefix) for prefix in PREFIXES)
  data = mujoco.MjData(model)

  server = viser.ViserServer(port=cfg.port, label="walk2kick diffusion")
  scene = ViserMujocoScene(server, model, num_envs=1)
  scene.camera_tracking_enabled = False
  with server.gui.add_folder("Handoff"):
    regenerate = server.gui.add_button("Generate")
    trigger = server.gui.add_slider(
      "Trigger distance, m",
      min=0.0,
      max=2.0,
      step=0.05,
      initial_value=cfg.trigger_distance,
    )
    duration = server.gui.add_slider(
      "Bridge duration, s",
      min=min_duration,
      max=max_duration,
      step=1.0 / fps,
      initial_value=cfg.duration_s,
    )
    entry = server.gui.add_slider(
      "Kick state", min=0, max=len(entries) - 1, step=1, initial_value=cfg.entry
    )
    walk_speed = server.gui.add_slider(
      "Walk speed, m/s",
      min=0.1,
      max=2.0,
      step=0.05,
      initial_value=cfg.walk_speed,
    )
  with server.gui.add_folder("Playback"):
    play = server.gui.add_button("Play")
    stop = server.gui.add_button("Stop")
    reset = server.gui.add_button("Reset")
    cursor = server.gui.add_slider(
      "Time",
      min=0.0,
      max=(len(plan.states) - 1) / plan.fps,
      step=1.0 / plan.fps,
      initial_value=0.0,
    )
    speed = server.gui.add_slider(
      "Speed", min=0.1, max=2.0, step=0.1, initial_value=cfg.speed
    )

  roots = []

  def draw_roots() -> None:
    for root in roots:
      root.remove()
    roots.clear()
    for phase, name, color in (
      (WALK, "walk", (60, 220, 100)),
      (BRIDGE, "bridge", (50, 150, 255)),
      (KICK, "kick", (255, 140, 50)),
    ):
      points = plan.states[plan.phase == phase, :3]
      if len(points) > 1:
        roots.append(
          server.scene.add_spline_catmull_rom(f"/{name}", points=points, color=color)
        )

  draw_roots()

  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    center = 0.5 * (plan.states[:, :3].min(0) + plan.states[:, :3].max(0))
    client.camera.position = center + np.array([2.0, -3.0, 1.5])
    client.camera.look_at = center + np.array([0.0, 0.0, 0.5])

  playing = True
  requested = False

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

  @regenerate.on_click
  def _(_) -> None:
    nonlocal requested
    requested = True

  while True:
    if requested:
      plan = generate(
        replace(
          cfg,
          trigger_distance=float(trigger.value),
          duration_s=float(duration.value),
          entry=int(entry.value),
          walk_speed=float(walk_speed.value),
        )
      )
      cursor.max = (len(plan.states) - 1) / plan.fps
      cursor.step = 1.0 / plan.fps
      cursor.value = 0.0
      draw_roots()
      requested = False
    if playing:
      cursor.value = (float(cursor.value) + 1.0 / plan.fps) % (
        len(plan.states) / plan.fps
      )
    index = min(round(float(cursor.value) * plan.fps), len(plan.states) - 1)
    for phase, where in enumerate(slots[:3]):
      shift = np.zeros(3) if plan.phase[index] == phase else UNDERGROUND
      show(data.qpos, where, plan.states[index], shift)
    show(data.qpos, slots[3], plan.states[plan.start], np.zeros(3))
    show(data.qpos, slots[4], plan.states[plan.end], np.zeros(3))
    mujoco.mj_kinematics(model, data)
    scene.update_from_mjdata(data)
    time.sleep(1.0 / (plan.fps * max(float(speed.value), 1e-3)))


if __name__ == "__main__":
  serve(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
