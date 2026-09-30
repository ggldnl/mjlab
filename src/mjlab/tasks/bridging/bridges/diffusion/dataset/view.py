"""Inspect the retargeted BABEL trajectories used by the planner and tracker.

Run:

    uv run python -m mjlab.tasks.bridging.bridges.diffusion.dataset.view
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro
import viser

import mjlab
from mjlab.retargeting.floor import g1_foot_positions
from mjlab.tasks.bridging.bridges.diffusion.dataset.motions import (
  BABEL_EVAL_MOTIONS,
  motion_files,
)
from mjlab.tasks.bridging.bridges.diffusion.evaluation.view import (
  END_COLOR,
  PLAN_COLOR,
  build_scene,
  ghost_qpos,
)
from mjlab.viewer.viser.scene import MjlabViserScene


@dataclass
class ViewCfg:
  motions: tuple[str, ...] = BABEL_EVAL_MOTIONS
  index: int = 0
  corrected_first: bool = False
  show_original: bool = True
  speed: float = 1.0
  port: int = 8080


@dataclass(frozen=True)
class ClipInfo:
  path: Path
  offset: float


@dataclass(frozen=True)
class Clip:
  path: Path
  states: np.ndarray
  feet: np.ndarray
  correction: np.ndarray
  fps: float


def clip_infos(patterns: tuple[str, ...], corrected_first: bool) -> list[ClipInfo]:
  infos = []
  for path in motion_files(patterns):
    with np.load(path, allow_pickle=False) as raw:
      offset = float(np.abs(raw["ground_correction"]).max(initial=0.0))
    infos.append(ClipInfo(path, offset))
  if corrected_first:
    infos.sort(key=lambda info: (-info.offset, str(info.path)))
  return infos


def load_clip(path: Path) -> Clip:
  with np.load(path, allow_pickle=False) as raw:
    fps = float(np.asarray(raw["fps"]).reshape(-1)[0])
    correction = np.asarray(raw["ground_correction"], dtype=np.float64)
    joint_pos = np.asarray(raw["joint_pos"], dtype=np.float64)
    joint_vel = np.asarray(raw["joint_vel"], dtype=np.float64)
    body_pos = np.asarray(raw["body_pos_w"], dtype=np.float64)
    body_quat = np.asarray(raw["body_quat_w"], dtype=np.float64)
    body_lin_vel = np.asarray(raw["body_lin_vel_w"], dtype=np.float64)
    body_ang_vel = np.asarray(raw["body_ang_vel_w"], dtype=np.float64)
  states = np.concatenate(
    (
      body_pos[:, 0],
      body_quat[:, 0],
      body_lin_vel[:, 0],
      body_ang_vel[:, 0],
      joint_pos,
      joint_vel,
    ),
    axis=-1,
  )
  feet = g1_foot_positions(body_pos, body_quat)
  origin = states[0, :2].copy()
  states[:, :2] -= origin
  feet[..., :2] -= origin
  return Clip(path, states, feet, correction, fps)


def describe(clip: Clip, index: int, count: int) -> str:
  sole = clip.feet[..., 2].min(axis=1)
  return (
    f"| Clip | |\n|---|---|\n"
    f"| index | {index + 1}/{count} |\n"
    f"| path | ``{clip.path.as_posix()}`` |\n"
    f"| frames | {len(clip.states)} at {clip.fps:g} Hz |\n"
    f"| z correction | {clip.correction.min():.4f} to {clip.correction.max():.4f} m |\n"
    f"| lowest sole | {sole.min():.4f} m |\n"
    f"| 2% sole height | {np.quantile(sole, 0.02):.4f} m |\n"
    f"| median sole height | {np.median(sole):.4f} m |"
  )


def line_segments(points: np.ndarray) -> np.ndarray:
  """Connect recorded samples without interpolating between them."""
  return np.stack((points[:-1], points[1:]), axis=1)


def serve(cfg: ViewCfg) -> None:
  infos = clip_infos(cfg.motions, cfg.corrected_first)
  if not 0 <= cfg.index < len(infos):
    raise ValueError(f"index must be between 0 and {len(infos) - 1}")

  model, where, _start_model, original_model, corrected_model = build_scene()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  server = viser.ViserServer(port=cfg.port, label="BABEL dataset")
  scene = MjlabViserScene(server, model, num_envs=1)
  scene.camera_tracking_enabled = False
  scene.debug_visualization_enabled = True

  previous = server.gui.add_button("Previous clip")
  next_clip = server.gui.add_button("Next clip")
  playing = server.gui.add_checkbox("Play", initial_value=True)
  original = server.gui.add_checkbox(
    "Show pre-correction ghost", initial_value=cfg.show_original
  )
  cursor = server.gui.add_slider("Frame", 0, 1, 1, 0)
  speed = server.gui.add_slider("Speed", 0.1, 2.0, 0.1, cfg.speed)
  readout = server.gui.add_markdown("")

  requested = 0
  index = cfg.index
  clip = load_clip(infos[index].path)
  path_handles: list[viser.SceneNodeHandle] = []

  def use_clip() -> None:
    nonlocal clip
    clip = load_clip(infos[index].path)
    cursor.max = len(clip.states) - 1
    cursor.value = 0
    previous.disabled = index == 0
    next_clip.disabled = index == len(infos) - 1
    readout.content = describe(clip, index, len(infos))
    for handle in path_handles:
      handle.remove()
    path_handles.clear()
    path_handles.append(
      server.scene.add_line_segments(
        "/dataset/root",
        points=line_segments(clip.states[:, :3]),
        colors=(25, 115, 255),
        line_width=2.0,
      )
    )
    for foot, color in enumerate(((255, 170, 30), (40, 220, 140))):
      path_handles.append(
        server.scene.add_line_segments(
          f"/dataset/foot_{foot}",
          points=line_segments(clip.feet[:, foot]),
          colors=color,
          line_width=2.0,
        )
      )

  @previous.on_click
  def _(_) -> None:
    nonlocal requested
    requested = -1

  @next_clip.on_click
  def _(_) -> None:
    nonlocal requested
    requested = 1

  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    client.camera.position = (3.0, -3.0, 1.8)
    client.camera.look_at = (0.0, 0.0, 0.8)

  use_clip()
  print(f"[dataset] serving {len(infos)} clips on http://localhost:{cfg.port}")
  next_frame = time.monotonic()
  while True:
    if requested:
      index = min(max(index + requested, 0), len(infos) - 1)
      requested = 0
      use_clip()

    now = time.monotonic()
    if bool(playing.value) and now >= next_frame:
      cursor.value = (int(cursor.value) + 1) % len(clip.states)
      next_frame = now + 1.0 / (clip.fps * max(float(speed.value), 1e-3))
    frame = min(int(cursor.value), len(clip.states) - 1)
    corrected = clip.states[frame]
    scene.clear()
    if bool(original.value) and clip.correction[frame] != 0:
      before = corrected.copy()
      before[2] -= clip.correction[frame]
      scene.add_ghost_mesh(
        ghost_qpos(model, where, before),
        original_model,
        alpha=END_COLOR[3],
        label="before correction",
      )
    scene.add_ghost_mesh(
      ghost_qpos(model, where, corrected),
      corrected_model,
      alpha=PLAN_COLOR[3],
      label="stored trajectory",
    )
    for foot, color in enumerate(((1.0, 0.65, 0.1, 1.0), (0.1, 0.9, 0.55, 1.0))):
      scene.add_sphere(clip.feet[frame, foot], 0.018, color)
    scene.update_from_mjdata(data)
    time.sleep(1.0 / 60.0)


if __name__ == "__main__":
  serve(tyro.cli(ViewCfg, config=mjlab.TYRO_FLAGS))
