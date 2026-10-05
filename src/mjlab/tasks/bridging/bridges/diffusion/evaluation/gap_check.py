"""Throwaway check: is the planner's gait bad, or does the gap smear make it bad.

For every episode the planner runs once and its output is shown two ways:
  raw      the model's steps added up from A, untouched; may end away from B
  smeared  what we ship today; the leftover gap to B spread over the bridge

Two kinds of episodes:
  eval     A and B from the same held out clip, like training
  cross    A and B from different clips, B placed like at inference

Prints a table and opens a Viser viewer. Foot marks: green planted, red planted
but sliding, grey in the air.

Run:

  1. uv run python -m mjlab.tasks.bridging.bridges.diffusion.evaluation.gap_check \
       --checkpoint logs/rsl_rl/g1_kinematic_diffusion_planner/<run>/model_30000.pt
  2. Open http://localhost:8080
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro
import viser

import mjlab
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from mjlab.tasks.bridging.bridges.dataset.view import show, slot, tint
from mjlab.tasks.bridging.bridges.diffusion.config import motion_patterns
from mjlab.tasks.bridging.bridges.diffusion.dataset.motions import (
  Layout,
  Windows,
  decode,
  load_motions,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.bridge import DiffusionBridge
from mjlab.tasks.bridging.bridges.diffusion.planner.process import (
  RobotFootKinematics,
)
from mjlab.tasks.bridging.bridges.diffusion.train import (
  CrossTrajectoryPairs,
  PairCfg,
)
from mjlab.viewer.viser.scene import MjlabViserScene


@dataclass
class GapCheckCfg:
  checkpoint: Path
  durations_s: tuple[float, ...] = (0.4, 0.8, 1.2)
  episodes: int = 32
  holdout: int = 8
  sample_steps: int | None = None
  device: str = "cuda:0"
  seed: int = 0
  contact_height: float = 0.05
  slip_speed: float = 0.2
  port: int = 8080
  view: bool = True


VARIANTS = ("raw", "smeared", "recorded")
COLORS = {
  "A": (0.25, 0.9, 0.4, 0.25),
  "B": (1.0, 0.3, 0.25, 0.25),
  "raw": (0.1, 0.45, 1.0, 0.5),
  "smeared": (1.0, 0.6, 0.1, 0.5),
  "recorded": (0.9, 0.9, 0.9, 0.35),
}
ROBOT = "robot/"


def sample_features(
  bridge: DiffusionBridge,
  history: torch.Tensor,
  target: torch.Tensor,
  duration: torch.Tensor,
) -> torch.Tensor:
  """Same as the bridge's generate, but stops before the gap is smeared"""
  return bridge.denoise(history, target, duration)


def integrate_raw(
  features: torch.Tensor, layout: Layout, history: int, duration: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Add up the steps from A with no correction. Also returns the gap left at B"""
  anchor = history - 1
  batch = torch.arange(features.shape[0], device=features.device)
  steps = torch.cat(
    (features[..., layout.root_step], features[..., layout.joint_steps]), dim=-1
  )
  known = torch.cat(
    (features[..., layout.root_position], features[..., layout.joint_positions]),
    dim=-1,
  )
  total = steps.cumsum(dim=1)
  raw = known[:, anchor, None] + total - total[:, anchor, None]
  time_index = torch.arange(features.shape[1], device=features.device)
  raw = torch.where((time_index < anchor)[None, :, None], known, raw)
  gap = known[batch, anchor + duration] - raw[batch, anchor + duration]
  pose = torch.cat((raw[..., :3], features[..., layout.rotation], raw[..., 3:]), -1)
  return pose, gap


@torch.no_grad()
def raw_and_smeared(
  bridge: DiffusionBridge,
  history: torch.Tensor,
  target: torch.Tensor,
  duration: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """One sample shown both ways: raw path, shipped path, and the gap left at B

  Paths start at A, in world frame, gap is in A's heading frame
  """
  anchor = history[:, -1]
  denoised = sample_features(bridge, history, target, duration)
  smeared = bridge.assemble(denoised, anchor, target, duration).states
  features = bridge.normalizer.denormalize(denoised)
  pose, gap = integrate_raw(features, bridge.layout, bridge.history, duration)
  raw = decode(pose, anchor, bridge.fps)[:, bridge.history - 1 :]
  raw[:, 0] = anchor
  return raw, smeared, gap


def path_metrics(
  paths: torch.Tensor,
  duration: torch.Tensor,
  fps: float,
  feet: RobotFootKinematics,
  contact_height: float,
  slip_speed: float,
) -> dict[str, torch.Tensor]:
  """Per episode quality numbers over frames 0 to duration"""
  edges = torch.arange(paths.shape[1] - 1, device=paths.device)[None]
  inside = (edges < duration[:, None])[..., None]
  soles = feet(paths)
  low = soles[..., 2] <= contact_height
  speed = (
    torch.linalg.vector_norm(soles[:, 1:, :, :2] - soles[:, :-1, :, :2], dim=-1) * fps
  )
  grounded = low[:, 1:] & low[:, :-1] & inside
  count = grounded.sum((1, 2)).clamp_min(1)
  touchdowns = (low[:, 1:] & ~low[:, :-1] & inside).sum((1, 2)).float()
  root_speed = (
    torch.linalg.vector_norm(paths[:, 1:, :3] - paths[:, :-1, :3], dim=-1) * fps
  )
  joints = (paths.shape[-1] - 13) // 2
  q = paths[..., 13 : 13 + joints]
  batch = torch.arange(paths.shape[0], device=paths.device)
  time_index = torch.arange(paths.shape[1], device=paths.device)[None]
  phase = (time_index / duration[:, None]).clamp(0, 1)[..., None]
  blend = q[:, :1] + phase * (q[batch, duration][:, None] - q[:, :1])
  frames = (time_index <= duration[:, None]).float()
  deviation = (q - blend).abs().mean(-1)
  seconds = duration.float() / fps
  return {
    "slip_mps": (speed * grounded).sum((1, 2)) / count,
    "slip_rate": ((speed > slip_speed) & grounded).sum((1, 2)) / count,
    "steps_per_s": touchdowns / seconds,
    "max_root_mps": (root_speed * inside[..., 0]).amax(1),
    "off_blend_deg": torch.rad2deg((deviation * frames).sum(1) / frames.sum(1)),
  }


@dataclass
class Episode:
  source: str
  seconds: float
  paths: dict[str, np.ndarray]
  feet: dict[str, np.ndarray]
  metrics: dict[str, dict[str, float]]
  gap_root_cm: float
  gap_joint_deg: float


def foot_marks(
  soles: np.ndarray, fps: float, contact_height: float, slip_speed: float
) -> np.ndarray:
  """Point colors per frame and foot: green planted, red sliding, grey airborne"""
  low = soles[..., 2] <= contact_height
  speed = np.zeros(low.shape)
  speed[1:] = np.linalg.norm(soles[1:, :, :2] - soles[:-1, :, :2], axis=-1) * fps
  colors = np.full((*low.shape, 3), 170, dtype=np.uint8)
  colors[low] = (40, 200, 70)
  colors[low & (speed > slip_speed)] = (230, 40, 40)
  return colors


@torch.no_grad()
def run(cfg: GapCheckCfg) -> tuple[list[Episode], float]:
  bridge = DiffusionBridge.load(cfg.checkpoint, cfg.device, cfg.sample_steps)
  corpus = load_motions(
    motion_patterns(bridge.robot, "val"),
    bridge.process.denoiser.columns,
    cfg.device,
    "all",
    cfg.holdout,
    robot=bridge.robot,
  )
  windows = Windows(
    corpus, bridge.history, bridge.future, bridge.min_steps, bridge.max_steps
  )
  pairs = CrossTrajectoryPairs(
    corpus.dataset(),
    bridge.history,
    bridge.future - 1,
    PairCfg(min_steps=bridge.min_steps, max_steps=bridge.max_steps),
    windows.starts + bridge.history - 1,
  )
  feet = RobotFootKinematics(bridge.robot).to(cfg.device)
  torch.manual_seed(cfg.seed)
  count = cfg.episodes
  device = cfg.device
  batch = torch.arange(count, device=device)[:, None]
  history_offsets = torch.arange(1 - bridge.history, 1, device=device)
  future_offsets = torch.arange(bridge.future, device=device)

  # Same endpoints for every duration, so durations compare fairly
  eval_states, _ = windows.states(count)
  a, b = pairs._draw_indexes(count)
  cross_history = pairs.data.states[pairs.order[a[:, None] + history_offsets]]
  cross_target = pairs.data.states[pairs.order[b[:, None] + future_offsets]]

  episodes: list[Episode] = []
  table: list[tuple[str, float, str, dict[str, float], float, float]] = []
  for seconds in cfg.durations_s:
    ticks = round(seconds * bridge.fps)
    if not bridge.min_steps <= ticks <= bridge.max_steps:
      print(
        f"skip {seconds:g} s: {ticks} ticks is outside "
        f"{bridge.min_steps}..{bridge.max_steps}"
      )
      continue
    duration = torch.full((count,), ticks, device=device, dtype=torch.long)
    sources = {
      "eval": (
        eval_states[:, : bridge.history],
        eval_states[batch, bridge.history - 1 + ticks + future_offsets],
      ),
      "cross": (
        cross_history,
        pairs._place(cross_history, cross_target, duration),
      ),
    }
    for source, (history, target) in sources.items():
      anchor = history[:, -1]
      raw, smeared, gap = raw_and_smeared(bridge, history, target, duration)
      paths = {"raw": raw, "smeared": smeared}
      if source == "eval":
        paths["recorded"] = eval_states[:, bridge.history - 1 :]
      gap_root = torch.linalg.vector_norm(gap[:, :2], dim=-1) * 100
      gap_joint = torch.rad2deg(gap[:, 3:].abs().mean(-1))
      metrics = {
        name: path_metrics(
          path, duration, bridge.fps, feet, cfg.contact_height, cfg.slip_speed
        )
        for name, path in paths.items()
      }
      for name in paths:
        mean = {key: float(value.mean()) for key, value in metrics[name].items()}
        shown_gap = name == "raw"
        table.append(
          (
            source,
            seconds,
            name,
            mean,
            float(gap_root.mean()) if shown_gap else 0.0,
            float(gap_joint.mean()) if shown_gap else 0.0,
          )
        )
      if not cfg.view:
        continue
      soles = {name: feet(path).cpu().numpy() for name, path in paths.items()}
      center = anchor[:, :2].cpu().numpy()
      for index in range(count):
        shift = np.array([*center[index], 0.0])
        cut = ticks + 1
        episode_paths = {
          name: path[index, :cut].cpu().numpy().astype(np.float64)
          for name, path in paths.items()
        }
        episode_paths["A"] = episode_paths["smeared"][:1]
        episode_paths["B"] = episode_paths["smeared"][-1:]
        for states in episode_paths.values():
          states[:, :3] -= shift
        episode_feet = {
          name: value[index, :cut] - shift for name, value in soles.items()
        }
        episodes.append(
          Episode(
            source,
            seconds,
            episode_paths,
            episode_feet,
            {
              name: {key: float(value[index]) for key, value in values.items()}
              for name, values in metrics.items()
            },
            float(gap_root[index]),
            float(gap_joint[index]),
          )
        )

  header = (
    f"{'source':<7}{'sec':>5}  {'path':<9}{'gap cm':>8}{'gap deg':>9}"
    f"{'slip m/s':>10}{'slip %':>8}{'steps/s':>9}{'max root m/s':>14}"
    f"{'off blend deg':>15}"
  )
  print()
  print(header)
  print("-" * len(header))
  for source, seconds, name, mean, gap_cm, gap_deg in table:
    gap_text = f"{gap_cm:8.1f}{gap_deg:9.2f}" if name == "raw" else " " * 17
    print(
      f"{source:<7}{seconds:5.2f}  {name:<9}{gap_text}"
      f"{mean['slip_mps']:10.3f}{100 * mean['slip_rate']:8.1f}"
      f"{mean['steps_per_s']:9.2f}{mean['max_root_mps']:14.2f}"
      f"{mean['off_blend_deg']:15.2f}"
    )
  print()
  print("gap: distance the raw path ends from B, smeared over the bridge today")
  print(
    "off blend: how far joints stray from a straight A to B blend, low = interpolating"
  )
  return episodes, bridge.fps


def serve(cfg: GapCheckCfg, episodes: list[Episode], fps: float) -> None:
  world = mujoco.MjSpec()
  world.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[20.0, 20.0, 0.1],
    rgba=[0.3, 0.3, 0.32, 1.0],
  )
  world.attach(get_spec(), prefix=ROBOT, frame=world.worldbody.add_frame())
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

  def qpos(state: np.ndarray) -> np.ndarray:
    value = np.array(model.qpos0, dtype=np.float64)
    show(value, where, state, np.zeros(3))
    return value

  server = viser.ViserServer(port=cfg.port, label="gap check")
  scene = MjlabViserScene(server, model, num_envs=1)
  scene.camera_tracking_enabled = False
  scene.debug_visualization_enabled = True

  durations = sorted({episode.seconds for episode in episodes})
  with server.gui.add_folder("Episode"):
    source = server.gui.add_dropdown("Source", ("cross", "eval"))
    seconds = server.gui.add_dropdown(
      "Duration s", tuple(f"{value:g}" for value in durations)
    )
    index = server.gui.add_slider(
      "Episode", min=0, max=cfg.episodes - 1, step=1, initial_value=0
    )
    readout = server.gui.add_markdown("")
  with server.gui.add_folder("Show"):
    visible = {
      name: server.gui.add_checkbox(name, initial_value=name != "recorded")
      for name in VARIANTS
    }
    marks = server.gui.add_dropdown("Foot marks", ("smeared", "raw", "recorded"))
  with server.gui.add_folder("Playback"):
    playing = server.gui.add_checkbox("Play", initial_value=True)
    cursor = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    speed = server.gui.add_slider(
      "Speed", min=0.1, max=2.0, step=0.1, initial_value=0.5
    )

  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    client.camera.position = (3.0, -3.0, 1.8)
    client.camera.look_at = (0.5, 0.0, 0.6)

  def current() -> Episode:
    picked = [
      episode
      for episode in episodes
      if episode.source == source.value and f"{episode.seconds:g}" == seconds.value
    ]
    return picked[min(int(index.value), len(picked) - 1)]

  drawn: tuple = ()
  handles: list = []
  next_frame = time.monotonic()
  while True:
    episode = current()
    key = (
      id(episode),
      marks.value,
      *(box.value for box in visible.values()),
    )
    if key != drawn:
      for handle in handles:
        handle.remove()
      handles = []
      for name, color in (
        ("raw", (25, 115, 255)),
        ("smeared", (255, 150, 25)),
        ("recorded", (230, 230, 230)),
      ):
        if name in episode.paths and visible[name].value:
          handles.append(
            server.scene.add_spline_catmull_rom(
              f"/root/{name}", points=episode.paths[name][:, :3], color=color
            )
          )
      if marks.value in episode.feet:
        soles = episode.feet[marks.value]
        handles.append(
          server.scene.add_point_cloud(
            "/feet",
            points=soles.reshape(-1, 3).astype(np.float32),
            colors=foot_marks(soles, fps, cfg.contact_height, cfg.slip_speed).reshape(
              -1, 3
            ),
            point_size=0.02,
          )
        )
      lines = [
        f"gap at B: {episode.gap_root_cm:.1f} cm root, "
        f"{episode.gap_joint_deg:.2f} deg joints",
        "",
        "| path | slip m/s | slip % | steps/s | off blend deg |",
        "|---|---|---|---|---|",
      ]
      for name, values in episode.metrics.items():
        lines.append(
          f"| {name} | {values['slip_mps']:.2f} | {100 * values['slip_rate']:.0f} "
          f"| {values['steps_per_s']:.1f} | {values['off_blend_deg']:.1f} |"
        )
      readout.content = "\n".join(lines)
      cursor.max = len(episode.paths["smeared"]) - 1
      drawn = key

    now = time.monotonic()
    if playing.value and now >= next_frame:
      cursor.value = (int(cursor.value) + 1) % (int(cursor.max) + 1)
      next_frame = now + 1.0 / (fps * max(float(speed.value), 1e-3))
    frame = int(cursor.value)

    scene.clear()
    for name in ("A", "B"):
      scene.add_ghost_mesh(
        qpos(episode.paths[name][0]), ghosts[name], alpha=COLORS[name][3], label=name
      )
    for name in VARIANTS:
      if name in episode.paths and visible[name].value:
        states = episode.paths[name]
        scene.add_ghost_mesh(
          qpos(states[min(frame, len(states) - 1)]),
          ghosts[name],
          alpha=COLORS[name][3],
          label=name,
        )
    scene.update_from_mjdata(data)
    time.sleep(1.0 / 60.0)


def main(cfg: GapCheckCfg) -> None:
  episodes, fps = run(cfg)
  if cfg.view and episodes:
    serve(cfg, episodes, fps)


if __name__ == "__main__":
  main(tyro.cli(GapCheckCfg, config=mjlab.TYRO_FLAGS))
