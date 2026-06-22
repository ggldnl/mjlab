"""Harvest the data the bridge is built from, by rolling out the real skills.

We never invent a representative junction. Instead we take the skills and the world
as they are and run them, in the real simulator, to collect two things:

* windows  the early stretch of each skill's tube, i.e. the states it actually
           passes through just after it starts. These are the candidate states the
           bridge may join.
* interrupts  the states the robot is in when a switch fires, i.e. where the
           previous skill leaves it at the junction corner. These are where the
           bridge must start from, and they are spread out by jittering the
           previous skill's start within its initiation set.

The same primitives serve training (which needs both, per corridor transition) and
deployment (which needs only the windows, to pick a goal at the moment of a switch).

Run as a script to inspect what is harvested, one corridor transition at a time:

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.bridge.rollouts
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.tasks.skills.experiments.diffdrive.bridge import config
from mjlab.tasks.skills.experiments.diffdrive.controller import junction_map
from mjlab.tasks.skills.experiments.diffdrive.experiment import build_model
from mjlab.tasks.skills.experiments.diffdrive.gridworld import GridWorld
from mjlab.tasks.skills.experiments.diffdrive.robot import (
  DiffDrive,
  X,
  Y,
)
from mjlab.tasks.skills.experiments.diffdrive.skills import (
  CorridorSkill,
  corridor_skills,
)


@dataclass
class Harvest:
  """Per corridor transition: the interrupt states and the next tube's window.

  Arrays are stacked to a fixed length so the env can index them as plain tensors:
  interrupts is [T, N, 5], windows is [T, M, 5], aligned with transitions (a list
  of (source, target) corridor ids).
  """

  transitions: list[tuple[int, int]]
  interrupts: np.ndarray
  windows: np.ndarray


def _entry_state(skill: CorridorSkill, speed: float) -> np.ndarray:
  """Reduced state at the skill's geometric start, aligned, at the given speed."""
  ex, ey = skill.world.cell_center(*skill.entry)
  return np.array([float(ex), float(ey), skill.heading, speed, 0.0])


def _rollout(
  model: mujoco.MjModel,
  robot: DiffDrive,
  skill: CorridorSkill,
  start: np.ndarray,
  *,
  max_steps: int,
  stop_cell: tuple[int, int] | None = None,
) -> np.ndarray:
  """Run one skill from `start` and return its reduced-state track.

  Stops early when the robot reaches `stop_cell` (the junction corner) or leaves a
  corridor. The skill is re-armed first, so it re-checks its initiation set against
  `start` and only drives if `start` is a valid place to begin.
  """
  skill.reset()
  data = mujoco.MjData(model)
  robot.reset(model, data, start)
  mujoco.mj_forward(model, data)
  world = skill.world
  track = [robot.sense(model, data).copy()]
  for _ in range(max_steps):
    state = robot.sense(model, data)
    data.ctrl[:] = robot.actuate(model, data, skill(state))
    for _ in range(config.DECIMATION):
      mujoco.mj_step(model, data)
    state = robot.sense(model, data)
    track.append(state.copy())
    x, y = float(state[X]), float(state[Y])
    if not world.is_free(x, y):
      break
    if stop_cell is not None:
      r, c = world.world_to_cell(x, y)
      if (int(r), int(c)) == stop_cell:
        break
  return np.asarray(track)


def harvest_window(
  model: mujoco.MjModel,
  robot: DiffDrive,
  skill: CorridorSkill,
  window_steps: int,
) -> np.ndarray:
  """The early window of a skill's tube: its first `window_steps` states from rest.

  The skill begins at its entry cell at rest and ramps up, exactly as it does in the
  experiment, so the window holds the real low-to-cruise states the bridge can aim at.
  """
  start = _entry_state(skill, speed=0.0)
  track = _rollout(model, robot, skill, start, max_steps=window_steps)
  return track[: window_steps + 1]


def interrupt_tracks(
  model: mujoco.MjModel,
  robot: DiffDrive,
  skill: CorridorSkill,
  junction_cell: tuple[int, int],
  count: int,
  rng: np.random.Generator,
) -> list[np.ndarray]:
  """Up to `count` interrupt rollouts: full tracks of the skill driven to the corner.

  Each sample jitters the start (lateral offset, heading, speed) within the skill's
  initiation set and runs the skill to the corner, keeping the whole track. Samples
  that crash or never reach the corner are dropped, so fewer than `count` may come
  back. The interrupt states are the last row of each track; the full tracks also show
  where the previous skill carries the robot, which is what the visualizer draws.
  """
  half = skill.d_tol * 0.7
  phi = skill.phi_tol * 0.7
  v_hi = min(1.2, skill.speed)
  ex, ey = skill.world.cell_center(*skill.entry)
  lateral = (-math.sin(skill.heading), math.cos(skill.heading))
  tracks: list[np.ndarray] = []
  for _ in range(count * 8):
    if len(tracks) >= count:
      break
    offset = rng.uniform(-half, half)
    start = np.array(
      [
        float(ex) + offset * lateral[0],
        float(ey) + offset * lateral[1],
        skill.heading + rng.uniform(-phi, phi),
        rng.uniform(0.0, v_hi),
        0.0,
      ]
    )
    track = _rollout(model, robot, skill, start, max_steps=600, stop_cell=junction_cell)
    last = track[-1]
    r, c = skill.world.world_to_cell(float(last[X]), float(last[Y]))
    if (int(r), int(c)) == junction_cell and skill.world.is_free(
      float(last[X]), float(last[Y])
    ):
      tracks.append(track)
  return tracks


def harvest_interrupts(
  model: mujoco.MjModel,
  robot: DiffDrive,
  skill: CorridorSkill,
  junction_cell: tuple[int, int],
  count: int,
  rng: np.random.Generator,
) -> np.ndarray:
  """Interrupt states at the junction corner, spread over the skill's initiation set.

  The reduced state where each accepted rollout (see `interrupt_tracks`) ends, i.e.
  where the previous skill leaves the robot at the junction corner.
  """
  tracks = interrupt_tracks(model, robot, skill, junction_cell, count, rng)
  if not tracks:
    raise RuntimeError(f"No interrupt states reached corner {junction_cell}.")
  return np.asarray([track[-1] for track in tracks])


def _fix_length(arr: np.ndarray, length: int) -> np.ndarray:
  """Pad (by repeating the last row) or truncate `arr` to exactly `length` rows."""
  if len(arr) >= length:
    return arr[:length]
  pad = np.repeat(arr[-1:], length - len(arr), axis=0)
  return np.concatenate([arr, pad], axis=0)


def harvest_transitions(
  world: GridWorld,
  speeds: dict[int, float],
  mode: str,
  *,
  window_steps: int = config.WINDOW_STEPS,
  n_interrupts: int = config.N_INTERRUPTS,
  seed: int = 0,
) -> Harvest:
  """Roll out every corridor transition and stack its interrupts and target window."""
  robot = DiffDrive()
  model = build_model(world, robot)
  skills = corridor_skills(world, speeds, mode=mode)
  rng = np.random.default_rng(seed)
  m = window_steps + 1
  transitions: list[tuple[int, int]] = []
  interrupts: list[np.ndarray] = []
  windows: list[np.ndarray] = []
  for src, (cell, tgt) in sorted(junction_map(world).items()):
    transitions.append((src, tgt))
    window = harvest_window(model, robot, skills[tgt], window_steps)
    windows.append(_fix_length(window, m))
    inter = harvest_interrupts(model, robot, skills[src], cell, n_interrupts, rng)
    interrupts.append(_fix_length(inter, n_interrupts))
  return Harvest(transitions, np.stack(interrupts), np.stack(windows))


def harvest_windows(
  world: GridWorld,
  speeds: dict[int, float],
  mode: str,
  *,
  window_steps: int = config.WINDOW_STEPS,
) -> dict[int, np.ndarray]:
  """The early window of every corridor's skill, keyed by corridor id (for deployment)."""
  robot = DiffDrive()
  model = build_model(world, robot)
  skills = corridor_skills(world, speeds, mode=mode)
  m = window_steps + 1
  return {
    cid: _fix_length(harvest_window(model, robot, skill, window_steps), m)
    for cid, skill in skills.items()
  }


# Visualization (kept out of the harvest primitives, like the world's renderer).

# Playback speed multipliers, mirroring the other diffdrive viewers.
_SPEEDS: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def main() -> None:
  """Spin the viser viewer on the harvested rollouts, one transition at a time.

  For a chosen transition src -> tgt this overlays, in the world:

  * the source skill's interrupt rollouts: many semi-transparent ghost robots driving
    corridor src from jittered starts to the junction corner, where the previous skill
    leaves the robot. They are translucent so the spread of rollouts stays legible, and
    their endpoints (the interrupt states, what `N_INTERRUPTS` samples) are marked.
  * the target skill's window: the early tube of skill tgt from rest, drawn as a bright
    path with its sampled states as points -- the window the bridge may aim at. A ghost
    replays it in the target color.

  A dropdown picks the transition; Play/Pause/Step/Reset and the speed buttons mirror
  the experiment and skills viewers.

      uv run python -m mjlab.tasks.skills.experiments.diffdrive.bridge.rollouts
  """
  import time
  from dataclasses import dataclass as _dataclass

  import tyro
  import viser
  from mjviser import ViserMujocoScene
  from mjviser.conversions import merge_geoms

  from mjlab.tasks.skills.experiments.diffdrive.controller import junction_map
  from mjlab.tasks.skills.experiments.diffdrive.experiment import (
    build_model,
    corridor_speeds,
  )
  from mjlab.tasks.skills.experiments.diffdrive.gridworld import _color, build_world
  from mjlab.tasks.skills.experiments.diffdrive.robot import DiffDrive
  from mjlab.tasks.skills.experiments.diffdrive.skills import corridor_skills

  @_dataclass
  class Args:
    cell: float = 1.0  # metres per grid cell
    mode: str = "cruise"  # skill mode: "cruise" (non-steering) or "hold"
    slow: float = 0.5  # slow-corridor cruise speed (m/s)
    fast: float = 1.5  # fast-corridor cruise speed (m/s)
    rollouts: int = 16  # source interrupt rollouts to overlay per transition
    window_steps: int = config.WINDOW_STEPS  # control ticks of the target window
    alpha: float = 0.35  # ghost-robot opacity
    seed: int = 0

  args = tyro.cli(Args)

  world = GridWorld(cell=args.cell)
  robot = DiffDrive()
  speeds = corridor_speeds(world, slow=args.slow, fast=args.fast)
  skills = corridor_skills(world, speeds, mode=args.mode)
  model = build_model(world, robot)
  transitions = sorted(junction_map(world).items())  # [(src, (cell, tgt)), ...]
  n_src = max(1, args.rollouts)
  num_envs = n_src + 1  # +1 ghost replays the target window

  # One mesh per moving robot body, instanced across all ghosts.
  robot_bodies = [
    b
    for b in range(model.nbody)
    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(
      "robot/"
    )
  ]
  body_meshes = {
    b: merge_geoms(model, [g for g in range(model.ngeom) if model.geom_bodyid[g] == b])
    for b in robot_bodies
  }
  n_bodies = len(robot_bodies)

  def rgb(cid: int) -> np.ndarray:
    """Corridor cid's palette color as a uint8 RGB triple."""
    return (np.array(_color(cid)[:3]) * 255.0).astype(np.uint8)

  # World background + camera GUI. The model carries no robot, so the only moving
  # geometry is the ghost robots we add and drive ourselves.
  server = viser.ViserServer()
  scene = ViserMujocoScene(server, build_world(world), num_envs=1)
  scene.camera_tracking_enabled = False  # keep the whole maze in view
  scene.create_scene_gui()

  scratch = mujoco.MjData(model)

  def body_poses(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-body (position, wxyz) of the robot placed at a reduced state."""
    robot.reset(model, scratch, state)
    mujoco.mj_kinematics(model, scratch)
    pos = np.stack([scratch.xpos[b] for b in robot_bodies])
    quat = np.stack([scratch.xquat[b] for b in robot_bodies])
    return pos, quat

  def pose_frames(tracks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack ghost poses over time: (L, num_envs, n_bodies, 3) and (..., 4).

    Tracks shorter than the longest hold their final pose, so short source rollouts
    pile up at the corner while the rest keep moving.
    """
    length = max(len(t) for t in tracks)
    xpos = np.zeros((length, num_envs, n_bodies, 3), np.float32)
    xquat = np.zeros((length, num_envs, n_bodies, 4), np.float32)
    for e, track in enumerate(tracks):
      for f, state in enumerate(track):
        xpos[f, e], xquat[f, e] = body_poses(state)
      xpos[len(track) :, e] = xpos[len(track) - 1, e]
      xquat[len(track) :, e] = xquat[len(track) - 1, e]
    return xpos, xquat

  # Mutable playback state, rebuilt whenever the transition changes.
  frame = 0
  length = 1
  xpos = np.zeros((1, num_envs, n_bodies, 3), np.float32)
  xquat = np.zeros((1, num_envs, n_bodies, 4), np.float32)
  ghosts: list = []
  decor: list = []
  label = ""

  def line(name: str, track: np.ndarray, color: np.ndarray, width: float):
    pts = np.column_stack([track[:, X], track[:, Y], np.full(len(track), 0.03)]).astype(
      np.float32
    )
    rgb3 = (int(color[0]), int(color[1]), int(color[2]))
    return server.scene.add_spline_catmull_rom(name, pts, color=rgb3, line_width=width)

  def points(name: str, states: np.ndarray, color: np.ndarray, size: float, shape):
    pts = np.column_stack(
      [states[:, X], states[:, Y], np.full(len(states), 0.03)]
    ).astype(np.float32)
    return server.scene.add_point_cloud(
      name, pts, np.tile(color, (len(pts), 1)), point_size=size, point_shape=shape
    )

  def build_transition(idx: int) -> None:
    nonlocal frame, length, xpos, xquat, ghosts, decor, label
    src, (corner, tgt) = transitions[idx]
    rng = np.random.default_rng(args.seed)
    accepted = interrupt_tracks(model, robot, skills[src], corner, n_src, rng)
    if not accepted:
      raise RuntimeError(f"No source rollouts reached corner {corner}.")
    # Cycle the accepted tracks up to n_src so the ghost count stays fixed.
    src_tracks = [accepted[i % len(accepted)] for i in range(n_src)]
    window = harvest_window(model, robot, skills[tgt], args.window_steps)

    xpos, xquat = pose_frames(src_tracks + [window])
    length = xpos.shape[0]
    frame = 0
    label = f"{src} → {tgt}"

    # Source rollouts get src's color; the window ghost gets tgt's.
    colors = np.zeros((num_envs, 3), np.uint8)
    colors[:n_src] = rgb(src)
    colors[n_src] = rgb(tgt)

    for handle in ghosts:
      handle.remove()
    ghosts = [
      server.scene.add_batched_meshes_simple(
        f"/ghosts/body{b}",
        body_meshes[b].vertices.astype(np.float32),
        body_meshes[b].faces.astype(np.int32),
        batched_wxyzs=xquat[0, :, i],
        batched_positions=xpos[0, :, i],
        batched_colors=colors,
        opacity=args.alpha,
        lod="off",
        cast_shadow=False,
      )
      for i, b in enumerate(robot_bodies)
    ]

    for handle in decor:
      handle.remove()
    decor = [line(f"/paths/src{j}", t, rgb(src), 1.5) for j, t in enumerate(accepted)]
    decor.append(line("/paths/window", window, rgb(tgt), 4.0))
    decor.append(points("/paths/window_pts", window, rgb(tgt), 0.04, "circle"))
    # Interrupt states: where the source rollouts end (what N_INTERRUPTS samples).
    ends = np.array([t[-1] for t in accepted])
    decor.append(points("/paths/interrupts", ends, rgb(src), 0.06, "diamond"))
    cx, cy = (float(v) for v in world.cell_center(*corner))
    decor.append(
      server.scene.add_icosphere(
        "/paths/corner", radius=0.08, color=(245, 215, 60), position=(cx, cy, 0.03)
      )
    )

  def show(f: int) -> None:
    for i, handle in enumerate(ghosts):
      handle.batched_positions = xpos[f, :, i]
      handle.batched_wxyzs = xquat[f, :, i]

  ui = {"paused": False, "pending": 0, "speed_i": _SPEEDS.index(1.0)}

  options = [f"{s} → {t}" for s, (_c, t) in transitions]
  with server.gui.add_folder("Transition"):
    trans_dd = server.gui.add_dropdown("Pair", options, initial_value=options[0])
  info = None
  with server.gui.add_folder("Info"):
    info = server.gui.add_html("")
  with server.gui.add_folder("Simulation"):
    pause_btn = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
    step_btn = server.gui.add_button("Step", icon=viser.Icon.PLAYER_TRACK_NEXT)
    reset_btn = server.gui.add_button("Reset", icon=viser.Icon.REFRESH)
    speed_btns = server.gui.add_button_group("Speed", ("Slower", "1x", "Faster"))

  def on_pause(_) -> None:
    ui["paused"] = not ui["paused"]
    pause_btn.label = "Play" if ui["paused"] else "Pause"
    pause_btn.icon = viser.Icon.PLAYER_PLAY if ui["paused"] else viser.Icon.PLAYER_PAUSE

  def on_speed(event) -> None:
    if event.target.value == "Slower":
      ui["speed_i"] = max(0, ui["speed_i"] - 1)
    elif event.target.value == "Faster":
      ui["speed_i"] = min(len(_SPEEDS) - 1, ui["speed_i"] + 1)
    else:
      ui["speed_i"] = _SPEEDS.index(1.0)

  def on_reset() -> None:
    nonlocal frame
    frame = 0
    show(frame)

  def on_transition(_) -> None:
    build_transition(options.index(trans_dd.value))
    show(0)

  def on_step(_) -> None:
    ui["pending"] += 1

  pause_btn.on_click(on_pause)
  step_btn.on_click(on_step)
  reset_btn.on_click(lambda _: on_reset())
  speed_btns.on_click(on_speed)
  trans_dd.on_update(on_transition)

  def render_info() -> None:
    if info is None:
      return
    rows = {
      "transition": label,
      "rollouts (src)": f"{n_src}",
      "window steps (tgt)": f"{args.window_steps}",
      "frame": f"{frame + 1}/{length}",
      "legend": "ghosts = rollouts, bright path + dots = window",
    }
    body = "".join(f"<strong>{k}:</strong> {v}<br/>" for k, v in rows.items())
    info.content = (
      f'<div style="font-size:0.85em;line-height:1.4;padding:0 0.5em 0.4em;">'
      f"{body}</div>"
    )

  build_transition(0)
  show(0)
  control_dt = config.TIMESTEP * config.DECIMATION
  while True:
    if not ui["paused"]:
      frame = (frame + 1) % length
      show(frame)
      ui["pending"] = 0
    elif ui["pending"] > 0:
      frame = (frame + 1) % length
      show(frame)
      ui["pending"] -= 1
    render_info()
    time.sleep(control_dt / _SPEEDS[ui["speed_i"]])


if __name__ == "__main__":
  main()
