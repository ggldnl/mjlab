"""Harvest the data the bridge is built from, by rolling out the real skills.

We collect two things:

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

import mujoco
import numpy as np

from mjlab.tasks.skills.experiments.diffdrive.experiment import CONFIG, build_model
from mjlab.tasks.skills.experiments.diffdrive.gridworld import GridWorld
from mjlab.tasks.skills.experiments.diffdrive.robot import (
  THETA,
  DiffDrive,
  V,
  X,
  Y,
)
from mjlab.tasks.skills.experiments.diffdrive.skills import (
  CorridorSkill,
  corridor_skills,
)


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
    for _ in range(CONFIG.decimation):
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


def _initiation_start(
  skill: CorridorSkill, rng: np.random.Generator, *, speed: float | None = None
) -> np.ndarray:
  """A random start inside the skill's initiation set, at the entry cell.

  Jitters the lateral offset and heading within the initiation tolerances. The start
  speed is sampled across the band, unless `speed` is given: the window family pins it to
  rest so its rollouts ramp up together and stay time-aligned.
  """
  half = skill.d_tol * 0.7
  phi = skill.phi_tol * 0.7
  ex, ey = skill.world.cell_center(*skill.entry)
  lateral = (-math.sin(skill.heading), math.cos(skill.heading))
  offset = rng.uniform(-half, half)
  v = rng.uniform(0.0, min(1.2, skill.speed)) if speed is None else speed
  return np.array(
    [
      float(ex) + offset * lateral[0],
      float(ey) + offset * lateral[1],
      skill.heading + rng.uniform(-phi, phi),
      v,
      0.0,
    ]
  )


def window_family(
  model: mujoco.MjModel,
  robot: DiffDrive,
  skill: CorridorSkill,
  window_steps: int,
  count: int,
  rng: np.random.Generator,
) -> list[np.ndarray]:
  """A family of the skill's early tubes, one rollout per sampled initiation state.

  A complex robot has no single perfect start, only an initiation set, so the tube is a
  family rather than one line. Each member starts from a jittered initiation state (here
  the spatial part, lateral offset and heading, from rest, so the ramps stay aligned) and
  is rolled forward. Members are clipped to their common length so they can be compared
  and combined into one representative.
  """
  tracks: list[np.ndarray] = []
  for _ in range(count * 4):
    if len(tracks) >= count:
      break
    start = _initiation_start(skill, rng, speed=0.0)
    track = _rollout(model, robot, skill, start, max_steps=window_steps)
    if len(track) > 1:
      tracks.append(track)
  if not tracks:
    raise RuntimeError(f"No window rollouts for skill {skill.cid}.")
  common = min(len(t) for t in tracks)
  return [t[:common] for t in tracks]


def _traj_distance(a: np.ndarray, b: np.ndarray) -> float:
  """Summed per-step pose-and-speed distance between two equal-length tracks."""
  dpos = np.hypot(a[:, X] - b[:, X], a[:, Y] - b[:, Y])
  dyaw = a[:, THETA] - b[:, THETA]
  dhead = np.abs(np.arctan2(np.sin(dyaw), np.cos(dyaw)))
  dspd = np.abs(a[:, V] - b[:, V])
  return float(
    (CONFIG.w_pos * dpos + CONFIG.w_head * dhead + CONFIG.w_speed * dspd).sum()
  )


def representative(family: list[np.ndarray], method: str = "medoid") -> np.ndarray:
  """One representative tube from a family of equal-length tracks.

  medoid  the member closest to all the others: a real, feasible rollout, robust to
          outliers. The safe default.
  mean    the pointwise average, with a circular mean for the heading. Simple, but the
          average of feasible tracks need not itself be feasible.
  """
  stack = np.stack(family)  # [count, m, 5]
  if method == "mean":
    out = stack.mean(axis=0)
    th = stack[:, :, THETA]
    out[:, THETA] = np.arctan2(np.sin(th).mean(axis=0), np.cos(th).mean(axis=0))
    return out
  if method == "medoid":
    total = np.zeros(len(family))
    for i in range(len(family)):
      for j in range(i + 1, len(family)):
        d = _traj_distance(stack[i], stack[j])
        total[i] += d
        total[j] += d
    return stack[int(total.argmin())].copy()
  raise ValueError(f"Unknown representative method: {method!r}")


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
  tracks: list[np.ndarray] = []
  for _ in range(count * 8):
    if len(tracks) >= count:
      break
    start = _initiation_start(skill, rng)
    track = _rollout(model, robot, skill, start, max_steps=600, stop_cell=junction_cell)
    last = track[-1]
    r, c = skill.world.world_to_cell(float(last[X]), float(last[Y]))
    if (int(r), int(c)) == junction_cell and skill.world.is_free(
      float(last[X]), float(last[Y])
    ):
      tracks.append(track)
  return tracks


# Window extraction: trim families of rollouts to fixed-length state windows. These are
# what the dataset stores. A skill is treated as a black box that emits a state sequence;
# nothing here assumes how it was produced.


def _tail(track: np.ndarray, length: int) -> np.ndarray:
  """The last `length` states of a track, front-padded with the first if it is shorter.

  Padding at the front keeps the switch point (the final state) at the last index.
  """
  if len(track) >= length:
    return track[-length:]
  pad = np.repeat(track[:1], length - len(track), axis=0)
  return np.concatenate([pad, track], axis=0)


def _head(track: np.ndarray, length: int) -> np.ndarray:
  """The first `length` states of a track, back-padded with the last if it is shorter."""
  if len(track) >= length:
    return track[:length]
  pad = np.repeat(track[-1:], length - len(track), axis=0)
  return np.concatenate([track, pad], axis=0)


def _stack(members: list[np.ndarray], count: int) -> np.ndarray:
  """Stack a family to [count, L, 5], repeating the last member or truncating to fit."""
  if len(members) < count:
    members = members + [members[-1]] * (count - len(members))
  return np.stack(members[:count])


def start_window_family(
  model: mujoco.MjModel,
  robot: DiffDrive,
  skill: CorridorSkill,
  count: int,
  window_steps: int,
  rng: np.random.Generator,
) -> np.ndarray:
  """[count, L, 5]: many rollouts of a skill's start window (its early tube from rest).

  Each rollout starts inside the initiation set and keeps the first L = window_steps + 1
  states, so the first index is the start.
  """
  length = window_steps + 1
  tracks = window_family(model, robot, skill, window_steps, count, rng)
  return _stack([_head(t, length) for t in tracks], count)


def end_window_family(
  model: mujoco.MjModel,
  robot: DiffDrive,
  skill: CorridorSkill,
  junction_cell: tuple[int, int],
  count: int,
  window_steps: int,
  rng: np.random.Generator,
) -> np.ndarray:
  """[count, L, 5]: many rollouts of a skill's end window (its approach to the switch).

  Each rollout runs to the junction corner and keeps the last L = window_steps + 1 states,
  so the last index is the switch point.
  """
  length = window_steps + 1
  tracks = interrupt_tracks(model, robot, skill, junction_cell, count, rng)
  if not tracks:
    raise RuntimeError(f"No skill {skill.cid} rollouts reached corner {junction_cell}.")
  return _stack([_tail(t, length) for t in tracks], count)


# Visualization (kept out of the harvest primitives, like the world's renderer).

# Playback speed multipliers, mirroring the other diffdrive viewers.
_SPEEDS: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def main() -> None:
  """Spin the viser viewer on the harvested rollouts, one transition at a time.

  For a chosen transition src -> tgt this overlays, in the world:

  * the source skill's interrupt rollouts: many semi-transparent ghost robots driving
    corridor src from jittered starts to the junction corner, where the previous skill
    leaves the robot. Their endpoints (the interrupt states, the last state of each end
    window) are marked with diamonds, and the last window_seconds of every rollout are drawn as
    points: the approach states a same-duration window before the interrupt would save.
  * the target skill's tube family: one rollout per sampled initiation state, each a
    thin line. Their representative (medoid or mean) is the bright path, with dots for
    the states it saves, the single window the bridge aims to join. A ghost replays it.

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

  from mjlab.tasks.skills.experiments.diffdrive.experiment import (
    corridor_speeds,
  )
  from mjlab.tasks.skills.experiments.diffdrive.gridworld import _color, build_world
  from mjlab.tasks.skills.experiments.diffdrive.robot import DiffDrive

  @_dataclass
  class Args:
    cell: float = 1.0  # metres per grid cell
    slow: float = 0.5  # slow-corridor cruise speed (m/s)
    fast: float = 1.5  # fast-corridor cruise speed (m/s)
    rollouts: int = 16  # source interrupt rollouts to overlay per transition
    window_seconds: float = CONFIG.window_seconds  # duration of each saved window
    samples: int = CONFIG.couples_per_junction  # target tube rollouts (the family)
    representative: str = CONFIG.representative  # family summary: medoid or mean
    alpha: float = 0.35  # ghost-robot opacity
    seed: int = 0

  args = tyro.cli(Args)

  world = GridWorld(cell=args.cell)
  robot = DiffDrive()
  speeds = corridor_speeds(world, slow=args.slow, fast=args.fast)
  skills = corridor_skills(world, speeds)
  model = build_model(world, robot)
  transitions = sorted(world.junction_map().items())  # [(src, (cell, tgt)), ...]
  window_steps = round(args.window_seconds / CONFIG.control_dt)  # states per window
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
  n_fam = 0

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
    nonlocal frame, length, xpos, xquat, ghosts, decor, label, n_fam
    src, (corner, tgt) = transitions[idx]
    rng = np.random.default_rng(args.seed)
    accepted = interrupt_tracks(model, robot, skills[src], corner, n_src, rng)
    if not accepted:
      raise RuntimeError(f"No source rollouts reached corner {corner}.")
    # Cycle the accepted tracks up to n_src so the ghost count stays fixed.
    src_tracks = [accepted[i % len(accepted)] for i in range(n_src)]
    # The target tube is a family of initiation-set rollouts; the bridge aims at a single
    # representative of it (medoid or mean).
    family = window_family(model, robot, skills[tgt], window_steps, args.samples, rng)
    window = representative(family, args.representative)
    n_fam = len(family)

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
    # Target tube family: each initiation-set rollout as a thin line, and the chosen
    # representative as a bright line with its saved states as dots.
    decor += [line(f"/paths/fam{j}", t, rgb(tgt), 0.8) for j, t in enumerate(family)]
    decor.append(line("/paths/window", window, rgb(tgt), 4.0))
    decor.append(points("/paths/window_pts", window, rgb(tgt), 0.01, "circle"))
    # The same-duration window before each interrupt: the last window_steps states of
    # every source rollout, the approach states a pre-interrupt window would save.
    pre = np.concatenate([t[-window_steps:] for t in accepted], axis=0)
    decor.append(points("/paths/interrupt_window", pre, rgb(src), 0.01, "circle"))
    # Interrupt states: where the source rollouts end (each end window's last state).
    ends = np.array([t[-1] for t in accepted])
    decor.append(points("/paths/interrupts", ends, rgb(src), 0.01, "diamond"))

    """
    cx, cy = (float(v) for v in world.cell_center(*corner))
    decor.append(
      server.scene.add_icosphere(
        "/paths/corner", radius=0.08, color=(245, 215, 60), position=(cx, cy, 0.03)
      )
    )
    """

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
      "window": f"{args.window_seconds:g} s ({window_steps} states)",
      "target family": f"{n_fam} rollouts via {args.representative}",
      "frame": f"{frame + 1}/{length}",
      "legend": "thin tgt lines = tube family, bright line + dots = representative",
    }
    body = "".join(f"<strong>{k}:</strong> {v}<br/>" for k, v in rows.items())
    info.content = (
      f'<div style="font-size:0.85em;line-height:1.4;padding:0 0.5em 0.4em;">'
      f"{body}</div>"
    )

  build_transition(0)
  show(0)
  control_dt = CONFIG.control_dt
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
