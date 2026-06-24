"""A saved, reusable rollout dataset for bridge training experiments.

For every skill transition (skill1 -> skill2) the dataset stores two families of rollouts:

  end_windows    many rollouts of skill1's end window: the states leading up to the switch,
                 with the switch point at the last index of each rollout.
  start_windows  many rollouts of skill2's start window: the states it passes through just
                 after it starts, from the start at the first index.

Skills are treated as black boxes: each window is just a sequence of reduced states, with
no assumption about how the skill produced them. That is all an experiment needs to learn a
bridge from one to the other; it can reduce a family, sample from it, or use the whole
spread however it likes. Harvesting runs the real skills in MuJoCo, so we do it once and
save it to a single file.

Run as a script to step through the dataset in the viewer: pick a transition, a window
(skill1's end or skill2's start), and a rollout with the side controls, then Play to watch
that rollout play back in the world.

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.bridge.dataset
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mjlab.tasks.skills.experiments.diffdrive import experiment
from mjlab.tasks.skills.experiments.diffdrive.bridge import rollouts
from mjlab.tasks.skills.experiments.diffdrive.gridworld import GridWorld
from mjlab.tasks.skills.experiments.diffdrive.robot import DiffDrive, X, Y
from mjlab.tasks.skills.experiments.diffdrive.skills import corridor_skills


@dataclass
class RolloutDataset:
  """Per junction, C couples. Couple i at junction t is the pair (end_windows[t, i],
  start_windows[t, i]): one skill1 end-window trajectory and one skill2 start-window
  trajectory, each a single rollout. Skills are black boxes, just state sequences; training
  learns to merge the two trajectories of a couple.

  transitions    list of (src, tgt) ids, length T (which two skills each junction joins).
  end_windows    [T, C, L, 5] skill1 end-window trajectory of each couple.
  start_windows  [T, C, L, 5] skill2 start-window trajectory of each couple.
  """

  transitions: list[tuple[int, int]]
  end_windows: np.ndarray
  start_windows: np.ndarray


def harvest_dataset(
  world: GridWorld,
  speeds: dict[int, float],
  *,
  window_steps: int = experiment.CONFIG.window_steps,
  count: int = experiment.CONFIG.couples_per_junction,
  seed: int = 0,
) -> RolloutDataset:
  """Harvest `count` (skill1 end, skill2 start) trajectory couples per junction."""
  robot = DiffDrive()
  model = experiment.build_model(world, robot)
  skills = corridor_skills(world, speeds)
  rng = np.random.default_rng(seed)
  transitions: list[tuple[int, int]] = []
  end_windows: list[np.ndarray] = []
  start_windows: list[np.ndarray] = []
  for src, (cell, tgt) in sorted(world.junction_map().items()):
    transitions.append((src, tgt))
    end_windows.append(
      rollouts.end_window_family(
        model, robot, skills[src], cell, count, window_steps, rng
      )
    )
    start_windows.append(
      rollouts.start_window_family(model, robot, skills[tgt], count, window_steps, rng)
    )
  return RolloutDataset(transitions, np.stack(end_windows), np.stack(start_windows))


def save_dataset(dataset: RolloutDataset, path: str) -> None:
  """Write a dataset to a single compressed .npz file."""
  np.savez_compressed(
    path,
    transitions=np.asarray(dataset.transitions, dtype=int),
    end_windows=dataset.end_windows,
    start_windows=dataset.start_windows,
  )


def load_dataset(path: str) -> RolloutDataset:
  """Read a dataset written by save_dataset."""
  data = np.load(path, allow_pickle=False)
  transitions = [(int(a), int(b)) for a, b in data["transitions"]]
  return RolloutDataset(transitions, data["end_windows"], data["start_windows"])


_SPEEDS: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def main() -> None:
  """Step through the dataset in the viser viewer, one couple at a time.

  A slider iterates through the dataset couples. Each couple is one skill1 end-window
  trajectory and one skill2 start-window trajectory, drawn as two semi-transparent ghost
  robots (the end window in the source corridor's colour, the start window in the target's)
  with a thin line for each path. Play/Pause/Step animate them along their recorded states.
  The robots are placed at the stored states, not simulated, so this shows exactly what the
  dataset holds. The CLI harvests a fresh dataset or loads a saved one.

      uv run python -m mjlab.tasks.skills.experiments.diffdrive.bridge.dataset
      uv run python -m mjlab.tasks.skills.experiments.diffdrive.bridge.dataset --load rollouts.npz
  """
  import time
  from dataclasses import dataclass as _dataclass

  import mujoco
  import tyro
  import viser
  from mjviser import ViserMujocoScene
  from mjviser.conversions import merge_geoms

  from mjlab.tasks.skills.experiments.diffdrive.gridworld import _color, build_world

  @_dataclass
  class Args:
    load: str | None = None  # load a saved .npz instead of harvesting fresh
    save: str | None = None  # also save the harvested dataset to this .npz
    cell: float = 1.0  # metres per grid cell
    slow: float = 0.5  # slow-corridor cruise speed (m/s)
    fast: float = 1.5  # fast-corridor cruise speed (m/s)
    count: int = 24  # couples to harvest per junction (fresh harvest only)
    alpha: float = 0.35  # ghost-robot opacity
    seed: int = 0

  args = tyro.cli(Args)
  world = GridWorld(cell=args.cell)
  if args.load is not None:
    dataset = load_dataset(args.load)
  else:
    speeds = experiment.corridor_speeds(world, slow=args.slow, fast=args.fast)
    dataset = harvest_dataset(world, speeds, count=args.count, seed=args.seed)
  if args.save is not None:
    save_dataset(dataset, args.save)
    print(f"Saved dataset to {args.save}")

  robot = DiffDrive()
  model = experiment.build_model(world, robot)

  n_junctions, n_couples = dataset.end_windows.shape[:2]
  total = n_junctions * n_couples  # every couple in the dataset
  num_envs = 2  # the couple's end-window ghost and its start-window ghost

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
    """Corridor cid's palette colour as a uint8 RGB triple."""
    return (np.array(_color(cid)[:3]) * 255.0).astype(np.uint8)

  # World background + camera GUI. The model carries no robot, so the only moving geometry
  # is the ghost robots we add and drive ourselves.
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
    """Stack ghost poses over time: (L, num_envs, n_bodies, 3) and (..., 4)."""
    length = max(len(t) for t in tracks)
    xpos = np.zeros((length, num_envs, n_bodies, 3), np.float32)
    xquat = np.zeros((length, num_envs, n_bodies, 4), np.float32)
    for e, track in enumerate(tracks):
      for f, state in enumerate(track):
        xpos[f, e], xquat[f, e] = body_poses(state)
    return xpos, xquat

  # Mutable playback state, rebuilt whenever the element changes.
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

  def build_couple(idx: int) -> None:
    nonlocal frame, length, xpos, xquat, ghosts, decor, label
    t, i = divmod(idx, n_couples)
    src, tgt = dataset.transitions[t]
    end_traj = dataset.end_windows[t, i]
    start_traj = dataset.start_windows[t, i]

    xpos, xquat = pose_frames([end_traj, start_traj])
    length = xpos.shape[0]
    frame = 0
    label = f"{src} -> {tgt}"

    # The end-window ghost gets src's colour; the start-window ghost gets tgt's.
    colors = np.stack([rgb(src), rgb(tgt)]).astype(np.uint8)

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
    decor = [
      line("/paths/end", end_traj, rgb(src), 1.5),
      line("/paths/start", start_traj, rgb(tgt), 1.5),
    ]

  def show(f: int) -> None:
    for i, handle in enumerate(ghosts):
      handle.batched_positions = xpos[f, :, i]
      handle.batched_wxyzs = xquat[f, :, i]

  ui = {"paused": False, "pending": 0, "speed_i": _SPEEDS.index(1.0)}

  with server.gui.add_folder("Dataset couple"):
    couple_sl = server.gui.add_slider(
      "Couple", min=0, max=total - 1, step=1, initial_value=0
    )
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

  def on_couple(_) -> None:
    build_couple(int(couple_sl.value))
    show(0)

  def on_step(_) -> None:
    ui["pending"] += 1

  pause_btn.on_click(on_pause)
  step_btn.on_click(on_step)
  reset_btn.on_click(lambda _: on_reset())
  speed_btns.on_click(on_speed)
  couple_sl.on_update(on_couple)

  def render_info() -> None:
    idx = int(couple_sl.value)
    _, i = divmod(idx, n_couples)
    rows = {
      "couple": f"{idx + 1}/{total}",
      "junction": label,
      "couple in junction": f"{i + 1}/{n_couples}",
      "frame": f"{frame + 1}/{length}",
    }
    body = "".join(f"<strong>{k}:</strong> {v}<br/>" for k, v in rows.items())
    info.content = (
      f'<div style="font-size:0.85em;line-height:1.4;padding:0 0.5em 0.4em;">'
      f"{body}</div>"
    )

  build_couple(0)
  show(0)
  control_dt = experiment.CONFIG.control_dt
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
