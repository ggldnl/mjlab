"""Viewer for the windows the bridge is trained on.

A window is a start state, a target state and a deadline. Both ends come from one
contiguous stretch of one rollout, so between them there is a real motion a real robot
performed under physics. The bridge uses that motion as a learning signal.

The mask is what this shows:

    green   context, before the start and after the target. Not part of the training data,
            shown so the window can be read against the motion it was cut out of
    red     the masked window, start state to target state. What the bridge has to perform

Two robots are compiled, one green and one red, and the one not wanted is parked under the
floor. A geom color is baked at compile time and cannot be repainted frame by frame, so
switching color has to be switching robots.

Run

1. Serve the viewer, then open the printed address. Next window draws another from the same
   corpus, duration range and segment index the command term draws from, so what is on
   screen is a sample of the training distribution.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.view

2. Restrict to one clip, and to the side of the split uv run play and evaluate read.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.view \
      --source jumps1_subject1 --split eval
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
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.dataset import (
  DEFAULT_DATASET,
  ROOT_STATE_DIM,
  Dataset,
  load_dataset,
)

OUTSIDE = "context/"
"""Prefix of the green robot, shown before the start and after the target."""

INSIDE = "window/"
"""Prefix of the red robot, shown while the recording is inside the masked window."""

GREEN = (0.35, 0.85, 0.45, 0.7)
RED = (0.95, 0.32, 0.32, 0.9)

UNDERGROUND = np.array([0.0, 0.0, -50.0])
"""Where the copy that is not wanted goes."""

ANY = "any source"


@dataclass
class ViewCfg:
  path: Path = DEFAULT_DATASET
  split: str = "train"
  """Which side of the holdout split to draw from. train is what the bridge learns on,
  eval is what uv run play and evaluate read."""

  source: str = ""
  """Restrict windows to one clip or skill. Empty starts on any of them, and the dropdown
  changes it without restarting."""

  duration_s: tuple[float, float] = (0.3, 1.2)
  """How long a window may be. The default is BridgeCommandCfg.duration_s_range, so what
  is drawn here is what the task draws. Change both together or this stops being a picture
  of the training distribution."""

  context_s: float = 0.6
  """How much recording to play either side of the window. Clipped at the ends of the
  rollout, since context from a different rollout would be a different robot."""

  speed: float = 1.0
  port: int = 8080


##
# The two ghosts.
##


@dataclass
class Slot:
  """Where one robot's numbers live in the shared qpos."""

  free: int
  """qpos address of its free joint. Position is [free, free + 3), orientation is
  [free + 3, free + 7)."""
  joints: np.ndarray
  """(J,) qpos addresses, in model joint order, which is the order the dataset joint block
  was recorded in."""


def build() -> tuple[mujoco.MjModel, Slot, Slot]:
  """One model holding both colored copies and a floor, and where to write each pose."""
  world = mujoco.MjSpec()
  world.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[20.0, 20.0, 0.1],
    rgba=[0.3, 0.3, 0.32, 1.0],
  )
  for prefix in (OUTSIDE, INSIDE):
    # A fresh spec per copy. Attaching one twice asks MuJoCo to adopt the same bodies into
    # two places in the tree
    world.attach(get_spec(), prefix=prefix, frame=world.worldbody.add_frame())
  model = world.compile()

  tint(model, OUTSIDE, GREEN)
  tint(model, INSIDE, RED)
  return model, slot(model, OUTSIDE), slot(model, INSIDE)


def tint(model: mujoco.MjModel, prefix: str, color: tuple[float, ...]) -> None:
  """Paint the visual geoms of one copy and hide its collision ones.

  Collision geoms are the crude convex stand-ins the solver works with, and drawing them
  puts a robot made of boxes inside the robot.

  The material has to be dropped, not merely recolored. A geom color resolves as
  material first and geom_rgba only as a fallback, so a G1 mesh that still carries its
  material comes out in the robot's own color no matter what is written here. That failure
  is not just cosmetic: the renderer batches bodies whose geometry fingerprints match, the
  fingerprint is over type, mesh, material and rgba, and two copies that were never
  actually recolored fingerprint identically. They merge into one mesh, and instead of a
  green robot and a red one there is a single robot in its factory colors.
  """
  for geom in range(model.ngeom):
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom])
    if body is None or not body.startswith(prefix):
      continue
    if model.geom_contype[geom] or model.geom_conaffinity[geom]:
      model.geom_rgba[geom, 3] = 0.0
      continue
    model.geom_matid[geom] = -1
    model.geom_rgba[geom] = color


def slot(model: mujoco.MjModel, prefix: str) -> Slot:
  """Resolve the qpos addresses of one copy, in model joint order.

  Order and not name, because a dataset does not record the joint names it was written
  against. Both copies are the same spec attached twice, so both give the same order, and
  it is the order robot.data.joint_pos was in when the rows were recorded. The count is
  checked against the dataset, which is as much of that assumption as a file can carry.
  """
  free: int | None = None
  joints: list[int] = []
  for joint in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
    if not name.startswith(prefix):
      continue
    address = int(model.jnt_qposadr[joint])
    if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
      free = address
    else:
      joints.append(address)
  if free is None:
    raise SystemExit(f"No free joint under '{prefix}'.")
  return Slot(free=free, joints=np.asarray(joints, dtype=np.int64))


def show(qpos: np.ndarray, where: Slot, state: np.ndarray, shift: np.ndarray) -> None:
  """Write one state into the shared qpos, moved by shift."""
  qpos[where.free : where.free + 3] = state[0:3] + shift
  qpos[where.free + 3 : where.free + 7] = state[3:7]
  qpos[where.joints] = state[ROOT_STATE_DIM : ROOT_STATE_DIM + where.joints.size]


##
# Finding windows.
##


@dataclass
class Window:
  """One drawn window, with its context, ready to play."""

  source: str
  states: np.ndarray
  """(L, 13 + 2J). Everything played, in time order: context, window, context."""
  start: int
  """Index into states of the start state the bridge is teleported onto."""
  stop: int
  """Index of the target state it has to arrive in. Red covers [start, stop]."""
  fps: float

  @property
  def steps(self) -> int:
    return self.stop - self.start

  @property
  def duration_s(self) -> float:
    return self.steps / self.fps


def runs(
  data: Dataset, rows: torch.Tensor
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
  """Rows in time order, and the half open bounds of every contiguous stretch in them.

  The same runs Dataset.segments indexes, found the same way: sort into (trajectory,
  frame) order, then cut wherever the frame does not step by one. Rebuilt here rather than
  read off a Segments, because that holds where a window may open and this needs where the
  rollout around it ends, which is what stops context being borrowed from the next one.
  """
  width = int(data.frame.max().item()) + 1
  order = rows[torch.argsort(data.trajectory[rows] * width + data.frame[rows])]
  trajectory, frame = data.trajectory[order], data.frame[order]
  steps_by_one = (trajectory[1:] == trajectory[:-1]) & (frame[1:] == frame[:-1] + 1)
  edges = (~steps_by_one).nonzero().flatten() + 1
  bounds = [0, *edges.tolist(), int(order.numel())]
  return order, list(zip(bounds[:-1], bounds[1:], strict=True))


def draw(
  data: Dataset,
  order: torch.Tensor,
  spans: list[tuple[int, int]],
  min_steps: int,
  max_steps: int,
  context: int,
  rng: np.random.Generator,
) -> Window:
  """One window, drawn the way the command term draws one, plus context either side.

  The duration is uniform over what the chosen start actually admits and not over the
  configured range, because a start near the end of its rollout only offers short windows.
  That is the command term's rule too, and getting it wrong here would show a distribution
  the policy never sees.
  """
  usable = [(a, b) for a, b in spans if b - a > min_steps]
  if not usable:
    raise SystemExit(
      f"No rollout here is longer than {min_steps} control steps, so no window of "
      f"{min_steps / data.fps:.2f} s can be cut from one."
    )
  a, b = usable[rng.integers(len(usable))]
  start = int(rng.integers(a, b - min_steps))
  steps = int(rng.integers(min_steps, min(max_steps, b - 1 - start) + 1))

  low, high = max(a, start - context), min(b, start + steps + context + 1)
  return Window(
    source=data.names[int(data.skill[order[start]].item())],
    states=data.states[order[low:high]].numpy().astype(np.float64),
    start=start - low,
    stop=start + steps - low,
    fps=data.fps,
  )


##
# Reading one out.
##


def describe(window: Window, at: int, drawn: int) -> str:
  """What the bridge is being asked for, in the units the question is asked in."""
  begin, end = window.states[window.start], window.states[window.stop]
  num_joints = (window.states.shape[1] - ROOT_STATE_DIM) // 2
  joints = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + num_joints)
  travel = float(np.abs(end[joints] - begin[joints]).max())
  return (
    f"| Window | |\n|---|---|\n"
    f"| source | {window.source} |\n"
    f"| duration | {window.duration_s:.2f} s ({window.steps} steps) |\n"
    f"| played | {window.states.shape[0]} frames, red on {window.steps + 1} |\n"
    f"| to travel | {float(np.linalg.norm(end[0:3] - begin[0:3])):.2f} m |\n"
    f"| speed | {float(np.linalg.norm(begin[7:10])):.2f} "
    f"-> {float(np.linalg.norm(end[7:10])):.2f} m/s |\n"
    f"| turn rate | {float(np.linalg.norm(begin[10:13])):.2f} "
    f"-> {float(np.linalg.norm(end[10:13])):.2f} rad/s |\n"
    f"| pelvis | {begin[2]:.2f} -> {end[2]:.2f} m |\n"
    f"| widest joint move | {travel:.2f} rad |\n"
  )


def listing(data: Dataset, min_steps: int) -> str:
  """What is in the corpus, per source, and how much of it can be asked about."""
  lines = [
    f"{data.states.shape[0]} states at {data.fps:.0f} Hz",
    "| source | states  | windows |",
    "|---|---|---|",
  ]
  print(f"[view] {data.states.shape[0]} states at {data.fps:.0f} Hz")
  for name in data.names:
    rows = data.of((name,))
    _, spans = runs(data, rows)
    windows = sum(max(b - a - min_steps, 0) for a, b in spans)
    lines.append(f"| {name} | {rows.numel()} | {windows} |")
    print(
      f"[view]   {name}: {rows.numel()} states, {len(spans)} rollouts, "
      f"{windows} windows of {min_steps} steps"
    )
  return "\n".join(lines)


##
# The viewer.
##


def serve(cfg: ViewCfg) -> None:
  data = load_dataset(cfg.path, "cpu", cfg.split)
  if cfg.source and cfg.source not in data.names:
    raise SystemExit(f"This dataset holds {data.names}, not '{cfg.source}'.")

  min_steps = max(1, int(np.ceil(cfg.duration_s[0] * data.fps)))
  max_steps = max(min_steps, int(np.floor(cfg.duration_s[1] * data.fps)))
  context = max(0, int(round(cfg.context_s * data.fps)))
  rng = np.random.default_rng()

  model, outside, inside = build()
  if outside.joints.size != data.num_joints:
    raise SystemExit(
      f"The dataset holds {data.num_joints}-joint states and this G1 has "
      f"{outside.joints.size}, so it was recorded against a different robot."
    )
  mj_data = mujoco.MjData(model)

  # Indexed once per source. of builds a mask over the whole table, and redoing it per
  # window would scan the corpus every few seconds to draw one pair
  indexed = {name: runs(data, data.of((name,))) for name in data.names}
  indexed[ANY] = runs(data, data.of(None))

  server = viser.ViserServer(port=cfg.port)
  scene = ViserMujocoScene(server, model, num_envs=1)
  # Off, or the parked copy takes the whole scene with it. Camera tracking translates
  # everything by minus the position of the first body that has a joint, which here is the
  # green robot, and it keeps that robot pinned at the origin. So the green one never
  # appears to move, the red one is drawn 50 m in the air the moment green is parked, and
  # the floor grid goes with the shift and leaves a blank background. Every one of those
  # reads as a bug in this file and none of them is
  scene.camera_tracking_enabled = False

  # With tracking off nothing places the camera, and viser's default look is along the
  # floor. A rollout root position has its environment origin subtracted off x and y, so
  # every window starts near here and wanders a metre or two
  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    client.camera.position = (3.0, -3.0, 1.8)
    client.camera.look_at = (0.0, 0.0, 0.8)

  server.gui.add_markdown(listing(data, min_steps))
  source = server.gui.add_dropdown(
    "Source", [ANY, *data.names], initial_value=cfg.source or ANY
  )
  back = server.gui.add_button("Previous")
  forward = server.gui.add_button("Next")
  play = server.gui.add_button("Play")
  stop = server.gui.add_button("Stop")
  reset = server.gui.add_button("Reset")
  cursor = server.gui.add_slider("Frame", 0, 1, 1, 0)
  readout = server.gui.add_markdown("")

  # Every window drawn this session, and where in it we are. Kept so Previous goes back to
  # the window that was just on screen rather than drawing a different one: a window worth
  # a second look is gone the moment Next redraws, and the draw is random, so without this
  # there is no way back to it
  history: list[Window] = []
  at = 0
  playing = True

  def rewind() -> None:
    cursor.value = 0

  def use(index: int) -> None:
    nonlocal at
    at = index
    window = history[at]
    cursor.max = window.states.shape[0] - 1
    rewind()
    readout.content = describe(window, at, len(history))
    back.disabled = at == 0

  def restart() -> None:
    history.clear()
    history.append(
      draw(data, *indexed[source.value], min_steps, max_steps, context, rng)
    )
    use(0)

  @forward.on_click
  def _(_) -> None:
    if at + 1 == len(history):
      history.append(
        draw(data, *indexed[source.value], min_steps, max_steps, context, rng)
      )
    use(at + 1)

  @back.on_click
  def _(_) -> None:
    if at > 0:
      use(at - 1)

  @reset.on_click
  def _(_) -> None:
    rewind()

  @play.on_click
  def _(_) -> None:
    nonlocal playing
    playing = True

  @stop.on_click
  def _(_) -> None:
    nonlocal playing
    playing = False

  @source.on_update
  def _(_) -> None:
    restart()

  restart()
  print(f"[view] serving on http://localhost:{cfg.port}")
  while True:
    window = history[at]
    if playing:
      cursor.value = (int(cursor.value) + 1) % window.states.shape[0]
    frame = min(int(cursor.value), window.states.shape[0] - 1)

    state = window.states[frame]
    masked = window.start <= frame <= window.stop
    show(mj_data.qpos, inside, state, np.zeros(3) if masked else UNDERGROUND)
    show(mj_data.qpos, outside, state, UNDERGROUND if masked else np.zeros(3))
    mujoco.mj_kinematics(model, mj_data)
    scene.update_from_mjdata(mj_data)
    time.sleep(1.0 / (window.fps * max(cfg.speed, 1e-3)))


if __name__ == "__main__":
  serve(tyro.cli(ViewCfg, config=mjlab.TYRO_FLAGS))
