"""Viewer for the windows the bridge is trained on.

A window is a start state, a target state and a deadline. Both ends come from one
contiguous stretch of one clip, and the motion between them is the learning signal.

The mask is what this shows:

    green   context, before the start and after the target. Not part of the training data,
            shown so the window can be read against the motion it was cut out of
    red     the masked window, start state to target state. What the bridge has to perform

Two robots are compiled, one green and one red, and the one not wanted is parked under the
floor. A geom color is baked at compile time and cannot be repainted frame by frame, so
switching color has to be switching robots.

The Corpus dropdown picks what the windows are cut from:

    BABEL, LAFAN  retargeted and filtered clips
    Stitched      two of those joined at a seam, see motion_graph/build.py. Windows are
                  drawn across the seam, as in training, and the readout says where it is

Run

1. Serve the viewer, then open the printed address. Next window draws another from the same
   corpus, duration range and segment index the command term draws from, so what is on
   screen is a sample of the training distribution.

    uv run python -m mjlab.tasks.bridging.bridges.dataset.view

2. Start on one corpus, and on the side of the split play and evaluate read.

    uv run python -m mjlab.tasks.bridging.bridges.dataset.view \
      --corpus Stitched --split eval
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
import tyro
import viser
from mjviser import ViserMujocoScene

import mjlab
from mjlab.tasks.bridging.bridges.dataset.dataset import (
  ROOT_STATE_DIM,
  Dataset,
  crosses_seam,
)
from mjlab.tasks.bridging.bridges.dataset.motion_graph import clips
from mjlab.tasks.bridging.config import get_robot

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
  corpus: str = "BABEL"
  """One of BABEL, LAFAN, Stitched. The dropdown changes it."""
  robot: str = "g1"
  split: str = "train"
  """Which side of the holdout split to draw from. train is what the bridge learns on,
  eval is what uv run play and evaluate read."""

  source: str = ""
  """Restrict windows to one category, clip or skill. Empty starts on any of them, and the
  dropdown changes it without restarting."""

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


def build(robot: str = "g1") -> tuple[mujoco.MjModel, Slot, Slot]:
  """One model holding both colored copies and a floor, and where to write each pose."""
  get_spec = get_robot(robot).get_spec
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
  rows: np.ndarray
  """(L,) dataset row of every played state."""

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
  allowed: np.ndarray,
) -> Window:
  """One window, drawn the way the command term draws one, plus context either side.

  The duration is uniform over what the chosen start actually admits and not over the
  configured range, because a start near the end of its clip only offers short windows.
  That is the command term's rule too, and getting it wrong here would show a distribution
  the policy never sees. allowed says which dataset rows may open a window, see
  crosses_seam.
  """
  usable = [(a, b) for a, b in spans if b - a > min_steps]
  if not usable:
    raise SystemExit(
      f"No clip here is longer than {min_steps} control steps, so no window of "
      f"{min_steps / data.fps:.2f} s can be cut from one."
    )
  while True:
    a, b = usable[rng.integers(len(usable))]
    starts = np.arange(a, b - min_steps)
    starts = starts[allowed[order[starts].numpy()]]
    if len(starts):
      break
  start = int(rng.choice(starts))
  steps = int(rng.integers(min_steps, min(max_steps, b - 1 - start) + 1))

  low, high = max(a, start - context), min(b, start + steps + context + 1)
  return Window(
    source=data.names[int(data.skill[order[start]].item())],
    states=data.states[order[low:high]].numpy().astype(np.float64),
    start=start - low,
    stop=start + steps - low,
    fps=data.fps,
    rows=order[low:high].numpy(),
  )


##
# Reading one out.
##


def describe(window: Window, note: str = "") -> str:
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
    f"{note}"
  )


def seams(window: Window, data: Dataset, stitched: list[clips.Clip]) -> str:
  """Readout rows for a stitched clip: its pieces and the seams inside the window."""
  if not stitched:
    return ""
  clip = stitched[int(data.trajectory[window.rows[window.start]])]
  frames = data.frame[window.rows].numpy()
  begin, end = frames[window.start], frames[window.stop]
  seam = int(clip.seams[0])
  where = (
    f"{(seam - begin) / window.fps:.2f} s into the window"
    if begin < seam <= end
    else "outside the window, a bug"
  )
  return f"| pieces | {' / '.join(clip.pieces)} |\n| seam | {where} |\n"


def kinematic(corpus: str, robot: str, split: str) -> tuple[Dataset, list[clips.Clip]]:
  """Retargeted or stitched clips as a Dataset, one trajectory per clip."""
  loaded = clips.load(
    clips.corpus_files(corpus, robot, "val" if split == "eval" else "train")
  )
  names = tuple(sorted({c.category for c in loaded.clips}))
  states, skill, trajectory, frame, seam = [], [], [], [], []
  for index, c in enumerate(loaded.clips):
    kept = np.flatnonzero(c.valid)
    states.append(c.states[kept])
    skill.append(np.full(len(kept), names.index(c.category)))
    trajectory.append(np.full(len(kept), index))
    frame.append(kept)
    seam.append(np.full(len(kept), c.seams[0] if len(c.seams) else -1))

  def column(values: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.concatenate(values)).long()

  data = Dataset(
    states=torch.from_numpy(np.concatenate(states)),
    skill=column(skill),
    trajectory=column(trajectory),
    frame=column(frame),
    names=names,
    fps=loaded.fps,
    seam=column(seam),
  )
  return data, loaded.clips if corpus == "Stitched" else []


def listing(data: Dataset, min_steps: int, allowed: np.ndarray) -> str:
  """What is in the corpus, per source, and how much of it can be asked about."""
  lines = [
    f"{data.states.shape[0]} states at {data.fps:.0f} Hz",
    "| source | states  | windows |",
    "|---|---|---|",
  ]
  print(f"[view] {data.states.shape[0]} states at {data.fps:.0f} Hz")
  for name in data.names:
    rows = data.of((name,))
    order, spans = runs(data, rows)
    windows = sum(
      int(allowed[order[a : b - min_steps].numpy()].sum()) for a, b in spans
    )
    lines.append(f"| {name} | {rows.numel()} | {windows} |")
    print(
      f"[view]   {name}: {rows.numel()} states, {len(spans)} runs, "
      f"{windows} windows of {min_steps} steps"
    )
  return "\n".join(lines)


##
# The viewer.
##


@dataclass
class Loaded:
  """One corpus, indexed for drawing."""

  data: Dataset
  indexed: dict[str, tuple[torch.Tensor, list[tuple[int, int]]]]
  stitched: list[clips.Clip]
  allowed: np.ndarray
  """(N,) rows that may open a window."""
  listing: str
  min_steps: int
  max_steps: int
  context: int


def open_corpus(name: str, cfg: ViewCfg, joints: int) -> Loaded:
  data, stitched = kinematic(name, cfg.robot, cfg.split)
  if joints != data.num_joints:
    raise SystemExit(
      f"{name} holds {data.num_joints}-joint states and {cfg.robot} has {joints}, "
      "so it was recorded against a different robot."
    )
  min_steps = max(1, int(np.ceil(cfg.duration_s[0] * data.fps)))
  assert data.seam is not None
  allowed = crosses_seam(data.seam, data.frame, 0, min_steps).numpy()
  # Indexed once per source. of builds a mask over the whole table, and redoing it per
  # window would scan the corpus every few seconds to draw one pair
  indexed = {n: runs(data, data.of((n,))) for n in data.names}
  indexed[ANY] = runs(data, data.of(None))
  return Loaded(
    data=data,
    indexed=indexed,
    stitched=stitched,
    allowed=allowed,
    listing=listing(data, min_steps, allowed),
    min_steps=min_steps,
    max_steps=max(min_steps, int(np.floor(cfg.duration_s[1] * data.fps))),
    context=max(0, int(round(cfg.context_s * data.fps))),
  )


def serve(cfg: ViewCfg) -> None:
  if cfg.corpus not in clips.CORPORA:
    raise SystemExit(f"corpus is one of {clips.CORPORA}, not '{cfg.corpus}'.")
  rng = np.random.default_rng()
  model, outside, inside = build(cfg.robot)
  mj_data = mujoco.MjData(model)

  # Loaded on first use, kept for the session
  cache: dict[str, Loaded] = {}

  def corpus_named(name: str) -> Loaded:
    if name not in cache:
      cache[name] = open_corpus(name, cfg, outside.joints.size)
    return cache[name]

  current = corpus_named(cfg.corpus)
  if cfg.source and cfg.source not in current.data.names:
    raise SystemExit(f"{cfg.corpus} holds {current.data.names}, not '{cfg.source}'.")

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

  summary = server.gui.add_markdown(current.listing)
  corpus = server.gui.add_dropdown("Corpus", clips.CORPORA, initial_value=cfg.corpus)
  source = server.gui.add_dropdown(
    "Source", [ANY, *current.data.names], initial_value=cfg.source or ANY
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
    readout.content = describe(window, seams(window, current.data, current.stitched))
    back.disabled = at == 0

  def fresh() -> Window:
    c = current
    spans = c.indexed.get(source.value, c.indexed[ANY])
    return draw(c.data, *spans, c.min_steps, c.max_steps, c.context, rng, c.allowed)

  def restart() -> None:
    history[:] = [fresh()]
    use(0)

  @forward.on_click
  def _(_) -> None:
    if at + 1 == len(history):
      history.append(fresh())
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

  @corpus.on_update
  def _(_) -> None:
    nonlocal current
    current = corpus_named(corpus.value)
    summary.content = current.listing
    source.options = [ANY, *current.data.names]
    source.value = ANY
    restart()

  restart()
  print(f"[view] serving on http://localhost:{cfg.port}")
  while True:
    # The dropdowns redraw from another thread, so at can briefly run past history
    window = history[min(at, len(history) - 1)]
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
