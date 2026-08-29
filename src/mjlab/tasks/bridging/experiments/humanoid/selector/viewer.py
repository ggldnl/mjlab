"""Look at what the profiler picked.

Run:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.viewer \
      --skill jump

One ghost, replaying the run the selected candidate came out of. Blue for ordinary
execution, red on the frames where the robot is close enough to that candidate to count as
being in it. So the picture answers "when, in the middle of doing this, is the robot in
that state", which is what the bridge will be aiming at.

Close enough is a distance. Frames are compared in the same standardized feature space the
clustering used, every feature divided by its own spread across the recording, and
`epsilon` is that distance. One is a sensible default: it is near the mean spread of a
cluster about its own centre, so a frame inside epsilon is one the profiler would have put
in the same group. Turn it down until the red flickers on for a moment and you are looking
at the exact instant; turn it up to see how wide a window the bridge would have to hit.

The slider walks one goal band's candidates in increasing score, so dragging right moves
toward the states the skill goes to more often, more safely and more repeatably.

No physics runs. The robot is posed by writing qpos and running forward kinematics, and the
copy that is not wanted is parked under the floor, since a mesh's colour is baked when the
scene is built and cannot be repainted frame by frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro
import viser
from mjviser import ViserMujocoScene

import mjlab
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.dataset import (
  ROOT_STATE_DIM,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.profile import PROFILE_ROOT

NORMAL = "normal/"
"""Prefix of the blue robot, shown while the run is inside no candidate."""

PICKED = "picked/"
"""Prefix of the red robot, shown while the run is inside a candidate."""

BLUE = (0.35, 0.55, 0.95, 0.75)
RED = (0.95, 0.32, 0.32, 0.9)

UNDERGROUND = np.array([0.0, 0.0, -50.0])
"""Where the copy that is not wanted goes."""


@dataclass
class ViewerCfg:
  skill: str = "walk"
  """Which profile to look at. One npz per skill, written by profile.py."""

  root: Path = PROFILE_ROOT
  epsilon: float = 1.0
  """How near a frame has to be to the candidate to turn red, in feature spreads."""

  port: int = 8080


@dataclass
class Slot:
  """Where one robot's numbers live in the shared qpos."""

  free: int
  """qpos address of the free joint. Position is [free, free + 3), orientation is
  [free + 3, free + 7)."""
  joints: np.ndarray
  """(J,) qpos addresses, in the profile's own joint order."""


def build(joint_names: list[str]) -> tuple[mujoco.MjModel, Slot, Slot]:
  """One model holding both coloured copies and a floor, and where to write each pose."""
  world = mujoco.MjSpec()
  world.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[20.0, 20.0, 0.1],
    rgba=[0.3, 0.3, 0.32, 1.0],
  )
  for prefix in (NORMAL, PICKED):
    # A fresh spec per copy. Attaching one twice asks MuJoCo to adopt the same bodies into
    # two places in the tree
    world.attach(get_spec(), prefix=prefix, frame=world.worldbody.add_frame())
  model = world.compile()

  tint(model, NORMAL, BLUE)
  tint(model, PICKED, RED)
  return model, slot(model, NORMAL, joint_names), slot(model, PICKED, joint_names)


def tint(model: mujoco.MjModel, prefix: str, color: tuple[float, ...]) -> None:
  """Paint one copy's visual geoms and hide its collision ones.

  Collision geoms are the crude convex stand-ins the solver works with, and drawing them
  puts a robot made of boxes inside the robot.
  """
  for geom in range(model.ngeom):
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom])
    if body is None or not body.startswith(prefix):
      continue
    if model.geom_contype[geom] or model.geom_conaffinity[geom]:
      model.geom_rgba[geom, 3] = 0.0
    else:
      model.geom_rgba[geom] = color


def slot(model: mujoco.MjModel, prefix: str, joint_names: list[str]) -> Slot:
  """Resolve one copy's qpos addresses, by name, in the profile's own joint order."""
  free = None
  for joint in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE and name.startswith(prefix):
      free = int(model.jnt_qposadr[joint])
      break
  if free is None:
    raise SystemExit(f"No free joint under '{prefix}'.")

  addresses = []
  for name in joint_names:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
    if joint < 0:
      raise SystemExit(
        f"The profile names a joint '{name}' this G1 does not have. It was recorded "
        f"against a different robot."
      )
    addresses.append(int(model.jnt_qposadr[joint]))
  return Slot(free=free, joints=np.asarray(addresses, dtype=np.int64))


def show(qpos: np.ndarray, where: Slot, state: np.ndarray, shift: np.ndarray) -> None:
  """Write one state into the shared qpos, moved by `shift`."""
  qpos[where.free : where.free + 3] = state[0:3] + shift
  qpos[where.free + 3 : where.free + 7] = state[3:7]
  qpos[where.joints] = state[ROOT_STATE_DIM : ROOT_STATE_DIM + where.joints.size]


def distances(profile) -> np.ndarray:
  """(K, L). How far every frame of each stored run is from its own candidate.

  In the standardized space the clustering worked in, so this reads on the same scale as a
  cluster's tightness and the two can be compared by eye.
  """
  mean, spread = profile["feat_mean"], profile["feat_std"]
  runs = (profile["play_feats"] - mean) / spread
  picks = (profile["feats"] - mean) / spread
  return np.linalg.norm(runs - picks[:, None, :], axis=-1)


def runs_of(matched: np.ndarray) -> str:
  """Contiguous stretches of True, as "12-19, 88-95"."""
  edges = np.flatnonzero(np.diff(np.concatenate(([0], matched.view(np.int8), [0]))))
  spans = [
    f"{a}-{b - 1}" if b - 1 > a else f"{a}"
    for a, b in zip(edges[::2], edges[1::2], strict=True)
  ]
  return ", ".join(spans[:6]) + (" ..." if len(spans) > 6 else "")


def describe(profile, pick: int, gap: np.ndarray, epsilon: float, here: float) -> str:
  """The readout: what this candidate scored, and where in the run it happens."""
  length = int(profile["play_len"][pick])
  matched = gap[pick, :length] < epsilon
  rows = "\n".join(
    f"| {name} | {value:+.3f} |"
    for name, value in zip(
      profile["feature_names"], profile["feats"][pick], strict=True
    )
  )
  return (
    f"**score {profile['score'][pick]:.3f}**  ·  stage {profile['stage'][pick]}\n\n"
    f"| | |\n|---|---|\n"
    f"| runs passing through | {profile['coverage'][pick]:.0%} |\n"
    f"| of those, finished up | {profile['clean'][pick]:.0%} |\n"
    f"| tightness | {profile['tightness'][pick]:.2f} |\n"
    f"| distance now | {here:.2f} |\n"
    f"| red on | {int(matched.sum())} of {length} frames |\n"
    f"| at | {runs_of(matched) or 'nowhere'} |\n\n"
    f"| feature | |\n|---|---|\n{rows}\n"
  )


def label(profile, band: int) -> str:
  parts = ", ".join(
    f"{name}={value:+.2f}"
    for name, value in zip(
      profile["goal_labels"], profile["band_goal"][band], strict=True
    )
  )
  return f"{profile['key_label']}={profile['band_key'][band]:+.2f}  ({parts})"


def serve(cfg: ViewerCfg) -> None:
  path = cfg.root / f"{cfg.skill}.npz"
  if not path.exists():
    raise SystemExit(
      f"No profile at {path}. Build it with "
      f"`uv run python -m ...selector.profile --skills \"('{cfg.skill}',)\"`."
    )
  profile = np.load(path)
  if profile["score"].size == 0:
    raise SystemExit(f"{path} holds no candidates.")

  joint_names = [str(name) for name in profile["joint_names"]]
  model, normal, picked = build(joint_names)
  data = mujoco.MjData(model)
  gap = distances(profile)

  bands = sorted(set(int(b) for b in profile["band"]))
  labels = [label(profile, b) for b in bands]
  # Ascending score within a band, which is the order profile.py appended them in
  per_band = {b: np.flatnonzero(profile["band"] == b) for b in bands}
  fps = float(profile["fps"])

  server = viser.ViserServer(port=cfg.port)
  scene = ViserMujocoScene(server, model, num_envs=1)

  top = len(per_band[bands[0]]) - 1
  server.gui.add_markdown(
    f"### {cfg.skill}\n{len(profile['score'])} candidates over {len(bands)} bands"
  )
  goal = server.gui.add_dropdown("Goal", labels)
  rank = server.gui.add_slider("Candidate", 0, max(top, 1), 1, top)
  epsilon = server.gui.add_number("Epsilon", cfg.epsilon, min=0.05, max=8.0, step=0.05)
  readout = server.gui.add_markdown("")
  playing = server.gui.add_checkbox("Play", True)
  frame = server.gui.add_slider("Frame", 0, int(profile["play_len"].max()) - 1, 1, 0)
  goto = server.gui.add_button("Go to the picked frame")

  @goal.on_update
  def _(_) -> None:
    here = per_band[bands[labels.index(goal.value)]]
    rank.max = max(len(here) - 1, 1)
    rank.value = len(here) - 1
    frame.value = 0

  @goto.on_click
  def _(_) -> None:
    band = bands[labels.index(goal.value)]
    here = per_band[band]
    playing.value = False
    frame.value = int(profile["play_at"][here[min(int(rank.value), len(here) - 1)]])

  shown = (-1, -1.0)
  while True:
    band = bands[labels.index(goal.value)]
    here = per_band[band]
    chosen = int(here[min(int(rank.value), len(here) - 1)])

    length = max(int(profile["play_len"][chosen]), 1)
    if playing.value:
      frame.value = (int(frame.value) + 1) % length
    cursor = min(int(frame.value), length - 1)

    state = profile["play"][chosen, cursor]
    near = float(gap[chosen, cursor])
    inside = near < float(epsilon.value)
    show(data.qpos, picked, state, np.zeros(3) if inside else UNDERGROUND)
    show(data.qpos, normal, state, UNDERGROUND if inside else np.zeros(3))
    mujoco.mj_kinematics(model, data)
    scene.update_from_mjdata(data)

    # Only when it would say something different. The readout costs a full markdown parse
    # on every client and the distance moves in the third decimal place most frames
    if (chosen, round(near, 2)) != shown:
      readout.content = describe(profile, chosen, gap, float(epsilon.value), near)
      shown = (chosen, round(near, 2))
    time.sleep(1.0 / fps)


if __name__ == "__main__":
  serve(tyro.cli(ViewerCfg, config=mjlab.TYRO_FLAGS))
