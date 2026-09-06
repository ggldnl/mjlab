"""Convert a PAiD football kick into an mjlab motion clip.

PAiD (Kong et al., "Learning Soccer Skills for Humanoid Robots: A Progressive
Perception-Action Framework", arXiv 2602.05310, CC BY-NC 4.0) publishes the kick motions it
trained a G1 on: thirteen npz files, ten plain and three stylized, each a human kick already
retargeted to the 29 joint G1 and labelled with the striking leg. That is the same robot
this repo runs, so nothing here retargets anything. The clips are downloaded, permuted into
mjlab's joint order, and replayed.

The permutation is the one thing worth explaining. PAiD logs the state of an Isaac Lab
articulation, whose joint order is breadth first over the kinematic tree, while mjlab's G1
lists joints limb by limb. ISAAC_JOINT_ORDER below is that breadth first order, and it was
not guessed: driving mjlab's G1 with it and comparing the resulting body positions against
the ones stored in the clip gives a maximum error of 0.00 mm over all thirty bodies. The two
models are the same robot and the files differ only by a column permutation.

What happens to a clip, in order:

    0. Download it from the PAiD repository, unless it is cached.
    1. Read the root pose out of body 0 and scatter the joint columns into mjlab's order.
    2. Rotate and translate so the clip starts at the origin facing +x.
    3. Resample to the control rate and hold the first frame still for half a second.
    4. Shift the root vertically so a planted foot sits at standing foot height.
    5. Replay through MuJoCo to log every body world pose and velocity.
    6. Find the strike, and search for a ball position only the striking foot can reach.

Step 6 is what makes this converter different from the martial one, and it is a search
rather than a formula. The environment never looks for the ball: the position is decided
here, stored in the clip, and read back by kick_env_cfg.py's reset event.

It has to be a search because the obvious formula does not work. Putting the ball one radius
ahead of the striking ankle looks right and leaves the support foot 4 cm of room in the
narrower clips, which the support foot then spends: it plants on the ball during the walk in
and knocks it away before the swing ever arrives. So where the ball goes is not decided by
the striking foot alone. search_ball scores every position near the strike on how hard the
sole closes on the ball, subject to the support foot staying clear of it at every frame, and
takes the best. Measured over the thirteen published clips the room available ranges from
4.5 cm to 19 cm depending on how wide the subject plants, so which clip is converted matters
as much as where the ball is put in it.

Run

1. Convert the default clip. Downloads it on first use.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.kick.dataset

2. Convert a different one of the thirteen, without adding it to MOTIONS.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.kick.dataset --motion kick --clip-source soccer-standard-005_right

3. Convert only part of a clip, when the approach is longer than the task needs.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.kick.dataset --motion kick --clip-start 40 --clip-end 200

4. Disagree with where the search put the ball. Open the viewer, move it with the two Ball
   sliders until it looks right, read the position off the Clip.ball box, write it into
   MOTIONS as Clip(..., ball=(x, y)) and convert again. The search still runs and still
   refuses a position the support foot would tread on, so a pinned position is checked
   rather than trusted.

    uv run play Mjlab-G1-Kick
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import tyro

import mjlab
from mjlab.asset_zoo.objects.ball import BALL_RADIUS
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.dataset import (
  FOOT_BODY_NAMES,
  MAX_GROUND_PENETRATION,
  RawMotion,
  describe_clip,
  replay,
  resample,
  stance_baseline,
  standing_foot_height,
  velocities,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.martial.dataset import (
  canonicalize,
  prepend_hold,
)
from mjlab.terrains import TerrainEntityCfg

# Joint order of the columns in a PAiD clip. Isaac Lab orders an articulation's joints by
# depth in the kinematic tree, so the two legs and the waist interleave rather than running
# limb by limb the way the Unitree CSVs and mjlab's model do. Verified rather than assumed,
# see the module docstring
ISAAC_JOINT_ORDER: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
  "waist_yaw_joint",
  "left_hip_roll_joint",
  "right_hip_roll_joint",
  "waist_roll_joint",
  "left_hip_yaw_joint",
  "right_hip_yaw_joint",
  "waist_pitch_joint",
  "left_knee_joint",
  "right_knee_joint",
  "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint",
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_shoulder_roll_joint",
  "right_shoulder_roll_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
  "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_elbow_joint",
  "left_wrist_roll_joint",
  "right_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_yaw_joint",
)

SOURCE_DIR = Path("data") / "paid"
"""Where the published clips are cached. Not a LAFAN1 directory: nothing here is cropped
out of a LAFAN1 performance."""

CLIP_DIR = SOURCE_DIR / "clips"
"""Where the converter puts its output, one directory per motion."""

SOURCE_URL = "https://raw.githubusercontent.com/TeleHuman/HumanoidSoccer/main/motions/{group}/{name}.npz"

# The two directories the clips are published in, tried in order. Which one a clip is in is
# recoverable from its name, but a lookup table of thirteen names is a table to maintain
SOURCE_GROUPS: tuple[str, ...] = ("soccer-standard", "soccer-stylized")

# How long the reference stands still before the clip starts. The published clips open with
# the subject settled, so this is not repairing anything: it guarantees that frame zero is a
# standstill, which is what a reset writes into the robot and what the stance baseline in
# convert_clip is measured over
STILL_HOLD_S = 0.5

##
# Where the ball goes. See search_ball, which is what these are for.
##

# Radius of the sole capsules the G1's feet collide through
CAPSULE_RADIUS = 0.01

# Frames either side of the strike within which contact has to happen
STRIKE_WINDOW = (-12, 3)

# How far around the strike point ball positions are searched, and how finely, in metres.
# A ball the swing can reach at all is within a stride of where the foot is at the strike
SEARCH_REACH = 0.35
SEARCH_STEP = 0.02

# How much room the striking foot must still have from the ball for an episode to be allowed
# to start at that frame, in metres
RESET_CLEARANCE = 0.10

# How much room the support foot must have from the ball, at every frame and every corner of
# the scatter, in metres. The clip that failed had 0.037 m and its support foot trod on the
# ball in a third of episodes, so this is that with room to spare. Raising it costs strike
# quality: the further the ball sits from the support foot, the more glancing the blow
MIN_CLEARANCE = 0.05

# How far the reset event may scatter the ball, in metres. Must match BALL_FORWARD_RANGE and
# BALL_LATERAL_RANGE in kick_env_cfg.py: the position search_ball picks is the one that
# survives this much scatter, and widening it there without reconverting invalidates that
BALL_SCATTER = 0.08


def motion_dir(name: str) -> Path:
  """Where one motion's npz and manifest live."""
  return CLIP_DIR / name


@dataclass(frozen=True)
class Clip:
  """One motion, as a published clip and an optional frame window into it.

  Frames are 0-indexed and end is exclusive, matching numpy. None takes the whole clip,
  which is the default: these files are already cropped to a single kick.
  """

  source: str
  start: int | None = None
  end: int | None = None
  ball: tuple[float, float] | None = None
  """Ball position in the clip's frame, overriding search_ball's choice.

  For a position found by eye with the viewer's ball sliders, which report exactly this.
  The search still runs, so its clearance is still measured and still refuses a position the
  support foot would tread on."""


MOTIONS: dict[str, Clip] = {
  # Picked on measurements, not on taste. It runs in: 1.05 m of approach at 1.98 m/s, two
  # strides rather than a step, which is the longest run-up of any plain clip. And it is the
  # clip a ball fits in best. Measured over all thirteen (see search_ball), the position that
  # survives the spawn scatter leaves the support foot 0.160 m of room here against 0.088 m
  # in soccer-standard-001_right, while still meeting the striking foot at 5.7 m/s. Those two
  # go together: the clips that walk in slowly also plant their feet close, and there is then
  # nowhere to put a 0.22 m ball that the support foot does not tread on first
  "kick": Clip("soccer-standard-006_right"),
}
"""Motion name to the clip it comes from.

Add a motion by adding a line here. It gets a task, its own clip directory and a g1_<name>
log directory. The other twelve published clips are named soccer-standard-0NN_<leg> and
football_stylized-00N_right.
"""


def fetch(name: str, output_dir: Path) -> Path:
  """Download one published clip, unless it is cached. Signature matches lafan1.fetch."""
  destination = output_dir / f"{name}.npz"
  if destination.exists():
    return destination

  output_dir.mkdir(parents=True, exist_ok=True)
  print(f"  downloading {name}.npz")

  # Written to a temporary name first, so an interrupted download cannot leave a truncated
  # file behind that later runs treat as cached
  temporary = destination.with_suffix(".part")
  for group in SOURCE_GROUPS:
    try:
      urllib.request.urlretrieve(SOURCE_URL.format(group=group, name=name), temporary)
    except OSError:
      continue
    temporary.replace(destination)
    return destination

  temporary.unlink(missing_ok=True)
  raise SystemExit(
    f"No clip named '{name}' in the PAiD repository. Published names are "
    "soccer-standard-0NN_left, soccer-standard-0NN_right and football_stylized-00N_right."
  )


def load_clip(
  path: Path, clip: Clip, joint_names: list[str], device: str
) -> tuple[RawMotion, str]:
  """Read a published clip into mjlab's joint order, and say which leg strikes.

  The root pose comes out of body 0, which is the pelvis and the articulation root in both
  models. Everything else in the file is recomputed downstream: the body states are Isaac
  Lab's and are replaced by mjlab's own replay, and the velocities are finite differenced
  again after the resample.
  """
  data = np.load(path)
  window = slice(clip.start, clip.end)

  root_pos = torch.tensor(
    data["body_pos_w"][window, 0], dtype=torch.float32, device=device
  )
  root_quat = torch.tensor(
    data["body_quat_w"][window, 0], dtype=torch.float32, device=device
  )
  root_quat = root_quat / root_quat.norm(dim=-1, keepdim=True)

  columns = torch.tensor(data["joint_pos"][window], dtype=torch.float32, device=device)
  if columns.shape[1] != len(ISAAC_JOINT_ORDER):
    raise ValueError(
      f"{path.name}: expected {len(ISAAC_JOINT_ORDER)} joints, got {columns.shape[1]}"
    )

  joint_pos = torch.zeros(
    columns.shape[0], len(joint_names), dtype=torch.float32, device=device
  )
  for column, name in enumerate(ISAAC_JOINT_ORDER):
    if name not in joint_names:
      raise ValueError(f"Joint '{name}' is missing from the mjlab G1 model")
    joint_pos[:, joint_names.index(name)] = columns[:, column]

  leg = str(data["kick_leg"]) if "kick_leg" in data.files else "right"
  fps = float(np.asarray(data["fps"]).reshape(-1)[0])
  motion = RawMotion(
    root_pos=root_pos, root_quat=root_quat, joint_pos=joint_pos, fps=fps
  )
  return motion, leg


def foot_capsules(
  joint_names: list[str], log: dict[str, np.ndarray], origin: np.ndarray
) -> dict[str, np.ndarray]:
  """Sole capsule endpoints per frame, per foot, in the clip's frame.

  The G1's foot collides through seven capsules along the sole and nothing else. There is no
  instep or toe box above them, so as far as a ball is concerned a foot is a flat plate
  0.025 m under the ankle, and every question about where the ball can go is a question
  about the volume those plates sweep.

  Forward kinematics on a bare robot model rather than the converter's scene, so the geom
  names are the robot's own and nothing here depends on how a scene was assembled.
  """
  model = get_g1_robot_cfg().spec_fn().compile()
  data = mujoco.MjData(model)
  address = {
    name: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
    for name in joint_names
  }
  geoms = {
    side: [
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot{i}_collision")
      for i in range(1, 8)
    ]
    for side in ("left", "right")
  }

  frames = log["joint_pos"].shape[0]
  ends = {side: np.zeros((frames, 7, 2, 3)) for side in geoms}
  for step in range(frames):
    data.qpos[:] = 0.0
    data.qpos[0:3] = log["body_pos_w"][step, 0] - origin
    data.qpos[3:7] = log["body_quat_w"][step, 0]
    for column, name in enumerate(joint_names):
      data.qpos[address[name]] = log["joint_pos"][step, column]
    mujoco.mj_kinematics(model, data)
    for side, ids in geoms.items():
      for i, geom in enumerate(ids):
        axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
        half = model.geom_size[geom][1]
        ends[side][step, i, 0] = data.geom_xpos[geom] - half * axis
        ends[side][step, i, 1] = data.geom_xpos[geom] + half * axis
  return ends


def surface_gap(capsules: np.ndarray, balls: np.ndarray) -> np.ndarray:
  """Distance from each ball's surface to a foot's, per candidate and frame.

  capsules is (T, 7, 2, 3), balls is (C, 3), the result is (C, T). Negative is overlap.
  Accumulated one capsule at a time rather than broadcast over all seven at once, which
  would allocate a candidate by frame by capsule by axis array for no gain.
  """
  gap = np.full((balls.shape[0], capsules.shape[0]), np.inf)
  for i in range(capsules.shape[1]):
    a = capsules[:, i, 0]
    ab = capsules[:, i, 1] - a
    denom = np.maximum(np.sum(ab * ab, axis=-1), 1e-9)
    offset = balls[:, None, :] - a[None]
    along = np.clip(np.sum(offset * ab[None], axis=-1) / denom[None], 0.0, 1.0)
    closest = a[None] + along[..., None] * ab[None]
    distance = np.linalg.norm(balls[:, None, :] - closest, axis=-1)
    gap = np.minimum(gap, distance - CAPSULE_RADIUS - BALL_RADIUS)
  return gap


def approach_speed(
  capsules: np.ndarray,
  log: dict[str, np.ndarray],
  foot_id: int,
  origin: np.ndarray,
  balls: np.ndarray,
  frames: np.ndarray,
) -> np.ndarray:
  """How fast the sole is closing on the ball along the contact normal, per candidate.

  This is what decides the strike. A foot travelling at 6 m/s that passes the ball
  tangentially delivers almost nothing, and foot speed alone cannot tell the two apart. The
  sole is part of a rigid body, so the velocity of the point that touches is the ankle's plus
  the foot's rotation about it, which is why both are read out of the replay.
  """
  a = capsules[frames, :, 0]  # (C, 7, 3)
  b = capsules[frames, :, 1]
  ab = b - a
  denom = np.maximum(np.sum(ab * ab, axis=-1), 1e-9)
  offset = balls[:, None, :] - a
  along = np.clip(np.sum(offset * ab, axis=-1) / denom, 0.0, 1.0)
  closest = a + along[..., None] * ab
  nearest = np.argmin(np.linalg.norm(balls[:, None, :] - closest, axis=-1), axis=-1)
  point = closest[np.arange(len(balls)), nearest]

  normal = balls - point
  normal = normal / np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9)

  lever = point - (log["body_pos_w"][frames, foot_id] - origin)
  velocity = log["body_lin_vel_w"][frames, foot_id] + np.cross(
    log["body_ang_vel_w"][frames, foot_id], lever
  )
  return np.sum(velocity * normal, axis=-1)


def search_ball(
  joint_names: list[str],
  log: dict[str, np.ndarray],
  leg: str,
  foot_id: int,
  fps: float,
  origin: np.ndarray,
  scatter: float = BALL_SCATTER,
  override: tuple[float, float] | None = None,
) -> dict[str, Any]:
  """Find the strike, and the ball position that only the striking foot can reach.

  The strike is the frame the striking foot is moving fastest horizontally. Nothing subtler
  is needed: a kick has exactly one such moment, and every published clip peaks between 5
  and 8 m/s, an order of magnitude above what the same foot reaches walking in.

  Where the ball goes is searched rather than derived, because deriving it does not work.
  Putting the ball one radius ahead of the striking ankle is the obvious rule and it leaves
  the support foot with 4 cm of room in the narrower clips, which the support foot then
  spends: it plants on the ball during the walk in and knocks it away before the swing
  arrives. The measurement that matters is not where the striking foot is, it is how much
  room the other one has.

  So every ball position near the strike is scored on how fast the sole closes on it, which
  is what actually decides a strike, subject to three constraints. The first two are checked
  at every corner of the scatter the reset event applies, not at the nominal position alone:

      the striking foot must reach it inside STRIKE_WINDOW, so contact happens during the
      swing. Without this the search walks the ball forward to where the foot brushes it
      while landing, which clears the support foot perfectly and is not a kick
      the support foot must never reach it, at any frame of the clip
      it must keep MIN_CLEARANCE of room from the support foot, so the margin survives
      contact solver noise as well as the scatter

  reset_limit is the last frame an episode may start at. Past it the striking foot is
  already inside RESET_CLEARANCE of the ball, so a reset there writes the robot into the
  ball rather than in front of it: contact happens because of the teleport and the strike
  terms pay for nothing. See mdp.KickCommand, which caps its sampling with this.
  """
  foot_pos = log["body_pos_w"][:, foot_id] - origin
  velocity = np.gradient(foot_pos, 1.0 / fps, axis=0)
  speed = np.linalg.norm(velocity[:, :2], axis=-1)
  step = int(speed.argmax())
  direction = velocity[step, :2] / max(float(speed[step]), 1e-6)

  frames = foot_pos.shape[0]
  low = max(step + STRIKE_WINDOW[0], 0)
  high = min(step + STRIKE_WINDOW[1], frames - 1)

  # Candidates on the ground around where the striking foot is at the strike. A ball the
  # swing can reach at all is within a stride of that point
  axis = np.arange(-SEARCH_REACH, SEARCH_REACH + 1e-9, SEARCH_STEP)
  grid = np.stack(
    [
      np.repeat(foot_pos[step, 0] + axis, len(axis)),
      np.tile(foot_pos[step, 1] + axis, len(axis)),
      np.full(len(axis) ** 2, BALL_RADIUS),
    ],
    axis=-1,
  )

  capsules = foot_capsules(joint_names, log, origin)
  support = "left" if leg == "right" else "right"

  corners = np.array(
    [
      [dx, dy, 0.0]
      for dx in (-scatter, 0.0, scatter)
      for dy in (-scatter, 0.0, scatter)
    ]
  )
  viable = np.ones(len(grid), dtype=bool)
  room = np.full(len(grid), np.inf)
  for corner in corners:
    kick_gap = surface_gap(capsules[leg], grid + corner)
    support_gap = surface_gap(capsules[support], grid + corner)
    touched = kick_gap <= 0.0
    first = np.where(touched.any(1), touched.argmax(1), frames)
    viable &= touched.any(1) & (first >= low) & (first <= high)
    room = np.minimum(room, support_gap.min(axis=1))

  pinned: int | None = None
  if override is not None:
    # A hand-picked position still has to pass the two constraints, it just does not have to
    # win the search. Snapped to the nearest candidate so everything below reads off the
    # same arrays, which costs at most half a grid step
    grid = np.concatenate([grid, [[override[0], override[1], BALL_RADIUS]]])
    pinned = len(grid) - 1
    for corner in corners:
      kick_gap = surface_gap(capsules[leg], grid[pinned : pinned + 1] + corner)
      support_gap = surface_gap(capsules[support], grid[pinned : pinned + 1] + corner)
      touched = kick_gap <= 0.0
      first = int(touched.argmax()) if touched.any() else frames
      viable = np.append(viable, touched.any() and low <= first <= high)
      room = np.append(room, support_gap.min())
    # The appends above added one entry per corner, so keep the worst of them
    worst_room = float(room[pinned:].min())
    all_viable = bool(viable[pinned:].all())
    viable = np.append(viable[:pinned], all_viable)
    room = np.append(room[:pinned], worst_room)
    if not all_viable or worst_room < MIN_CLEARANCE:
      raise SystemExit(
        f"The ball position {override} is not usable in this clip: "
        + (
          "the striking foot does not reach it during the swing"
          if not all_viable
          else f"the support foot comes within {worst_room:.3f} m of it, under the "
          f"{MIN_CLEARANCE:.2f} m this task asks for"
        )
      )

  allowed = viable & (room >= MIN_CLEARANCE)
  if not allowed.any():
    raise SystemExit(
      "No ball position in this clip is reachable by the striking foot and stays "
      f"{MIN_CLEARANCE:.2f} m clear of the support foot at a scatter of {scatter:.2f} m. "
      f"The best on offer is {room[viable].max() if viable.any() else 0.0:.3f} m, so this "
      "clip's stance is too narrow for a ball this size. Convert a different one, or lower "
      "--ball-scatter to match a narrower BALL_LATERAL_RANGE in kick_env_cfg.py."
    )

  # Among the positions that are safe, the one the foot hits hardest. Clearance is a
  # constraint rather than the objective: maximizing it alone walks the ball to the outside
  # edge of the swing, where the sole grazes past it and a 6 m/s foot delivers 4
  nominal = surface_gap(capsules[leg], grid)
  contacts = np.where((nominal <= 0.0).any(1), (nominal <= 0.0).argmax(1), 0)
  quality = np.where(
    allowed,
    approach_speed(capsules[leg], log, foot_id, origin, grid, contacts),
    -1.0,
  )
  best = pinned if pinned is not None else int(quality.argmax())

  ball = grid[best].astype(np.float32)
  gap = nominal[best]
  contact = int(contacts[best])
  clear = np.nonzero(gap[: contact + 1] > RESET_CLEARANCE)[0]

  return {
    "kick_step": np.int64(step),
    "kick_speed": np.float32(speed[step]),
    "kick_height": np.float32(foot_pos[step, 2]),
    "kick_dir": direction.astype(np.float32),
    "ball_pos": ball,
    "ball_clearance": np.float32(room[best]),
    "ball_approach": np.float32(quality[best]),
    "contact_step": np.int64(contact),
    "reset_limit": np.int64(clear[-1] if clear.size else 0),
  }


def convert_clip(
  sim: Simulation,
  scene: Scene,
  robot: Entity,
  joint_names: list[str],
  input_path: Path,
  clip: Clip,
  output_path: Path,
  output_fps: float,
  standing_height: float,
  hold_s: float,
  ball_scatter: float = BALL_SCATTER,
) -> dict[str, Any]:
  motion, leg = load_clip(input_path, clip, joint_names, str(sim.device))
  motion = prepend_hold(resample(canonicalize(motion), output_fps), hold_s)
  root_lin_vel, root_ang_vel, joint_vel = velocities(motion)

  foot_ids = robot.find_bodies(list(FOOT_BODY_NAMES), preserve_order=True)[0]
  strike_id = robot.find_bodies([f"{leg}_ankle_roll_link"])[0][0]

  # First pass: find how far off the ground the retargeted clip sits. The held opening is
  # exactly the standing phase the baseline wants, so it is measured over that
  probe = replay(sim, scene, robot, motion, root_lin_vel, root_ang_vel, joint_vel)
  probe_foot = probe["body_pos_w"][:, foot_ids, 2]
  stance_frames = int(round(output_fps))
  baseline = stance_baseline(probe_foot, stance_frames)
  z_shift = max(
    standing_height - baseline,
    (standing_height - MAX_GROUND_PENETRATION) - float(probe_foot.min()),
  )
  motion.root_pos[:, 2] += z_shift

  # Second pass: the real one
  log = replay(sim, scene, robot, motion, root_lin_vel, root_ang_vel, joint_vel)
  shifted_baseline = stance_baseline(log["body_pos_w"][:, foot_ids, 2], stance_frames)

  # The replay adds the env origin to every world position it logs, and the ball position
  # below is an absolute point rather than a displacement, so it has to come back off
  origin = scene.env_origins[0].cpu().numpy()
  strike = search_ball(
    joint_names,
    log,
    leg,
    strike_id,
    output_fps,
    origin,
    scatter=ball_scatter,
    override=clip.ball,
  )
  described = describe_clip(log, foot_ids, shifted_baseline)

  payload: dict[str, Any] = {
    "fps": np.array([output_fps], dtype=np.float32),
    **log,
    **described,
    **strike,
    "kick_leg": np.array(leg),
  }

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output_path, **payload)  # ty: ignore[invalid-argument-type]

  summary = {
    "name": output_path.stem,
    "file": output_path.name,
    "source": clip.source,
    "frames": int(log["joint_pos"].shape[0]),
    "fps": output_fps,
    # How far the clip walks in before it strikes. Named distance because that is the key
    # discover_motion_files orders a manifest by, and here it is the length of the approach
    "distance": round(float(np.linalg.norm(described["goal_xy"])), 3),
    "kick_leg": leg,
    "kick_step": int(strike["kick_step"]),
    "kick_speed": round(float(strike["kick_speed"]), 2),
    "kick_height": round(float(strike["kick_height"]), 3),
    "contact_step": int(strike["contact_step"]),
    "reset_limit": int(strike["reset_limit"]),
    "ball_clearance": round(float(strike["ball_clearance"]), 3),
    "ball_pos": [round(float(v), 3) for v in strike["ball_pos"]],
    "z_shift": round(z_shift, 4),
    "stance_float": round(shifted_baseline - standing_height, 4),
  }
  print(
    f"  {summary['name']:<10} {summary['frames']:>4} frames  {leg:>5} foot  "
    f"strike {summary['kick_step']:>3} at {summary['kick_speed']:.1f} m/s  "
    f"contact {summary['contact_step']:>3}  reset to {summary['reset_limit']:>3}  "
    f"ball ({summary['ball_pos'][0]:.2f}, {summary['ball_pos'][1]:.2f}) "
    f"clear {summary['ball_clearance']:.3f} m  "
    f"float {summary['stance_float']:+.3f} m"
  )
  return summary


def convert(
  motions: dict[str, Clip],
  clip_dir: Path,
  source_dir: Path,
  output_fps: float,
  hold_s: float,
  device: str,
  ball_scatter: float = BALL_SCATTER,
) -> None:
  """Convert published clips into mjlab motion npz files, one directory per motion.

  The scene is built once and every motion converted against it, so converting the whole
  table costs one startup rather than one each.
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARN] CUDA unavailable, falling back to CPU.")
    device = "cpu"

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene_cfg = SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    num_envs=1,
    entities={"robot": get_g1_robot_cfg()},
  )
  scene = Scene(scene_cfg, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  scene.reset()

  robot: Entity = scene["robot"]
  joint_names = list(robot.joint_names)
  standing_height = standing_foot_height(robot, sim, scene)
  print(f"Standing foot height in mjlab's G1: {standing_height:.4f} m")

  for name, clip in motions.items():
    output_dir = clip_dir / name
    print(f"Converting {name} from {clip.source} to {output_dir}:")
    summary = convert_clip(
      sim=sim,
      scene=scene,
      robot=robot,
      joint_names=joint_names,
      input_path=fetch(clip.source, source_dir),
      clip=clip,
      output_path=output_dir / f"{name}.npz",
      output_fps=output_fps,
      standing_height=standing_height,
      hold_s=hold_s,
      ball_scatter=ball_scatter,
    )

    # One motion per directory, so the manifest is one entry and is rewritten whole
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps([summary], indent=2))
    print(f"  wrote {output_dir / (name + '.npz')} and {manifest_path}")


def main(
  motion: str | None = None,
  clip_source: str | None = None,
  clip_start: int | None = None,
  clip_end: int | None = None,
  clip_dir: Path = CLIP_DIR,
  source_dir: Path = SOURCE_DIR,
  output_fps: float = 50.0,
  hold_s: float = STILL_HOLD_S,
  ball_scatter: float = BALL_SCATTER,
  device: str = "cuda:0",
) -> None:
  """Convert the PAiD kick clips into mjlab motion npz files.

  The three clip arguments override what MOTIONS says about one motion, for trying another
  of the thirteen out before writing it down. They need a motion, because there is one clip
  per motion and nothing to override without one.

  Args:
    motion: Which entry of MOTIONS to convert. Every one of them when left out.
    clip_source: Published clip to convert, instead of the one MOTIONS lists.
    clip_start: First frame, instead of the whole clip.
    clip_end: One past the last frame, instead of the whole clip.
    clip_dir: Parent of the per motion output directories.
    source_dir: Where the downloaded clips are cached.
    output_fps: Should match the env control rate, 1 / (timestep * decimation).
    hold_s: How long the reference stands still before the motion starts.
    ball_scatter: How far the reset event may move the ball off the position picked here.
      Must match BALL_FORWARD_RANGE and BALL_LATERAL_RANGE in kick_env_cfg.py.
    device: Torch device for the replay.
  """
  if motion is None:
    if clip_source or clip_start or clip_end:
      raise ValueError("A clip override needs a --motion to override")
    selected = MOTIONS
  else:
    if motion not in MOTIONS:
      raise ValueError(f"Unknown motion '{motion}'. Known: {', '.join(MOTIONS)}")
    listed = MOTIONS[motion]
    selected = {
      motion: Clip(
        source=clip_source or listed.source,
        start=clip_start if clip_start is not None else listed.start,
        end=clip_end if clip_end is not None else listed.end,
      )
    }

  convert(selected, clip_dir, source_dir, output_fps, hold_s, device, ball_scatter)


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
