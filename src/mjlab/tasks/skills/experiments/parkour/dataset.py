"""Build the parkour skill dataset: curated LAFAN1 clips, one folder per skill.

The Unitree-retargeted LAFAN1 files are long, multi-motion performances: one file
wanders through several behaviors with natural human transitions between them. This
script cuts them into short, single-behavior clips, normalizes each cut, labels it with
the goal it realizes, and files it under its skill:

  data/lafan1_g1/raw/<clip>.csv               the downloaded LAFAN1 clip
  data/lafan1_g1/clips/<skill>/<name>.npz     one normalized, labelled cut
  data/lafan1_g1/manifest.json                every cut and what it contains

Three skills: walk, run and jump. Sprint is gone -- one LAFAN1 source, a speed band
overlapping run's, and nothing the corridor asks for that run does not already cover.

## Cutting is task-specific, because the two behaviors are opposites

Steady locomotion and a jump are found by tests that are each other's negation, so they
get separate extractors rather than one parameterized one:

- `locomotion_segments` (walk, run) keeps the stretches where the yaw-frame velocity
  both *holds still* and *stays inside the skill's band*, frame by frame. Testing every
  frame rather than the cut's mean is what keeps the ends clean: a mean-only test happily
  accepts a cut that walks for three seconds and then stops, hops on the spot and turns
  away, which is exactly the junk that was showing up at the end of the walk and run
  clips. The accepted run is then eroded at both ends, because the smoothing window
  straddles the transition there and the last few frames are already contaminated by
  whatever comes next.

- `jump_segments` finds one cut per jump, and never merges two. A jump is located by its
  flight phase -- both feet off the ground, which the replayed motion reports directly --
  and then grown backwards through the crouch that launched it and forwards through the
  landing until the robot has absorbed it. Where two hops fall close together the cuts
  are split at the midpoint between them rather than merged, so a clip holds exactly one
  takeoff. That is the whole point: the jump skill is asked for *a* jump, and a reference
  set of run-on hopping teaches it to bounce continuously instead.

## Replay first, cut second

The source clip is replayed through the G1 once, in full, and everything after that
works on the replayed motion rather than on the CSV. Two reasons. The replay is what
gives foot heights, and therefore ground contact, which is what locates a jump; and the
velocities it reports are the ones the env will see at training time rather than a finite
difference of the CSV's root positions.

## Normalization

Each cut is placed canonically: the whole segment is rigidly transformed so its first
frame sits at the origin, facing +x. Global translation and global yaw are gone, so two
cuts of the same behavior recorded in different corners of the capture volume, facing
different ways, come out identical. Height, roll and pitch are absolute and stay that
way, because gravity does not care where the clip was recorded.

Alongside the placed motion, each clip stores what the policy actually reads: joint
positions and velocities, root height and orientation, root linear and angular velocity
*in the root's own yaw frame* (so they carry no heading either), foot contacts, and the
goal.

## Goals

The goal is what makes the skill conditionable, and it is per frame, so an episode
starting anywhere in the clip gets the goal measured from where it starts:

- walk, run: where the character ends up, as (dx, dy, dyaw) from frame t to the end of
  the cut, in frame t's own yaw frame. The continuous half of the conditioning is the
  reference root velocity, which the observation carries anyway.
- jump: apex height above the takeoff stance, and the landing displacement (dx, dy) from
  takeoff to touchdown in the takeoff yaw frame. Constant across the cut, since a jump
  realizes one goal, not a running one.

Run it (accept the dataset terms at
https://huggingface.co/datasets/unitreerobotics/LAFAN1_Retargeting_Dataset once, then
export your token -- the repo is gated):

  export HF_TOKEN=hf_...  # $env:HF_TOKEN="..." on Windows PowerShell
  uv run python -m mjlab.tasks.skills.experiments.parkour.dataset

  # rebuild one skill after retuning its spec
  uv run python -m mjlab.tasks.skills.experiments.parkour.dataset --skills "('jump',)"

  # report what the cuts would be without writing them (fast; still replays)
  uv run python -m mjlab.tasks.skills.experiments.parkour.dataset --dry-run

A produced clip is still a valid tracking motion, so it can be eyeballed with:

  uv run play Mjlab-Tracking-Flat-Unitree-G1 --agent zero --no-terminations True
    --motion-file data/lafan1_g1/clips/run/run1_subject2_00.npz
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

# LAFAN1 is 30 FPS. A CSV row is base pos (3) + base quat xyzw (4) + 29 joints.
FPS = 30.0

# Gated HuggingFace repo; the G1 CSVs live under the `g1/` subfolder.
HF_REPO = "lvhaidong/LAFAN1_Retargeting_Dataset"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/g1"

# The bodies whose height reports ground contact. Nothing else on the G1 touches the
# floor in normal locomotion, so a frame with neither of these down is a frame in flight.
FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")

# End effectors, whose position relative to the root is what a DeepMimic-style pose
# reward is weighted toward: they are where a tracking error is most visible.
END_EFFECTOR_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)

# The fields of a clip npz holding the placed motion, in the tracking format, so a clip
# stays playable with the tracking task.
MOTION_FIELDS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)

# The 29 G1 joints the CSV's DOF columns map to, in order.
G1_JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]


##
# Specs
##


@dataclass(frozen=True)
class LocomotionSpec:
  """How a steady-locomotion skill (walk, run) is cut out of its sources.

  Every threshold here is tested per frame, not against the cut's average. That is the
  difference between "this cut is mostly walking" and "every frame of this cut is
  walking", and only the second one produces clips that end cleanly.
  """

  clips: tuple[str, ...]
  """LAFAN1 clip names filed under this skill."""

  fwd_range: tuple[float, float]
  """Accepted forward speed [m/s], per frame. Signed, so a backwards stretch inside a
  run clip is not run data."""

  max_abs_lat: float = 0.6
  """Largest lateral speed [m/s] still counted as this behavior, per frame: sidestepping
  is not walking, even in a walk clip."""

  max_abs_yaw_rate: float = 1.0
  """Largest turn rate [rad/s], per frame. LAFAN1 has plenty of sharp turns, and a
  reference set that mixes them in teaches the skill that spinning is part of the gait."""

  max_abs_up: float = 0.5
  """Largest vertical root speed [m/s], per frame. This is what rejects the hop the
  human throws in at the end of a walk: it passes every planar test and fails this one."""

  change_tol: tuple[float, float, float] = (0.8, 0.8, 1.6)
  """Largest per-second change in [v_fwd, v_lat, yaw_rate] still counted as steady. It
  scales with speed: at walking pace a running-loose tolerance swallows the turns, and at
  running pace a walking-tight one finds only the stretches where the human stands
  still."""

  smooth_window: int = 15
  """Moving average, in frames (at the replay rate), applied to the velocity before it
  is cut on. The root surges once per stride, so this has to be long enough to see
  through the surge and short enough not to smear a real change of speed."""

  close_gap: float = 0.12
  """Holes in the accepted mask shorter than this [s] are filled before cutting. A
  single stride that grazes a tolerance would otherwise split an obviously continuous
  four-second walk into two two-second ones, or into nothing at all once the minimum
  duration is applied."""

  trim: float = 0.15
  """Seconds cut off each end of an accepted run. The smoothing window straddles the
  transition there, so those frames are already part of whatever comes next."""

  min_duration: float = 1.0
  """Reject a cut shorter than this [s]: too brief to be a behavior."""

  max_duration: float = 6.0
  """Split a cut longer than this [s] into equal pieces. Long cuts are fine motion but
  make for coarse goal conditioning -- the end goal of an eight-second walk says almost
  nothing about the next half second."""


@dataclass(frozen=True)
class JumpSpec:
  """How individual jumps are cut out of their sources.

  A jump is located by its flight phase and grown outwards into the crouch and the
  landing. Cuts are never merged: a clip holds one takeoff.
  """

  clips: tuple[str, ...]
  """LAFAN1 clip names filed under this skill."""

  flight_clearance: float = 0.06
  """How far above its grounded height a foot must be to count as off the floor [m].
  Loose enough to see past retargeting noise in the ankle, tight enough not to call the
  swing leg of an ordinary stride a flight phase (the other foot is still down then, and
  flight needs *both* up)."""

  min_flight: float = 0.10
  """Shortest flight phase that counts as a jump [s]. Below this it is a stumble or a
  retargeting artifact."""

  max_flight: float = 1.2
  """Longest flight phase that counts as a jump [s]. Beyond this the retargeted motion
  has lost the floor entirely and the cut is not a jump."""

  min_apex: float = 0.05
  """Smallest rise of the root above the character's stance height that counts as a jump
  [m]. This is what separates a jump from a bouncy running stride, both of which have a
  flight phase; the stride's root passes through stance height rather than well over
  it."""

  lead: float = 0.45
  """Seconds before takeoff the cut reaches back for, to hold the crouch."""

  trail: float = 0.55
  """Seconds after touchdown the cut runs on for, to hold the landing being absorbed."""

  max_abs_yaw_rate: float = 1.0
  """Reject a jump whose mean turn rate exceeds this [rad/s]: LAFAN1's jump sources are
  full of spinning hops, and they are a different behavior from a jump that goes
  somewhere."""

  max_abs_lat: float = 0.6
  """Reject a jump whose mean lateral speed exceeds this [m/s]. Same argument: a sideways
  hop is not the forward jump the corridor needs."""

  min_duration: float = 0.6
  """Reject a cut shorter than this [s]."""


SkillSpec = LocomotionSpec | JumpSpec

# The three skills. Walk and run are separated by the speed band their frames must stay
# inside; jump is separated by being found a different way entirely.
SKILLS: dict[str, SkillSpec] = {
  "walk": LocomotionSpec(
    clips=(
      "walk1_subject1",
      "walk1_subject2",
      "walk1_subject5",
      "walk3_subject1",
      "walk3_subject2",
    ),
    fwd_range=(0.3, 1.7),
    max_abs_lat=0.6,
    max_abs_yaw_rate=1.0,
    max_abs_up=0.35,
    # The yaw tolerance is much looser than the planar ones, and has to be: a walking
    # pelvis counter-rotates once per stride, so the yaw *rate* swings hard even while
    # the heading holds perfectly steady. Tightening this to the planar tolerance is
    # what reduced the walk set to a handful of seconds.
    change_tol=(0.8, 0.8, 2.5),
    # ~0.7 s at the 50 Hz replay rate, which is most of a walking stride. Sized in
    # replay frames, not in the 30 Hz frames of the source CSV.
    smooth_window=35,
  ),
  "run": LocomotionSpec(
    clips=("run1_subject2", "run1_subject5", "run2_subject1", "run2_subject4"),
    fwd_range=(1.5, 3.5),
    max_abs_lat=0.8,
    max_abs_yaw_rate=1.0,
    # A run has real vertical motion every stride, so this is looser than walk's. It is
    # still what keeps a standing jump inside a run clip out of the run set.
    max_abs_up=0.7,
    # Looser than walk's throughout: the surge is bigger at speed, so a walking-tight
    # tolerance here finds nothing but the stretches where the human is standing still.
    change_tol=(2.0, 2.0, 5.0),
    # ~0.8 s at 50 Hz, a little over a running stride.
    smooth_window=41,
  ),
  "jump": JumpSpec(
    clips=("jumps1_subject1", "jumps1_subject2", "jumps1_subject5"),
  ),
}

# What each skill's goal vector holds, in order. Read by the env to name its channels.
GOAL_CHANNELS: dict[str, tuple[str, ...]] = {
  "walk": ("goal_dx", "goal_dy", "goal_dyaw"),
  "run": ("goal_dx", "goal_dy", "goal_dyaw"),
  "jump": ("apex_height", "land_dx", "land_dy"),
}


##
# Download
##


def download(clip: str, raw_dir: Path) -> Path:
  """Fetch one G1 CSV into `raw_dir/<clip>.csv` (skips if already there)."""
  path = raw_dir / f"{clip}.csv"
  if path.exists():
    return path
  token = os.environ.get("HF_TOKEN")
  if not token:
    raise SystemExit(
      "The LAFAN1 repo is gated. Accept its terms on HuggingFace and set HF_TOKEN "
      "before running."
    )
  raw_dir.mkdir(parents=True, exist_ok=True)
  request = urllib.request.Request(
    f"{HF_BASE}/{clip}.csv", headers={"Authorization": f"Bearer {token}"}
  )
  print(f"Downloading {clip}.csv -> {path} ...")
  with urllib.request.urlopen(request) as response, open(path, "wb") as f:
    f.write(response.read())
  return path


##
# Quaternion helpers, on (N, 4) wxyz arrays
##


def quat_yaw(quat: np.ndarray) -> np.ndarray:
  """Yaw angle [rad] of each quaternion."""
  w, x, y, z = quat.T
  return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Hamilton product, broadcasting over the leading dimension."""
  aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
  bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
  return np.stack(
    [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ],
    axis=-1,
  )


def quat_from_yaw(yaw: np.ndarray) -> np.ndarray:
  """Rotation about z, as a wxyz quaternion."""
  half = 0.5 * np.asarray(yaw)
  zeros = np.zeros_like(half)
  return np.stack([np.cos(half), zeros, zeros, np.sin(half)], axis=-1)


def rotate_z(vectors: np.ndarray, yaw: np.ndarray | float) -> np.ndarray:
  """Rotate xyz vectors about z by `yaw`, broadcasting over the leading dimension."""
  cos, sin = np.cos(yaw), np.sin(yaw)
  x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
  return np.stack([cos * x - sin * y, sin * x + cos * y, z], axis=-1)


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
  return (angle + np.pi) % (2.0 * np.pi) - np.pi


def smooth(signal: np.ndarray, window: int) -> np.ndarray:
  """Centered per-channel moving average, edge-padded so length is preserved."""
  if window <= 1:
    return signal
  pad = window // 2
  padded = np.pad(signal, ((pad, pad), (0, 0)), mode="edge")
  kernel = np.ones(window) / window
  return np.stack(
    [
      np.convolve(padded[:, i], kernel, mode="valid")[: len(signal)]
      for i in range(signal.shape[1])
    ],
    axis=1,
  )


##
# Replay
##


def build_sim(device: str, output_fps: float) -> tuple[Simulation, Scene]:
  """The single-env G1 sim every source is replayed through (expensive; build once)."""
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return sim, scene


def replay(
  sim: Simulation, scene: Scene, csv_path: Path, output_fps: float
) -> dict[str, np.ndarray]:
  """Drive the robot through one whole source CSV and record what the sim reports.

  The recorded state comes off the entity rather than out of the CSV, which is why this
  is a replay and not a format conversion: joint and body ordering, and the derived body
  velocities, are then exactly the ones the env will see at training time. Cutting
  happens afterwards, on these arrays, because this is also where foot heights (and so
  ground contact) come from.
  """
  motion = MotionLoader(
    motion_file=str(csv_path),
    input_fps=int(FPS),
    output_fps=int(output_fps),
    device=sim.device,
  )
  robot: Entity = scene["robot"]
  joint_idx = robot.find_joints(G1_JOINT_NAMES, preserve_order=True)[0]
  log: dict[str, list[np.ndarray]] = {field: [] for field in MOTION_FIELDS}

  scene.reset()
  for _ in tqdm(
    range(motion.output_frames),
    desc=csv_path.stem,
    unit="frame",
    ncols=100,
    leave=False,
  ):
    (base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel), _ = (
      motion.get_next_state()
    )

    root_state = robot.data.default_root_state.clone()
    root_state[:, 0:3] = base_pos
    root_state[:, :2] += scene.env_origins[:, :2]
    root_state[:, 3:7] = base_rot
    root_state[:, 7:10] = base_lin_vel
    root_state[:, 10:] = base_ang_vel
    robot.write_root_state_to_sim(root_state)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, joint_idx] = dof_pos
    joint_vel[:, joint_idx] = dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    data = robot.data
    log["joint_pos"].append(data.joint_pos[0].cpu().numpy().copy())
    log["joint_vel"].append(data.joint_vel[0].cpu().numpy().copy())
    log["body_pos_w"].append(data.body_link_pos_w[0].cpu().numpy().copy())
    log["body_quat_w"].append(data.body_link_quat_w[0].cpu().numpy().copy())
    log["body_lin_vel_w"].append(data.body_link_lin_vel_w[0].cpu().numpy().copy())
    log["body_ang_vel_w"].append(data.body_link_ang_vel_w[0].cpu().numpy().copy())

  motion_arrays = {field: np.stack(frames, axis=0) for field, frames in log.items()}
  # The replay is placed at the scene origin, which for a one-env scene is the world
  # origin; subtracting it anyway keeps this honest if that ever changes.
  origin = scene.env_origins[0].cpu().numpy()
  motion_arrays["body_pos_w"] = motion_arrays["body_pos_w"] - np.array(
    [origin[0], origin[1], 0.0]
  )
  return motion_arrays


##
# Signals derived from a replay
##


@dataclass
class Signals:
  """What the segmenters read: everything is per frame, at the replay rate."""

  fps: float
  twist: np.ndarray
  """(T, 4) smoothed [v_fwd, v_lat, v_up, yaw_rate] in the root's own yaw frame."""
  raw_twist: np.ndarray
  """(T, 4) the same, unsmoothed. What a label should be read off."""
  foot_z: np.ndarray
  """(T, 2) height of each foot body."""
  grounded: np.ndarray
  """(T, 2) whether each foot is on the floor."""
  airborne: np.ndarray
  """(T,) whether neither foot is on the floor."""
  root_z: np.ndarray
  """(T,) root height."""
  stance_height: float
  """Root height with both feet planted, over this whole source. What a jump's apex is
  measured against: the rise from the takeoff frame alone is not it, since by then the
  legs are already extended and most of the rise has happened."""


def derive_signals(
  motion: dict[str, np.ndarray],
  foot_indexes: tuple[int, ...],
  fps: float,
  smooth_window: int,
  flight_clearance: float,
) -> Signals:
  """Turn a replayed source into the per-frame signals both segmenters cut on."""
  root_quat = motion["body_quat_w"][:, 0]
  yaw = quat_yaw(root_quat)
  lin_vel_w = motion["body_lin_vel_w"][:, 0]
  ang_vel_w = motion["body_ang_vel_w"][:, 0]

  # Into the root's own yaw frame, which is what makes the label independent of where in
  # the capture volume the motion happened and which way the human was facing.
  lin_vel_b = rotate_z(lin_vel_w, -yaw)
  ang_vel_b = rotate_z(ang_vel_w, -yaw)
  raw_twist = np.concatenate([lin_vel_b, ang_vel_b[:, 2:3]], axis=1)

  foot_z = motion["body_pos_w"][:, list(foot_indexes), 2]
  # The grounded height is read off the clip rather than assumed: retargeting leaves the
  # ankle a few centimetres above or below where the model would put it, and a fixed
  # threshold then reports a whole source as permanently airborne or never airborne.
  floor = np.percentile(foot_z, 5.0, axis=0)
  grounded = foot_z < (floor + flight_clearance)

  root_z = motion["body_pos_w"][:, 0, 2]
  airborne = np.asarray(~grounded[:, 0] & ~grounded[:, 1])
  planted = np.asarray(grounded[:, 0] & grounded[:, 1])
  stance_height = float(
    np.median(root_z[planted]) if planted.any() else np.median(root_z)
  )

  return Signals(
    fps=fps,
    twist=smooth(raw_twist, smooth_window),
    raw_twist=raw_twist,
    foot_z=foot_z,
    grounded=grounded,
    airborne=airborne,
    root_z=root_z,
    stance_height=stance_height,
  )


def _runs(mask: np.ndarray, min_frames: int = 1) -> list[tuple[int, int]]:
  """The [start, end) ranges where `mask` holds for at least `min_frames`."""
  ranges: list[tuple[int, int]] = []
  i, n = 0, len(mask)
  while i < n:
    if not mask[i]:
      i += 1
      continue
    j = i
    while j < n and mask[j]:
      j += 1
    if j - i >= min_frames:
      ranges.append((i, j))
    i = j
  return ranges


##
# Segmentation: steady locomotion
##


def locomotion_segments(
  signals: Signals, spec: LocomotionSpec
) -> list[tuple[int, int]]:
  """Cuts where every frame is this skill, steadily.

  Two tests, ANDed frame by frame. `in_band` asks whether this frame is the behavior at
  all -- forward speed inside the skill's window, not sidestepping, not turning, not
  leaving the ground. `steady` asks whether the velocity is holding still, so a cut is
  one goal rather than a ramp between two. The accepted runs are then eroded at both
  ends, since the smoothing window has already mixed the neighbouring behavior into those
  frames, and split if they run long.
  """
  twist = signals.twist
  v_fwd, v_lat, v_up, yaw_rate = twist[:, 0], twist[:, 1], twist[:, 2], twist[:, 3]

  low, high = spec.fwd_range
  in_band = (
    (v_fwd >= low)
    & (v_fwd <= high)
    & (np.abs(v_lat) <= spec.max_abs_lat)
    & (np.abs(yaw_rate) <= spec.max_abs_yaw_rate)
    & (np.abs(v_up) <= spec.max_abs_up)
  )

  planar = twist[:, [0, 1, 3]]
  change = np.abs(np.gradient(planar, axis=0)) * signals.fps
  steady = np.all(change <= np.array(spec.change_tol), axis=1)

  trim = int(round(spec.trim * signals.fps))
  min_frames = int(round(spec.min_duration * signals.fps))
  max_frames = int(round(spec.max_duration * signals.fps))
  accepted = _close_gaps(in_band & steady, int(round(spec.close_gap * signals.fps)))

  cuts: list[tuple[int, int]] = []
  for start, end in _runs(accepted, min_frames + 2 * trim):
    cuts.extend(_split_long((start + trim, end - trim), min_frames, max_frames))
  return [(s, e) for s, e in cuts if e - s >= min_frames]


def _close_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
  """Fill runs of False shorter than `max_gap`, leaving the ends alone."""
  if max_gap <= 0:
    return mask
  closed = mask.copy()
  for start, end in _runs(~mask):
    if end - start <= max_gap and start > 0 and end < len(mask):
      closed[start:end] = True
  return closed


def _split_long(
  segment: tuple[int, int], min_frames: int, max_frames: int
) -> list[tuple[int, int]]:
  """Break an over-long cut into equal pieces, each still at least `min_frames`."""
  start, end = segment
  length = end - start
  if length <= max_frames:
    return [segment]
  pieces = int(np.ceil(length / max_frames))
  if length // pieces < min_frames:
    return [segment]
  edges = np.linspace(start, end, pieces + 1).astype(int)
  return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]


##
# Segmentation: individual jumps
##


@dataclass
class Jump:
  """One located jump: the cut, and the measurements that make its goal."""

  start: int
  end: int
  takeoff: int
  touchdown: int
  apex_height: float
  land_dx: float
  land_dy: float


def jump_segments(signals: Signals, spec: JumpSpec) -> list[Jump]:
  """One cut per jump, from the crouch that launched it to the absorbed landing.

  Flight (neither foot down) is what locates a jump; the cut is then grown outwards. The
  crouch is found by walking back from takeoff to where the root stopped descending,
  which is the top of the countermovement, and the landing by running on past touchdown
  while the root is still being lowered. Where two hops sit closer together than the
  padding, the boundary between them is placed at the midpoint rather than the two being
  merged: one cut, one takeoff.
  """
  fps = signals.fps
  min_flight = max(int(round(spec.min_flight * fps)), 1)
  max_flight = int(round(spec.max_flight * fps))
  lead = int(round(spec.lead * fps))
  trail = int(round(spec.trail * fps))
  min_frames = int(round(spec.min_duration * fps))
  total = len(signals.root_z)

  flights = [
    (s, e) for s, e in _runs(signals.airborne, min_flight) if e - s <= max_flight
  ]

  jumps: list[Jump] = []
  for index, (takeoff, touchdown) in enumerate(flights):
    # Against the stance height, not against the root at takeoff: by takeoff the legs
    # have already extended, so a takeoff-relative rise measures only the part of the
    # jump that happens in the air and reports every jump as a few centimetres.
    apex_height = float(signals.root_z[takeoff:touchdown].max() - signals.stance_height)
    if apex_height < spec.min_apex:
      continue

    # Back to the top of the countermovement: the root descends into a crouch before it
    # is driven up, so the last frame before takeoff that is not descending is where the
    # jump begins as a motion.
    start = max(0, takeoff - lead)
    descending = np.diff(signals.root_z[start : takeoff + 1]) < 0.0
    if descending.any():
      start = start + int(np.argmax(descending))

    # Forward until the landing has been absorbed: past touchdown the root sinks as the
    # legs give, and the jump is over when it stops sinking.
    end = min(total, touchdown + trail)
    settling = np.diff(signals.root_z[touchdown:end]) < 0.0
    if settling.any() and not settling.all():
      # First frame after touchdown that is no longer sinking, plus a little to hold it.
      settled = int(np.argmin(settling)) + touchdown + 1
      end = min(end, settled + trail // 2)

    # Never merge: split the gap to the neighbouring jump down the middle instead.
    if index > 0:
      start = max(start, (flights[index - 1][1] + takeoff) // 2)
    if index + 1 < len(flights):
      end = min(end, (touchdown + flights[index + 1][0]) // 2)

    if end - start < min_frames:
      continue
    if abs(float(signals.twist[start:end, 3].mean())) > spec.max_abs_yaw_rate:
      continue
    if abs(float(signals.twist[start:end, 1].mean())) > spec.max_abs_lat:
      continue

    # The landing displacement needs the root positions, which the caller holds; it
    # fills these in with `measure_landing`.
    jumps.append(
      Jump(
        start=int(start),
        end=int(end),
        takeoff=int(takeoff),
        touchdown=int(touchdown),
        apex_height=apex_height,
        land_dx=0.0,
        land_dy=0.0,
      )
    )
  return jumps


def measure_landing(motion: dict[str, np.ndarray], jump: Jump) -> tuple[float, float]:
  """Horizontal displacement from takeoff to touchdown, in the takeoff yaw frame."""
  root_pos = motion["body_pos_w"][:, 0]
  yaw = quat_yaw(motion["body_quat_w"][:, 0])
  touchdown = min(jump.touchdown, len(root_pos) - 1)
  delta = root_pos[touchdown] - root_pos[jump.takeoff]
  local = rotate_z(delta[None, :], -yaw[jump.takeoff])[0]
  return float(local[0]), float(local[1])


##
# Normalization and labelling
##


def place_canonically(motion: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  """Rigidly move a cut so its first frame is at the origin, facing +x.

  Global translation and global yaw are what carry "where in the capture volume this
  happened, pointing which way", and neither is part of the behavior. Height, roll and
  pitch are left alone: gravity is not relative.
  """
  root_pos0 = motion["body_pos_w"][0, 0]
  yaw0 = float(quat_yaw(motion["body_quat_w"][0, 0][None, :])[0])
  offset = np.array([root_pos0[0], root_pos0[1], 0.0])
  spin = quat_from_yaw(np.array(-yaw0))

  placed = dict(motion)
  placed["body_pos_w"] = rotate_z(motion["body_pos_w"] - offset, -yaw0)
  placed["body_quat_w"] = quat_mul(
    np.broadcast_to(spin, motion["body_quat_w"].shape), motion["body_quat_w"]
  )
  placed["body_lin_vel_w"] = rotate_z(motion["body_lin_vel_w"], -yaw0)
  placed["body_ang_vel_w"] = rotate_z(motion["body_ang_vel_w"], -yaw0)
  return placed


def root_frame_state(motion: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  """The root-frame quantities the policy reads, derived from a placed cut.

  Velocities go into the root's own yaw frame, so they carry no heading at all: the same
  stride reads identically wherever the character has turned to by then.
  """
  root_quat = motion["body_quat_w"][:, 0]
  yaw = quat_yaw(root_quat)
  return {
    "root_height": motion["body_pos_w"][:, 0, 2].astype(np.float32),
    "root_pos_local": motion["body_pos_w"][:, 0].astype(np.float32),
    "root_yaw_local": yaw.astype(np.float32),
    "root_quat_local": root_quat.astype(np.float32),
    "root_lin_vel_b": rotate_z(motion["body_lin_vel_w"][:, 0], -yaw).astype(np.float32),
    "root_ang_vel_b": rotate_z(motion["body_ang_vel_w"][:, 0], -yaw).astype(np.float32),
  }


def end_effector_offsets(
  motion: dict[str, np.ndarray], ee_indexes: tuple[int, ...]
) -> np.ndarray:
  """End-effector positions relative to the root, in the root's yaw frame."""
  root_pos = motion["body_pos_w"][:, 0]
  yaw = quat_yaw(motion["body_quat_w"][:, 0])
  offsets = motion["body_pos_w"][:, list(ee_indexes)] - root_pos[:, None, :]
  return rotate_z(offsets, -yaw[:, None]).astype(np.float32)


def locomotion_goal(state: dict[str, np.ndarray]) -> np.ndarray:
  """(T, 3) where the character ends up, from each frame, in that frame's yaw frame.

  Per frame rather than per clip because an episode may start anywhere in the cut, and a
  goal measured from the cut's first frame would then be asking for a displacement the
  character has already partly made.
  """
  pos = state["root_pos_local"]
  yaw = state["root_yaw_local"]
  delta = pos[-1][None, :] - pos
  local = rotate_z(delta, -yaw)
  return np.stack([local[:, 0], local[:, 1], wrap_to_pi(yaw[-1] - yaw)], axis=1).astype(
    np.float32
  )


def jump_goal(jump: Jump, num_frames: int) -> np.ndarray:
  """(T, 3) apex height and landing displacement, constant across the cut.

  Constant because a jump realizes one goal rather than a running one: the height it
  reaches and where it puts the character down are properties of the whole jump, and
  asking for "the remaining apex" partway through would make the same jump a different
  command at every frame.
  """
  goal = np.array([jump.apex_height, jump.land_dx, jump.land_dy], dtype=np.float32)
  return np.broadcast_to(goal, (num_frames, 3)).copy()


def save_clip(
  path: Path,
  skill: str,
  motion: dict[str, np.ndarray],
  state: dict[str, np.ndarray],
  contacts: np.ndarray,
  ee_offsets: np.ndarray,
  goal: np.ndarray,
  fps: float,
) -> None:
  """Write one normalized, labelled reference clip.

  Every field is named here rather than expanded from a dict, so this call is the clip
  format: what a clip holds is readable in one place, and `motions.py` reads back exactly
  these names.
  """
  np.savez(
    path,
    fps=np.array([fps], dtype=np.float32),
    skill=np.array(skill),
    # The goal, and what its channels mean for this skill.
    goal=goal.astype(np.float32),
    goal_channels=np.array(GOAL_CHANNELS[skill]),
    # Root-frame state: no global position, no heading.
    root_height=state["root_height"],
    root_pos_local=state["root_pos_local"],
    root_yaw_local=state["root_yaw_local"],
    root_quat_local=state["root_quat_local"],
    root_lin_vel_b=state["root_lin_vel_b"],
    root_ang_vel_b=state["root_ang_vel_b"],
    contacts=contacts.astype(np.float32),
    ee_offsets=ee_offsets.astype(np.float32),
    # The placed motion, in the tracking format, so a clip stays playable.
    joint_pos=motion["joint_pos"].astype(np.float32),
    joint_vel=motion["joint_vel"].astype(np.float32),
    body_pos_w=motion["body_pos_w"].astype(np.float32),
    body_quat_w=motion["body_quat_w"].astype(np.float32),
    body_lin_vel_w=motion["body_lin_vel_w"].astype(np.float32),
    body_ang_vel_w=motion["body_ang_vel_w"].astype(np.float32),
  )


##
# Reporting
##


def print_summary(skill: str, entries: list[dict[str, Any]], fps: float) -> None:
  """Per-skill statistics: what was kept, and what the goals span."""
  if not entries:
    print(f"{skill}: no clips accepted")
    return

  durations = np.array([e["num_frames"] for e in entries]) / fps
  print(
    f"{skill}: {len(entries)} clip(s), {durations.sum():.1f} s of reference motion "
    f"({durations.min():.2f}-{durations.max():.2f} s each)"
  )
  twist = np.array([e["twist_mean"] for e in entries])
  for i, channel in enumerate(("v_fwd", "v_lat", "v_up", "yaw_rate")):
    column = twist[:, i]
    print(
      f"  mean {channel:9s} min {column.min():6.2f}  avg {column.mean():6.2f}  "
      f"max {column.max():6.2f}"
    )
  flight = np.array([e["flight_fraction"] for e in entries])
  print(
    f"  flight fraction  min {flight.min():6.2f}  avg {flight.mean():6.2f}  "
    f"max {flight.max():6.2f}"
  )
  goal = np.array([e["goal_start"] for e in entries])
  for i, channel in enumerate(GOAL_CHANNELS[skill]):
    column = goal[:, i]
    print(
      f"  goal {channel:12s} min {column.min():6.2f}  avg {column.mean():6.2f}  "
      f"max {column.max():6.2f}"
    )
  print(
    "  suggested goal box: "
    + ", ".join(
      f"{c} ({goal[:, i].min():.2f}, {goal[:, i].max():.2f})"
      for i, c in enumerate(GOAL_CHANNELS[skill])
    )
  )


##
# Entry point
##


@dataclass
class Config:
  data_dir: Path = Path("data/lafan1_g1")
  """Where raw sources, clips and the manifest live."""
  skills: tuple[str, ...] = tuple(SKILLS)
  """Which skills to rebuild. The others' folders are left alone."""
  dry_run: bool = False
  """Report the cuts without writing any clip. Still replays, since the cuts depend on
  the replay."""
  device: str = "cuda:0"
  output_fps: float = 50.0
  """Replay rate. Match the env's control rate so a reference frame is a control step."""
  clean: bool = True
  """Remove a skill's existing clips before rebuilding it, so a narrowed spec does not
  leave the cuts it no longer accepts lying in the folder."""


def main(cfg: Config) -> None:
  """Download, replay, cut, normalize, label and write the reference clips."""
  unknown = set(cfg.skills) - set(SKILLS)
  if unknown:
    raise SystemExit(f"Unknown skill(s) {sorted(unknown)}; known: {sorted(SKILLS)}.")

  device = cfg.device
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA is not available. Falling back to CPU. This may be slow.")
    device = "cpu"

  raw_dir = cfg.data_dir / "raw"
  print("Building the G1 replay sim...")
  sim, scene = build_sim(device, cfg.output_fps)
  robot: Entity = scene["robot"]
  foot_indexes = tuple(robot.body_names.index(n) for n in FOOT_BODY_NAMES)
  ee_indexes = tuple(robot.body_names.index(n) for n in END_EFFECTOR_NAMES)

  manifest: dict[str, list[dict[str, Any]]] = {}

  for skill in cfg.skills:
    spec = SKILLS[skill]
    clip_dir = cfg.data_dir / "clips" / skill
    if cfg.clean and clip_dir.exists() and not cfg.dry_run:
      for stale in clip_dir.glob("*.npz"):
        stale.unlink()
    entries: list[dict[str, Any]] = []

    for source in spec.clips:
      csv_path = download(source, raw_dir)
      motion = replay(sim, scene, csv_path, cfg.output_fps)
      smooth_window = spec.smooth_window if isinstance(spec, LocomotionSpec) else 5
      flight_clearance = spec.flight_clearance if isinstance(spec, JumpSpec) else 0.06
      signals = derive_signals(
        motion, foot_indexes, cfg.output_fps, smooth_window, flight_clearance
      )

      if isinstance(spec, JumpSpec):
        jumps = jump_segments(signals, spec)
        for jump in jumps:
          jump.land_dx, jump.land_dy = measure_landing(motion, jump)
        cuts: list[tuple[int, int]] = [(j.start, j.end) for j in jumps]
      else:
        jumps = []
        cuts = locomotion_segments(signals, spec)

      print(
        f"{skill}/{source}: {len(signals.root_z)} frames -> {len(cuts)} cut(s) "
        f"({sum(e - s for s, e in cuts) / cfg.output_fps:.1f} s)"
      )

      for index, (start, end) in enumerate(cuts):
        name = f"{source}_{index:02d}"
        segment = {f: motion[f][start:end] for f in MOTION_FIELDS}
        placed = place_canonically(segment)
        state = root_frame_state(placed)
        contacts = signals.grounded[start:end]
        ee_offsets = end_effector_offsets(placed, ee_indexes)
        goal = (
          jump_goal(jumps[index], end - start)
          if isinstance(spec, JumpSpec)
          else locomotion_goal(state)
        )

        entry: dict[str, Any] = {
          "name": name,
          "skill": skill,
          "source": source,
          "start": int(start),
          "end": int(end),
          "num_frames": int(end - start),
          "twist_mean": signals.raw_twist[start:end].mean(axis=0).round(4).tolist(),
          "goal_start": goal[0].round(4).tolist(),
          "goal_channels": list(GOAL_CHANNELS[skill]),
          "flight_fraction": round(float(signals.airborne[start:end].mean()), 4),
        }
        if isinstance(spec, JumpSpec):
          entry["apex_height"] = round(jumps[index].apex_height, 4)

        if not cfg.dry_run:
          clip_dir.mkdir(parents=True, exist_ok=True)
          clip_path = clip_dir / f"{name}.npz"
          save_clip(
            clip_path,
            skill,
            placed,
            state,
            contacts,
            ee_offsets,
            goal,
            cfg.output_fps,
          )
          entry["clip"] = str(clip_path)

        entries.append(entry)

    manifest[skill] = entries

  if not cfg.dry_run:
    manifest_path = cfg.data_dir / "manifest.json"
    if manifest_path.exists():
      # Only the rebuilt skills are replaced, so a partial run does not erase the rest.
      # Unknown keys are dropped rather than carried along: an older manifest sitting in
      # this directory is a different format, not a skill.
      previous = json.loads(manifest_path.read_text())
      kept = {k: v for k, v in previous.items() if k in SKILLS and k not in manifest}
      manifest = {**kept, **manifest}
    manifest_path.write_text(json.dumps(manifest, indent=2))

  print()
  for skill in cfg.skills:
    print_summary(skill, manifest[skill], cfg.output_fps)
  total = sum(len(e) for e in manifest.values())
  where = "would write" if cfg.dry_run else "wrote"
  print(f"\n{where} {total} clip(s) under {cfg.data_dir}/")


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
