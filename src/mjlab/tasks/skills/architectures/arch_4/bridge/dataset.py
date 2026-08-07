"""The corpus the in-betweener learns from: recorded motion with holes cut in it.

The in-betweener is given the frames before a gap, the frames after it, and how long the
gap is, and has to produce what goes in between. The only teacher that knows the answer
is a real body that actually did it, so this script turns LAFAN1 into a pile of such
questions.

LAFAN1 is long continuous mocap: a subject walks, turns, breaks into a run, jumps, stops,
stands around, and keeps going, for minutes at a time. That is the reason it is the
corpus rather than a set of curated single-skill clips. A clip of one skill has no hole
worth cutting, because the middle of a steady walk is filled by walking. What is wanted
is the thousands of moments where one thing becomes another, and those only exist in a
long take.

    uv run python -m mjlab.tasks.skills.architectures.arch_4.bridge.dataset

Two stages, both cached, both skipped for work already done.

1. Fetch and convert. Each retargeted CSV is downloaded, its 29 columns scattered into
   mjlab's G1 by name, resampled from 30 Hz to the control rate, differenced for
   velocities, and stood on the floor. One npz per clip in `data/lafan1/motions`.
2. Cut. Windows are laid over each clip at a fixed stride: some frames of context, a
   hole, some more context. A window is kept only if the body is doing something during
   the hole and is not on the ground, which is what keeps the corpus from being mostly
   idle standing.

Nothing here is made up. Every window is a real stretch of a real recording and the
frames inside the hole are the true answer. Pairs that were never continuations, stitched
from two different clips, are a separate idea and are not in this file.

The `__main__` plays the result back so the cut can be looked at rather than guessed at:
green while the ghost is in context, red while it is inside the hole.

Licensing. LAFAN1 is Ubisoft's, under CC BY-NC-ND 4.0: non-commercial use only. The
retargeting to Unitree humanoids is a separate release, downloaded here from a public
mirror because the official repository is access-gated. Cite Harvey et al., SIGGRAPH 2020
for the motions and the retargeting release for the G1 trajectories.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro

import mjlab
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.scene import Scene, SceneCfg
from mjlab.tasks.skills.architectures.arch_4.bridge.view import (
  CONTEXT_COLOR,
  MASKED_COLOR,
  Ghost,
  visual_meshes,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_conjugate, quat_mul

# The 29 joints the retargeted CSV holds, in the order its columns use. This is Unitree's
# own order for the G1 and it is not mjlab's; column k below is written into the joint of
# that name, wherever the model happens to keep it. Getting this wrong does not crash
# anything: the robot still moves, it just moves the wrong limbs.
LAFAN1_JOINT_NAMES: tuple[str, ...] = (
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
)

SOURCE_FPS = 30.0

# Unitree's own repository (unitreerobotics/LAFAN1_Retargeting_Dataset) holds the same
# files but is access-gated, so an unauthenticated download returns 401 rather than data.
# This public mirror is what makes the script run without a HuggingFace token.
LAFAN1_REPO = "lvhaidong/LAFAN1_Retargeting_Dataset"
LAFAN1_ROBOT = "g1"

FOOT_BODY_NAMES: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")

# What the robot is called inside the compiled scene, which prefixes every body and joint
# name the model knows.
ENTITY_NAME = "robot"

WALK_CLIPS: tuple[str, ...] = (
  "walk1_subject1",
  "walk1_subject2",
  "walk1_subject5",
  "walk2_subject1",
  "walk2_subject3",
  "walk2_subject4",
  "walk3_subject1",
  "walk3_subject2",
  "walk3_subject3",
  "walk3_subject4",
  "walk3_subject5",
  "walk4_subject1",
)

RUN_CLIPS: tuple[str, ...] = (
  "run1_subject2",
  "run1_subject5",
  "run2_subject1",
  "run2_subject4",
  "sprint1_subject2",
  "sprint1_subject4",
)

JUMP_CLIPS: tuple[str, ...] = (
  "jumps1_subject1",
  "jumps1_subject2",
  "jumps1_subject5",
)

# Not fetched by default, for two different reasons. The dance and fight takes are real
# motion and would widen the corpus, but they are upper-body performances the corridor
# never calls for. The fall takes are worse than merely irrelevant: they are minutes of a
# body on the ground, and an in-betweener that has learned lying down is a normal thing
# to do in the middle of a motion is one that will do it.
EXPRESSIVE_CLIPS: tuple[str, ...] = (
  "dance1_subject1",
  "dance1_subject2",
  "dance1_subject3",
  "dance2_subject1",
  "dance2_subject2",
  "dance2_subject3",
  "dance2_subject4",
  "dance2_subject5",
  "fight1_subject2",
  "fight1_subject3",
  "fight1_subject5",
  "fightAndSports1_subject1",
  "fightAndSports1_subject4",
)

FALL_CLIPS: tuple[str, ...] = (
  "fallAndGetUp1_subject1",
  "fallAndGetUp1_subject4",
  "fallAndGetUp1_subject5",
  "fallAndGetUp2_subject2",
  "fallAndGetUp2_subject3",
  "fallAndGetUp3_subject1",
)

CLIP_SETS: dict[str, tuple[str, ...]] = {
  "locomotion": WALK_CLIPS + RUN_CLIPS + JUMP_CLIPS,
  "walk": WALK_CLIPS,
  "run": RUN_CLIPS,
  "jump": JUMP_CLIPS,
  "expressive": EXPRESSIVE_CLIPS,
  "all": WALK_CLIPS + RUN_CLIPS + JUMP_CLIPS + EXPRESSIVE_CLIPS,
  "everything": WALK_CLIPS + RUN_CLIPS + JUMP_CLIPS + EXPRESSIVE_CLIPS + FALL_CLIPS,
}


@dataclass(frozen=True)
class CorpusCfg:
  """Which clips to fetch and how to convert them."""

  clips: str = "locomotion"
  """A key of CLIP_SETS."""

  raw_dir: Path = Path("data/lafan1/raw")
  """Where the downloaded CSVs are cached."""

  motion_dir: Path = Path("data/lafan1/corpus")
  """Where the converted clips go. Deliberately not `motions`, which holds the
  tracking-pipeline npz written by the other dataset scripts: same clips, different
  contents, and one silently loaded in place of the other is a bad afternoon."""

  fps: float = 50.0
  """Control rate. The clips arrive at 30 Hz and every velocity derived from a reference
  sampled at the wrong rate is wrong by the same factor, so this is not cosmetic."""

  trim_seconds: float = 2.0
  """Dropped from the head of every clip. A LAFAN1 take opens with the subject standing
  in the calibration pose while the capture settles: measured over this corpus, the
  first real movement is 2.8 to 4.1 s in. That stretch is not motion and nothing should
  learn from it."""

  max_seconds: float | None = None
  """Truncate every clip, for a quick run over a small corpus. Applied after the trim."""


@dataclass(frozen=True)
class WindowCfg:
  """Where the holes go."""

  past: int = 25
  """Context frames before the hole. Half a second at 50 Hz. A full gait cycle in this
  corpus runs 0.74 s at a run and 1.84 s at a slow walk, so this is roughly a half
  stride: enough to see a foot plant and to read whether the body is speeding up or
  slowing down. Much less and the context cannot say where in the cycle it is, which is
  most of what the in-betweener has to match."""

  future: int = 25
  """Context frames after it. This is the strip the in-betweener has to arrive at, and it
  is a strip rather than a single frame because a pose alone cannot tell a moving crouch
  from a stationary one. The same half-stride argument applies, and more sharply: this
  one is the target."""

  gap_range: tuple[int, int] = (10, 50)
  """Hole length, in frames. At 50 Hz that is 0.2 s to 1.5 s. Trained across a range even
  though inference will fix one value: varying it costs nothing here and it is what makes
  the duration a working input rather than a decoration."""

  stride: int = 10
  """How far apart window starts are. Neighbouring windows overlap heavily whatever this
  is, which is why the split below is by subject and never by window."""

  min_activity: float = 0.3
  """Mean absolute joint speed over the whole window, in rad/s, below which it is
  dropped. Measured over the hole alone this let through windows whose context was dead
  on both sides, which is the worst kind: the before says nothing about what the body was
  doing and the after is trivially reachable. Over this corpus the median window sits at
  0.74; this default drops the bottom 9%."""

  min_travel: float = 0.3
  """Distance the root actually covers over the window, in metres, below which it is
  dropped. Activity alone passes a subject who stands on the spot and waves their arms,
  which is most of what a mocap take contains between the interesting parts. The median
  window travels 1.07 m and the calibration stretch at the head of a clip travels 0.08 m,
  so this separates them with room to spare while keeping a jump on the spot, whose path
  length counts the vertical."""

  min_height: float = 0.4
  """Root height, in metres, below which the window is dropped. Catches the stretches
  where the subject is on the ground."""

  eval_subjects: tuple[str, ...] = ("subject5",)
  """Held out whole. Splitting by subject rather than by window is the only honest option
  when windows from one clip overlap."""

  seed: int = 0
  """Draws the gap lengths."""


# One frame of a converted clip, ready to be written into a simulator or read as state:
# root position (3), root orientation (4, wxyz), root linear and angular velocity in the
# world (3 + 3), then joint positions and velocities.
ROOT_STATE_DIM = 13


def state_dim(num_joints: int) -> int:
  return ROOT_STATE_DIM + 2 * num_joints


def _yaw_of(quat: torch.Tensor) -> torch.Tensor:
  """Heading of a wxyz quaternion, in radians."""
  w, x, y, z = quat.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def subject_of(clip_name: str) -> str:
  """The subject a clip name belongs to, e.g. walk1_subject5 -> subject5."""
  _, _, subject = clip_name.rpartition("_")
  return subject


@dataclass
class RawMotion:
  """A retargeted clip: where the root is, and what the joints are doing.

  Joint columns are in the model's order, not the source file's.
  """

  root_pos: torch.Tensor  # (T, 3)
  root_quat: torch.Tensor  # (T, 4) wxyz
  joint_pos: torch.Tensor  # (T, J)
  fps: float

  def __len__(self) -> int:
    return int(self.root_pos.shape[0])


def download_clip(
  name: str, raw_dir: Path, repo: str = LAFAN1_REPO, robot: str = LAFAN1_ROBOT
) -> Path:
  """Fetch one retargeted CSV, caching it on disk."""
  destination = raw_dir / f"{name}.csv"
  if destination.exists():
    return destination

  url = f"https://huggingface.co/datasets/{repo}/resolve/main/{robot}/{name}.csv"
  raw_dir.mkdir(parents=True, exist_ok=True)
  print(f"  downloading {name}.csv")
  try:
    # Written to a temporary name first so an interrupted download cannot leave a
    # truncated file behind that a later run would treat as cached.
    partial = destination.with_suffix(".csv.part")
    urllib.request.urlretrieve(url, partial)
    partial.replace(destination)
  except Exception as exc:
    raise RuntimeError(
      f"Could not download {url}: {exc}\n"
      f"If the repository is access-gated, accept its terms on HuggingFace and put the "
      f"{robot}/ folder into {raw_dir} by hand."
    ) from exc
  return destination


def load_csv(path: Path, joint_names: list[str]) -> RawMotion:
  """Read one retargeted CSV into the model's own joint order.

  Each row is root position (3), root quaternion (4, scalar last), then the 29 joint
  angles of LAFAN1_JOINT_NAMES. The quaternion is reordered to mjlab's scalar-first
  convention and renormalized, since interpolating a quaternion that is not quite unit
  length produces a pose that is not quite the recorded one.
  """
  raw = np.loadtxt(path, delimiter=",", dtype=np.float32)
  expected = 7 + len(LAFAN1_JOINT_NAMES)
  if raw.ndim != 2 or raw.shape[1] != expected:
    raise ValueError(
      f"{path.name}: expected {expected} columns (3 root position, 4 root quaternion, "
      f"{len(LAFAN1_JOINT_NAMES)} joints), got "
      f"{raw.shape[-1] if raw.ndim == 2 else raw.shape}."
    )

  values = torch.from_numpy(raw)
  root_pos = values[:, 0:3]
  root_quat = values[:, 3:7][:, [3, 0, 1, 2]]
  root_quat = root_quat / root_quat.norm(dim=-1, keepdim=True)

  source = values[:, 7:]
  joint_pos = torch.zeros(source.shape[0], len(joint_names), dtype=torch.float32)
  for column, name in enumerate(LAFAN1_JOINT_NAMES):
    if name not in joint_names:
      raise ValueError(f"Joint '{name}' is missing from the mjlab G1 model")
    joint_pos[:, joint_names.index(name)] = source[:, column]

  return RawMotion(
    root_pos=root_pos, root_quat=root_quat, joint_pos=joint_pos, fps=SOURCE_FPS
  )


def slerp(q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
  """Shortest-arc interpolation between two batches of quaternions."""
  dot = (q0 * q1).sum(dim=-1, keepdim=True)
  q1 = torch.where(dot < 0.0, -q1, q1)
  dot = dot.abs().clamp(max=1.0)
  theta = torch.acos(dot)
  sin_theta = torch.sin(theta)
  # Nearly parallel quaternions divide by nothing, so fall back to a straight lerp there.
  close = sin_theta < 1e-6
  safe = sin_theta.clamp(min=1e-6)
  w0 = torch.where(close, 1.0 - blend, torch.sin((1.0 - blend) * theta) / safe)
  w1 = torch.where(close, blend, torch.sin(blend * theta) / safe)
  out = w0 * q0 + w1 * q1
  return out / out.norm(dim=-1, keepdim=True)


def resample(motion: RawMotion, output_fps: float) -> RawMotion:
  """Resample to the control rate, lerping positions and slerping the root."""
  frames = len(motion)
  duration = (frames - 1) / motion.fps
  times = torch.arange(0.0, duration, 1.0 / output_fps)
  phase = times / duration
  index_0 = (phase * (frames - 1)).floor().long()
  index_1 = torch.clamp(index_0 + 1, max=frames - 1)
  blend = (phase * (frames - 1) - index_0).unsqueeze(1)

  return RawMotion(
    root_pos=motion.root_pos[index_0] * (1 - blend) + motion.root_pos[index_1] * blend,
    root_quat=slerp(motion.root_quat[index_0], motion.root_quat[index_1], blend),
    joint_pos=motion.joint_pos[index_0] * (1 - blend)
    + motion.joint_pos[index_1] * blend,
    fps=output_fps,
  )


def velocities(
  motion: RawMotion,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Root linear, root angular and joint velocities, by finite difference."""
  dt = 1.0 / motion.fps
  root_lin_vel = torch.gradient(motion.root_pos, spacing=dt, dim=0)[0]
  joint_vel = torch.gradient(motion.joint_pos, spacing=dt, dim=0)[0]
  # Central difference on the rotation group: the relative rotation between the frame
  # before and the frame after, read as an axis-angle and halved.
  q_prev, q_next = motion.root_quat[:-2], motion.root_quat[2:]
  omega = axis_angle_from_quat(quat_mul(q_next, quat_conjugate(q_prev))) / (2.0 * dt)
  root_ang_vel = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
  return root_lin_vel, root_ang_vel, joint_vel


class G1:
  """The compiled model, and the forward kinematics the conversion needs.

  Plain MuJoCo on the CPU rather than a simulation: nothing here is stepped, and all that
  is asked of the model is where the feet are for a given pose.
  """

  def __init__(self) -> None:
    scene_cfg = SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      entities={"robot": get_g1_robot_cfg()},
    )
    self.model = Scene(scene_cfg, device="cpu").compile()
    self.data = mujoco.MjData(self.model)

    # A compiled scene namespaces every element by the entity it came from, so the model
    # calls the knee `robot/left_knee_joint`. The corpus is written against the bare
    # names, which is what the source CSV and every other dataset script use.
    self.joint_names = [
      name.removeprefix(f"{ENTITY_NAME}/")
      for i in range(self.model.njnt)
      if (name := mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i))
      and self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]
    self.num_joints = len(self.joint_names)
    self.foot_ids = [
      mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{ENTITY_NAME}/{name}")
      for name in FOOT_BODY_NAMES
    ]
    if any(i < 0 for i in self.foot_ids):
      raise RuntimeError(f"Feet {FOOT_BODY_NAMES} are not bodies of this G1 model")

    self.standing_foot_height = self._standing_foot_height()

  def _standing_foot_height(self) -> float:
    """Height of the lower ankle roll link with the robot in its default pose.

    The number every clip is aligned against. The retargeting was fitted against somebody
    else's model of the same robot, and a couple of centimetres of mismatch is the
    difference between a stride and a stumble.
    """
    mujoco.mj_resetData(self.model, self.data)
    mujoco.mj_kinematics(self.model, self.data)
    return float(self.data.xpos[self.foot_ids, 2].min())

  def foot_heights(self, motion: RawMotion, stride: int = 1) -> np.ndarray:
    """Height of the lower foot, over every `stride`-th frame of a clip."""
    root_pos = motion.root_pos.numpy()
    root_quat = motion.root_quat.numpy()
    joint_pos = motion.joint_pos.numpy()
    out = []
    for t in range(0, len(motion), stride):
      self.data.qpos[0:3] = root_pos[t]
      self.data.qpos[3:7] = root_quat[t]
      self.data.qpos[7:] = joint_pos[t]
      mujoco.mj_kinematics(self.model, self.data)
      out.append(self.data.xpos[self.foot_ids, 2].min())
    return np.asarray(out, dtype=np.float32)


def convert_clip(
  g1: G1, name: str, csv_path: Path, output_path: Path, cfg: CorpusCfg
) -> dict[str, float | str | int]:
  """One CSV to one npz, resampled, differenced and stood on the floor."""
  motion = load_csv(csv_path, g1.joint_names)
  head = int(cfg.trim_seconds * motion.fps)
  tail = None if cfg.max_seconds is None else head + int(cfg.max_seconds * motion.fps)
  motion = RawMotion(
    root_pos=motion.root_pos[head:tail],
    root_quat=motion.root_quat[head:tail],
    joint_pos=motion.joint_pos[head:tail],
    fps=motion.fps,
  )
  motion = resample(motion, cfg.fps)

  # The median of the lower foot's height over the whole take is this clip's own idea of
  # a planted foot. A long performance is on the ground most of the time, so the median
  # lands in stance whatever else the subject got up to, and it is far more robust than
  # any single frame.
  probe = g1.foot_heights(motion, stride=3)
  shift = float(np.median(probe)) - g1.standing_foot_height
  motion.root_pos[:, 2] -= shift

  root_lin_vel, root_ang_vel, joint_vel = velocities(motion)
  states = torch.cat(
    [
      motion.root_pos,
      motion.root_quat,
      root_lin_vel,
      root_ang_vel,
      motion.joint_pos,
      joint_vel,
    ],
    dim=1,
  )

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    output_path,
    states=states.numpy(),
    fps=np.float32(cfg.fps),
    num_joints=np.int32(g1.num_joints),
    max_seconds=np.float32(-1.0 if cfg.max_seconds is None else cfg.max_seconds),
    trim_seconds=np.float32(cfg.trim_seconds),
    name=name,
  )

  after = g1.foot_heights(motion, stride=3) - g1.standing_foot_height
  speed = root_lin_vel[:, :2].norm(dim=-1)
  return {
    "name": name,
    "file": output_path.name,
    "frames": len(motion),
    "seconds": round(len(motion) / cfg.fps, 2),
    "z_shift": round(shift, 4),
    "stance_error": round(float(np.median(after)), 4),
    "top_speed": round(float(speed.max()), 2),
  }


def _converted_with(path: Path, cfg: CorpusCfg) -> bool:
  """Whether a cached clip was written under the settings asked for now.

  Without this a run that truncated the corpus leaves short clips behind that the next,
  full run happily reuses, and the only symptom is a corpus quietly smaller than the one
  that was asked for.
  """
  try:
    blob = np.load(path)
    max_seconds = -1.0 if cfg.max_seconds is None else cfg.max_seconds
    return bool(
      float(blob["fps"]) == cfg.fps
      and float(blob["max_seconds"]) == max_seconds
      and float(blob["trim_seconds"]) == cfg.trim_seconds
    )
  except (KeyError, OSError, ValueError):
    return False


def build_corpus(cfg: CorpusCfg) -> list[Path]:
  """Fetch and convert every clip of `cfg.clips`, skipping what is already there."""
  if cfg.clips not in CLIP_SETS:
    raise ValueError(f"Unknown clip set '{cfg.clips}'; have {sorted(CLIP_SETS)}.")
  names = CLIP_SETS[cfg.clips]

  g1 = G1()
  print(f"G1: {g1.num_joints} joints, foot stands at {g1.standing_foot_height:.4f} m")

  paths: list[Path] = []
  summaries: list[dict[str, float | str | int]] = []
  for name in names:
    output_path = cfg.motion_dir / f"{name}.npz"
    paths.append(output_path)
    if output_path.exists() and _converted_with(output_path, cfg):
      print(f"{name}: cached")
      continue
    print(f"{name}:")
    csv_path = download_clip(name, cfg.raw_dir)
    summary = convert_clip(g1, name, csv_path, output_path, cfg)
    summaries.append(summary)
    print(
      f"  {summary['frames']} frames ({summary['seconds']} s), "
      f"z shift {summary['z_shift']} m, stance error {summary['stance_error']} m"
    )

  if summaries:
    manifest = cfg.motion_dir / "manifest.json"
    existing = json.loads(manifest.read_text()) if manifest.exists() else []
    by_name = {entry["name"]: entry for entry in existing}
    by_name.update({entry["name"]: entry for entry in summaries})
    manifest.write_text(
      json.dumps(sorted(by_name.values(), key=lambda e: e["name"]), indent=2)
    )

  return paths


class Corpus:
  """Every converted clip, held as one tensor per clip."""

  def __init__(self, paths: list[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
      raise FileNotFoundError(
        f"{len(missing)} motions are not converted yet (first: {missing[0]}). "
        f"Run this module's build step first."
      )

    self.names: list[str] = []
    self.states: list[torch.Tensor] = []
    fps: set[float] = set()
    joints: set[int] = set()
    for path in paths:
      blob = np.load(path)
      if "states" not in blob:
        raise ValueError(
          f"{path} is not a corpus clip (its keys are {sorted(blob)}). The tracking "
          f"pipeline writes npz of the same name holding per-body poses; point "
          f"--corpus.motion-dir somewhere else."
        )
      self.names.append(str(blob["name"]))
      self.states.append(torch.from_numpy(blob["states"]))
      fps.add(float(blob["fps"]))
      joints.add(int(blob["num_joints"]))

    if len(fps) != 1 or len(joints) != 1:
      raise ValueError(f"Corpus is not uniform: fps={fps}, joints={joints}.")
    self.fps = fps.pop()
    self.num_joints = joints.pop()

  def __len__(self) -> int:
    return len(self.names)

  def length(self, clip: int) -> int:
    return int(self.states[clip].shape[0])


@dataclass(frozen=True)
class Window:
  """One question: context, a hole, more context, all inside a single clip."""

  clip: int
  start: int
  past: int
  gap: int
  future: int

  @property
  def hole_start(self) -> int:
    return self.start + self.past

  @property
  def hole_end(self) -> int:
    return self.hole_start + self.gap

  @property
  def end(self) -> int:
    return self.hole_end + self.future

  def __len__(self) -> int:
    return self.past + self.gap + self.future


class BridgeDataset:
  """The corpus cut into windows, split by subject.

  A window is an index range, not a copy: the states stay in the corpus and are sliced on
  demand. That keeps the whole thing small enough to hold in memory whatever the stride
  is, and means changing the cut costs nothing.
  """

  def __init__(self, corpus: Corpus, cfg: WindowCfg, split: str = "train") -> None:
    if split not in ("train", "eval"):
      raise ValueError(f"split is 'train' or 'eval', not '{split}'.")
    self.corpus = corpus
    self.cfg = cfg
    self.split = split
    self.rejected: dict[str, int] = {"idle": 0, "stationary": 0, "grounded": 0}
    self.windows = self._cut()

  def _cut(self) -> list[Window]:
    cfg = self.cfg
    corpus = self.corpus
    generator = torch.Generator().manual_seed(cfg.seed)
    low, high = cfg.gap_range
    windows: list[Window] = []

    for clip in range(len(corpus)):
      held_out = subject_of(corpus.names[clip]) in cfg.eval_subjects
      if held_out != (self.split == "eval"):
        continue

      states = corpus.states[clip]
      length = corpus.length(clip)
      height = states[:, 2]

      for start in range(0, length, cfg.stride):
        gap = int(torch.randint(low, high + 1, (1,), generator=generator))
        window = Window(clip, start, cfg.past, gap, cfg.future)
        if window.end > length:
          continue
        if self.activity(window) < cfg.min_activity:
          self.rejected["idle"] += 1
          continue
        if self.path_length(window) < cfg.min_travel:
          self.rejected["stationary"] += 1
          continue
        if float(height[window.start : window.end].min()) < cfg.min_height:
          self.rejected["grounded"] += 1
          continue
        windows.append(window)

    return windows

  def activity(self, window: Window) -> float:
    """Mean absolute joint speed over the whole window, in rad/s.

    Over the locomotion corpus a walk sits around 0.5 to 0.9 and a run around 1.2 to 1.4,
    against a median window of 0.74. Standing about with the arms drifting comes in under
    0.3, which is what this is for.
    """
    states = self.corpus.states[window.clip]
    joint_vel = states[
      window.start : window.end, ROOT_STATE_DIM + self.corpus.num_joints :
    ]
    return float(joint_vel.abs().mean())

  def path_length(self, window: Window) -> float:
    """How far the root travels over the window, in metres, along its path.

    Path length rather than net displacement, so a window in which the subject turns
    around and comes back still counts as having gone somewhere, and so a jump on the
    spot is credited with its vertical.
    """
    root = self.corpus.states[window.clip][window.start : window.end, :3]
    return float(root.diff(dim=0).norm(dim=1).sum())

  def travel(self, window: Window) -> tuple[float, float]:
    """What the hole itself has to cover: metres of displacement, and degrees of turn.

    Measured between the last context frame before the hole and the first one after, so
    it is exactly the distance the in-betweener is on the hook for. It is not enough to
    produce a plausible stretch of motion: the body has to end up where the after-context
    says it is, which means this displacement has to be part of what the target strip
    says, expressed in the frame of the hand-off rather than the frame of the world.
    """
    states = self.corpus.states[window.clip]
    before, after = states[window.hole_start - 1], states[window.hole_end]
    distance = float((after[:3] - before[:3]).norm())
    turn = float(
      torch.rad2deg(_yaw_of(after[3:7]) - _yaw_of(before[3:7])).remainder(360.0)
    )
    return distance, turn - 360.0 if turn > 180.0 else turn

  def edge_change(self, window: Window) -> float:
    """How much the root's planar velocity differs across the hole, in m/s.

    Not a filter, on purpose. A window where this is near zero is a continuation and one
    where it is large is a change of pace or direction, and the in-betweener needs both.
    It is reported so the mix can be looked at rather than assumed.
    """
    states = self.corpus.states[window.clip]
    before = states[window.hole_start - 1, 7:9]
    after = states[window.hole_end, 7:9]
    return float((after - before).norm())

  def __len__(self) -> int:
    return len(self.windows)

  def states(self, index: int) -> torch.Tensor:
    """The whole window, (frames, 13 + 2J), context and hole together."""
    window = self.windows[index]
    return self.corpus.states[window.clip][window.start : window.end]

  def parts(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The window split the way it is asked at training time: before, hole, after."""
    whole = self.states(index)
    cfg = self.cfg
    window = self.windows[index]
    return (
      whole[: cfg.past],
      whole[cfg.past : cfg.past + window.gap],
      whole[cfg.past + window.gap :],
    )

  def describe(self, index: int) -> str:
    window = self.windows[index]
    return (
      f"{self.corpus.names[window.clip]} "
      f"[{window.start}:{window.end}] gap {window.gap} frames "
      f"({window.gap / self.corpus.fps:.2f} s)"
    )


class DatasetViewer:
  """Plays one window at a time, red while the ghost is inside the hole.

  Almost everything that can go wrong with this corpus goes wrong here rather than in a
  trainer: holes cut where nothing happens, a joint map that moves the wrong limbs, a
  clip that hovers or wades. None of that shows up in a loss curve and all of it is
  obvious in a few seconds of playback.

  The ghost is two full sets of meshes, one green and one red, with only one set visible
  at a time. Colours are baked into the meshes when they are uploaded, so swapping
  visibility is what makes the switch instant.
  """

  def __init__(
    self,
    server,  # viser.ViserServer
    g1: G1,
    dataset: BridgeDataset,
    eval_dataset: BridgeDataset,
  ) -> None:
    self.server = server
    self.g1 = g1
    self.datasets = {"train": dataset, "eval": eval_dataset}
    self.dataset = dataset

    self.index = 0
    self.frame = 0
    self.origin = np.zeros(3, dtype=np.float32)
    self._syncing = False
    self._showing_hole: bool | None = None

    meshes = visual_meshes(self.g1.model)
    self.ghosts = {
      "context": Ghost(server, meshes, "context", CONTEXT_COLOR, visible=True),
      "masked": Ghost(server, meshes, "masked", MASKED_COLOR, visible=False),
    }
    print(f"ghost: {len(meshes)} visual geoms")
    self.server.scene.add_grid("/ground", width=12.0, height=12.0, cell_size=0.5)
    self._build_gui()
    self.load(0)

  def _build_gui(self) -> None:
    gui = self.server.gui

    with gui.add_folder("Dataset"):
      self.dd_split = gui.add_dropdown(
        "Split", options=("train", "eval"), initial_value="train"
      )
      self.sl_index = gui.add_slider("Entry", min=0, max=1, step=1, initial_value=0)
      self.btn_prev = gui.add_button("Previous")
      self.btn_next = gui.add_button("Next")
      self.html_info = gui.add_html("")

      @self.dd_split.on_update
      def _(_) -> None:
        self.dataset = self.datasets[self.dd_split.value]
        self.load(0)

      @self.sl_index.on_update
      def _(_) -> None:
        if not self._syncing:
          self.load(int(self.sl_index.value))

      @self.btn_prev.on_click
      def _(_) -> None:
        self.load(self.index - 1)

      @self.btn_next.on_click
      def _(_) -> None:
        self.load(self.index + 1)

    with gui.add_folder("Playback"):
      self.cb_play = gui.add_checkbox("Play", initial_value=True)
      self.sl_speed = gui.add_slider(
        "Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0
      )
      self.sl_frame = gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)

      @self.sl_frame.on_update
      def _(_) -> None:
        if not self._syncing:
          self.frame = int(self.sl_frame.value)

  def load(self, index: int) -> None:
    """Show entry `index`, wrapping at either end so the buttons never dead-end."""
    total = len(self.dataset)
    if total == 0:
      self.html_info.content = "<p>No windows survived the cut for this split.</p>"
      return

    self.index = index % total
    self.window_states = self.dataset.states(self.index).numpy()
    # Clips wander tens of metres from where they started, so each window is played at
    # the origin. Nothing downstream reads absolute position either.
    self.origin = self.window_states[0, :3].copy()
    self.origin[2] = 0.0
    self.frame = 0

    self._syncing = True
    self.sl_index.max = total - 1
    self.sl_index.value = self.index
    self.sl_frame.max = max(len(self.window_states) - 1, 1)
    self.sl_frame.value = 0
    self._syncing = False

    self._draw_path()
    self.render()

  def _draw_path(self) -> None:
    """The root's whole trajectory over the window, painted the same two colours.

    Without this the playback shows a body moving and says nothing about where it is
    going. With it the window reads as what it is: the ghost leaves one end of a green
    line, crosses the red stretch it has to invent, and has to arrive at the far end.
    """
    window = self.dataset.windows[self.index]
    root = self.window_states[:, :3] - self.origin
    segments = np.stack([root[:-1], root[1:]], axis=1)
    inside = np.array(
      [window.past <= t < window.past + window.gap for t in range(len(root) - 1)]
    )
    # One colour per endpoint of each segment, which is the shape add_line_segments wants.
    per_segment = np.where(
      inside[:, None],
      np.array(MASKED_COLOR, dtype=np.uint8),
      np.array(CONTEXT_COLOR, dtype=np.uint8),
    )
    colors = np.repeat(per_segment[:, None, :], 2, axis=1).astype(np.uint8)
    self.server.scene.add_line_segments(
      "/path", points=segments.astype(np.float32), colors=colors, line_width=3.0
    )

  def render(self) -> None:
    """Pose the ghost at the current frame and paint it."""
    window = self.dataset.windows[self.index]
    state = self.window_states[self.frame]

    data = self.g1.data
    data.qpos[0:3] = state[0:3] - self.origin
    data.qpos[3:7] = state[3:7]
    data.qpos[7:] = state[ROOT_STATE_DIM : ROOT_STATE_DIM + self.g1.num_joints]
    mujoco.mj_kinematics(self.g1.model, data)

    in_hole = window.past <= self.frame < window.past + window.gap

    # Pose the ghost that is about to be shown before showing it, and send the lot in one
    # group, so a colour change never reveals a stale pose. See Ghost.pose.
    with self.server.atomic():
      self.ghosts["masked" if in_hole else "context"].pose(data)
      if in_hole != self._showing_hole:
        self.ghosts["context"].visible = not in_hole
        self.ghosts["masked"].visible = in_hole
        self._showing_hole = in_hole

    self._update_info(in_hole)

  def _update_info(self, in_hole: bool) -> None:
    window = self.dataset.windows[self.index]
    fps = self.dataset.corpus.fps
    total = len(self.window_states)
    distance, turn = self.dataset.travel(window)
    self.html_info.content = (
      '<div style="font-size:0.85em;line-height:1.4;padding:0 1em 0.5em 1em;">'
      f"<b>Entry:</b> {self.index + 1} / {len(self.dataset)}"
      f"<br/><b>Clip:</b> {self.dataset.corpus.names[window.clip]}"
      f"<br/><b>Source frames:</b> {window.start}&ndash;{window.end}"
      f"<br/><b>Layout:</b> {window.past} + <span style='color:#e03a2d'>"
      f"{window.gap}</span> + {window.future} frames"
      f"<br/><b>Gap:</b> {window.gap / fps:.2f} s"
      f"<br/><b>Must cover:</b> {distance:.2f} m, {turn:+.0f}&deg;"
      f"<br/><b>Activity:</b> {self.dataset.activity(window):.2f} rad/s"
      f"<br/><b>Travelled:</b> {self.dataset.path_length(window):.2f} m"
      f"<br/><b>Edge change:</b> {self.dataset.edge_change(window):.2f} m/s"
      f"<br/><b>Showing:</b> {'hole' if in_hole else 'context'} "
      f"({self.frame + 1} / {total})"
      "</div>"
    )

  def step(self, dt: float) -> None:
    """Advance playback by wall-clock `dt` and redraw."""
    if not len(self.window_states):
      return
    if self.cb_play.value:
      advance = dt * self.dataset.corpus.fps * self.sl_speed.value
      self.frame = int(round(self.frame + advance)) % len(self.window_states)
      self._syncing = True
      self.sl_frame.value = self.frame
      self._syncing = False
    self.render()


@dataclass(frozen=True)
class Config:
  corpus: CorpusCfg = field(default_factory=CorpusCfg)
  windows: WindowCfg = field(default_factory=WindowCfg)
  port: int = 8080
  view: bool = True
  """Build the corpus and stop, without opening the viewer, when false."""


def main(cfg: Config) -> None:
  """Build the corpus, cut it into windows, and play the result back."""
  paths = build_corpus(cfg.corpus)
  corpus = Corpus(paths)

  train = BridgeDataset(corpus, cfg.windows, split="train")
  evaluation = BridgeDataset(corpus, cfg.windows, split="eval")
  held_out = sorted(
    {n for n in corpus.names if subject_of(n) in cfg.windows.eval_subjects}
  )
  print(f"\n{len(corpus)} clips at {corpus.fps:g} Hz, {corpus.num_joints} joints")
  for name, dataset in (("train", train), ("eval", evaluation)):
    rejected = dataset.rejected
    offered = len(dataset) + sum(rejected.values())
    kept = max(len(dataset), 1)
    moving = sum(1 for w in dataset.windows if dataset.edge_change(w) > 0.5)
    covered = [dataset.travel(w)[0] for w in dataset.windows] or [0.0]
    print(
      f"{name:<6}{len(dataset)} windows of {offered} (dropped "
      f"{rejected['idle']} idle, {rejected['stationary']} stationary, "
      f"{rejected['grounded']} grounded)\n"
      f"      {moving / kept:.0%} change pace by more than 0.5 m/s; "
      f"the hole has to cover {np.median(covered):.2f} m at the median, "
      f"{np.percentile(covered, 95):.2f} m at the 95th"
    )
  print(f"held out: {', '.join(held_out) or 'nothing'}")
  if not cfg.view:
    return

  import viser

  g1 = G1()
  server = viser.ViserServer(port=cfg.port, label="Bridge corpus")
  viewer = DatasetViewer(server, g1, train, evaluation)

  print(f"\nViewer at http://localhost:{cfg.port} -- Ctrl-C to quit.")
  last = time.time()
  try:
    while True:
      now = time.time()
      viewer.step(now - last)
      last = now
      time.sleep(1.0 / 60.0)
  except KeyboardInterrupt:
    print("\nShutting down.")


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
