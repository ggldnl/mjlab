"""Cut a climb out of an OmniRetarget scene and convert it, obstacle included.

One converter for every motion in this package. A motion is a scene, a height scale and a
frame window, listed in MOTIONS below, and adding one is adding a line there.

The obstacle comes out of the same conversion as the clip, which is the point of the file.
OmniRetarget solves the motion against a specific box and preserves the contacts, so the two
are one thing: rotate the clip without rotating the box and the hands stop landing on it.
Everything below moves them together.

What happens to a clip, in order:

    0. Download the archive and the scene's obstacle through
       tracking/scripts/dataset/omniretarget/download.py, unless they are cached.
    1. Slice the frame window and scatter the 29 joint columns into the model's own joint
       order by name. The source is already the canonical Unitree order, checked against
       the g1_29dof.urdf the dataset ships, so this is a rename rather than a remap.
    2. Read the obstacle out of the scene URDF and its meshes. The tallest box is the one
       being climbed, and how tall it is is measured against whatever the robot is standing
       on at the opening frame rather than against the source floor.
    3. Rotate and translate clip and box together, so the clip starts at the origin facing
       +x and the box keeps the pose it had relative to the robot.
    4. Resample to the control rate and prepend a synthetic lead-in: the robot standing
       upright on the clip's own footprint, then leaning forward until it reaches the
       clip's first frame, hands on the box edge. Leg IK keeps both feet planted throughout.
    5. Shift the clip vertically so the opening stance sits at mjlab's standing foot height.
       The capture platform disappears into that shift, and the box is rebuilt on the plane
       at the height measured in step 2, which is a difference of two surfaces and so does
       not carry the shift with it.
    6. Replay through MuJoCo to log every body world pose and velocity, and write the box
       into the manifest for the environment to read.

Why the lead-in exists: the source has no upright frame. The subject already has both hands
on the box edge when the capture starts, torso 35 degrees forward and centre of mass past the
toes, so the first frame is a lean that only holds while the hands carry load. The lead-in
gives the skill a standing start with the box within arm's reach, and the feet never move.
near_face in the printed summary is how much floor there is between the clip origin and the
box.

Run

1. Convert every motion in MOTIONS. Writes to data/omniretarget/clips/<name>.

    uv run python -m mjlab.tasks.bridging.config.g1.skills.climb.dataset

2. Convert one of them.

    uv run python -m mjlab.tasks.bridging.config.g1.skills.climb.dataset --motion climb

3. Convert one against a different window, when the one in MOTIONS opens too early.

    uv run python -m mjlab.tasks.bridging.config.g1.skills.climb.dataset --motion climb --crop-start 40 --crop-end 314
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import tyro

import mjlab
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.bridging.config.g1.skills.jump_continuous.dataset import (
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
from mjlab.tasks.tracking.scripts.datasets.omniretarget import download as omniretarget
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat

# Layout of one row of an OmniRetarget qpos, and the joint order of its last 29 columns.
#
# Quaternion first and wxyz, which is neither of the conventions the other converters here
# read: a Unitree CSV is position first and xyzw. The joint order is the canonical Unitree
# one and was checked against the g1_29dof.urdf the dataset ships
QPOS_COLUMNS = 36
G1_JOINT_ORDER: tuple[str, ...] = (
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

SOURCE_DIR = omniretarget.DEFAULT_DIR
"""Where the downloader caches the archive, the clips and the obstacle models."""

CLIP_DIR = SOURCE_DIR / "clips"
"""Where the converter puts its output, one directory per motion."""


def motion_dir(name: str) -> Path:
  """Where one motion's npz and manifest live."""
  return CLIP_DIR / name


# How long the reference stands upright before it starts leaning, and how long the lean onto
# the box takes
STAND_S = 0.4
LEAN_S = 0.8

# Fraction of the lean the arms move ahead of the torso, so the hands are over the box before
# the weight goes forward
ARM_LEAD = 0.3

LEG_JOINTS: tuple[str, ...] = (
  "hip_pitch",
  "hip_roll",
  "hip_yaw",
  "knee",
  "ankle_pitch",
  "ankle_roll",
)
ARM_JOINT_PATTERN = re.compile(r"shoulder|elbow|wrist")


def lead_in_steps(fps: float, stand_s: float = STAND_S, lean_s: float = LEAN_S) -> int:
  """Frames before the source clip's first frame: the stand plus the lean."""
  return int(round(stand_s * fps)) + int(round(lean_s * fps))


@dataclass(frozen=True)
class Crop:
  """One motion, as a frame window into an OmniRetarget scene.

  Frames are 1-indexed and inclusive, matching the tracking task's cropping tools.
  """

  scene: str
  z_scale: str
  start: int
  end: int

  @property
  def clip(self) -> str:
    return omniretarget.clip_name(self.scene, self.z_scale)


MOTIONS: dict[str, Crop] = {
  # climb_24 is one of the eight scenes in the release that go up, across and down without
  # turning round on top, and the best of those to start on: 0.645 m above the platform the
  # subject stood on, which is the middle of the band the demo asks for. The subject stands
  # still in front of the box from frame 29 to frame 62 and starts the climb at 74, so the
  # window opens late in that standstill and runs to the end of the clip, a stride past the
  # far face
  "climb": Crop("climb_24", z_scale="1.0", start=56, end=314),
}
"""Motion name to the window it is cut from.

Add a motion by adding a line here. It gets a task, its own clip directory and a g1_<name>
log directory, and nothing else has to be said because the environment is the same one for
all of them, obstacle included: the box is read out of the manifest the converter writes.
"""


@dataclass(frozen=True)
class Box:
  """The obstacle, as the environment needs it: a box resting on the plane.

  Position and yaw are in the converted clip's frame, so the robot starts at the origin
  facing +x and this is where the box is from there.
  """

  pos: tuple[float, float, float]
  half_size: tuple[float, float, float]
  yaw: float

  @property
  def height(self) -> float:
    return 2.0 * self.half_size[2]

  def as_dict(self) -> dict[str, Any]:
    return {
      "pos": [round(v, 4) for v in self.pos],
      "half_size": [round(v, 4) for v in self.half_size],
      "yaw": round(self.yaw, 4),
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> Box:
    pos = tuple(float(v) for v in data["pos"])
    half_size = tuple(float(v) for v in data["half_size"])
    return cls(pos=pos, half_size=half_size, yaw=float(data["yaw"]))  # ty: ignore[invalid-argument-type]


NOMINAL_BOX = Box(
  pos=(1.0458, -0.0967, 0.3226),
  half_size=(0.7416, 0.3523, 0.3226),
  yaw=-0.1362,
)
"""What the default crop of the default motion produces.

Read only when a task is built before its clip has been converted, which is every import of
this package on a machine that has not run the converter yet: registration happens at import
and must not fail there. The converter writes the real numbers into the manifest and prints
them, so a changed crop shows up as a printed box that disagrees with this one."""


def load_qpos(
  path: Path,
  crop: Crop,
  joint_names: list[str],
  device: str,
  input_fps: float = SOURCE_FPS,
) -> RawMotion:
  """Read one window of a scene into the model's joint order."""
  with np.load(path) as data:
    qpos = data["qpos"]
    fps = float(data["fps"])

  if qpos.ndim != 2 or qpos.shape[1] != QPOS_COLUMNS:
    raise ValueError(f"{path.name}: expected {QPOS_COLUMNS} columns, got {qpos.shape}")
  if not 1 <= crop.start <= crop.end <= qpos.shape[0]:
    raise ValueError(
      f"{path.name}: window {crop.start}-{crop.end} is outside its {qpos.shape[0]} frames"
    )
  if abs(fps - input_fps) > 1e-6:
    print(f"  [WARN] {path.name} is {fps:g} fps, reading it as {input_fps:g}")

  window = torch.tensor(
    qpos[crop.start - 1 : crop.end], dtype=torch.float32, device=device
  )
  root_quat = window[:, 0:4]
  root_quat = root_quat / root_quat.norm(dim=-1, keepdim=True)
  root_pos = window[:, 4:7]

  joint_pos = torch.zeros(
    window.shape[0], len(joint_names), dtype=torch.float32, device=device
  )
  for column, name in enumerate(G1_JOINT_ORDER):
    if name not in joint_names:
      raise ValueError(f"Joint '{name}' is missing from the mjlab G1 model")
    joint_pos[:, joint_names.index(name)] = window[:, 7 + column]

  return RawMotion(
    root_pos=root_pos, root_quat=root_quat, joint_pos=joint_pos, fps=input_fps
  )


def _base_corners(box: np.ndarray) -> np.ndarray:
  """The four corners of a box mesh's bottom face, in xy."""
  base = box[np.isclose(box[:, 2], box[:, 2].min())][:, :2]
  return np.unique(np.round(base, 5), axis=0)


def _covers(corners: np.ndarray, point: np.ndarray) -> bool:
  """Whether a convex quad's footprint contains a point."""
  centre = corners.mean(axis=0)
  offsets = corners - centre
  loop = corners[np.argsort(np.arctan2(offsets[:, 1], offsets[:, 0]))]
  signs = [
    float(np.cross(loop[(i + 1) % 4] - loop[i], point - loop[i])) for i in range(4)
  ]
  return all(s >= 0.0 for s in signs) or all(s <= 0.0 for s in signs)


def load_obstacle(urdf: Path, standing_xy: np.ndarray) -> tuple[np.ndarray, float]:
  """Read a scene's boxes, pick out the one being climbed and measure how tall it is.

  A scene URDF is a list of meshes with a per-axis scale, and the height variants differ
  only in the Z factor. The tallest box is the obstacle.

  How tall it is has to be measured against the surface the subject was standing on, not
  against the source floor. These captures were shot on a low platform, and some scenes put
  intermediate boxes down to step on, so the surface under the robot at the frame the window
  opens is the only one that means anything. It is also what makes the number survive the
  move into mjlab: the two G1 models do not stand at the same foot height, and a difference
  of two surfaces cancels an offset that applies to the box top and to the floor alike.

  Returns the obstacle's four base corners in xy, and its height above that surface.
  """
  text = urdf.read_text()
  meshes = sorted(set(re.findall(r'mesh filename="([^"]+)" scale="([^"]+)"', text)))
  if not meshes:
    raise ValueError(f"{urdf}: no meshes to read")

  boxes: list[np.ndarray] = []
  for filename, scale in meshes:
    factors = np.array([float(v) for v in scale.split()])
    obj = (urdf.parent / filename).read_text()
    vertices = np.array(
      [
        [float(v) for v in line.split()[1:4]]
        for line in obj.splitlines()
        if line.startswith("v ")
      ]
    )
    boxes.append(vertices * factors)

  tallest = int(np.argmax([box[:, 2].max() for box in boxes]))
  obstacle = boxes[tallest]

  # Whatever the robot is standing on when the window opens, and the source floor at zero
  # when it is standing on nothing
  surface = 0.0
  for index, box in enumerate(boxes):
    if index != tallest and _covers(_base_corners(box), standing_xy):
      surface = max(surface, float(box[:, 2].max()))

  corners = _base_corners(obstacle)
  if corners.shape[0] != 4:
    raise ValueError(
      f"{urdf}: obstacle base has {corners.shape[0]} corners, expected 4"
    )

  return corners, float(obstacle[:, 2].max()) - surface


def canonicalize(
  motion: RawMotion, corners: np.ndarray
) -> tuple[RawMotion, np.ndarray]:
  """Move the clip to the origin facing +x, and take the obstacle with it.

  The heading of the first frame is what is removed, so the clip opens the way every other
  skill in this package opens and the bridge has one convention to hand over into. The
  corners get the same rotation and the same translation, because the only thing about the
  obstacle that matters is where it is relative to the robot, and that must not change.
  """
  quat = motion.root_quat[0]
  heading = float(
    torch.atan2(
      2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
      1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
    )
  )

  correction = quat_inv(yaw_quat(motion.root_quat[0:1]))
  correction_seq = correction.expand(motion.root_quat.shape[0], 4)

  origin = motion.root_pos[0].clone()
  origin[2] = 0.0

  cos, sin = math.cos(-heading), math.sin(-heading)
  rotation = np.array([[cos, -sin], [sin, cos]])
  moved = (corners - origin[:2].cpu().numpy()) @ rotation.T

  return (
    RawMotion(
      root_pos=quat_apply(correction_seq, motion.root_pos - origin),
      root_quat=quat_mul(correction_seq, motion.root_quat),
      joint_pos=motion.joint_pos,
      fps=motion.fps,
    ),
    moved,
  )


FootPose = tuple[np.ndarray, np.ndarray]


class FootPinner:
  """Leg IK on a CPU copy of the robot model. Keeps both feet at a target pose.

  Only the twelve leg joints move. Root pose and every other joint are inputs.
  Damped least squares on the position and orientation of both ankle roll links.
  """

  def __init__(
    self,
    model: mujoco.MjModel,
    robot: Entity,
    iterations: int = 100,
    damping: float = 1e-4,
  ):
    self.model = model
    self.data = mujoco.MjData(model)
    self.iterations = iterations
    self.damping = damping

    self.root_q = robot.indexing.free_joint_q_adr.cpu().numpy()
    self.joint_q = robot.indexing.joint_q_adr.cpu().numpy()
    names = list(robot.joint_names)
    self.leg = np.array(
      [
        names.index(f"{side}_{j}_joint")
        for side in ("left", "right")
        for j in LEG_JOINTS
      ]
    )
    self.leg_dofs = robot.indexing.joint_v_adr.cpu().numpy()[self.leg]
    joint_ids = robot.indexing.joint_ids.cpu().numpy()[self.leg]
    self.leg_range = model.jnt_range[joint_ids]
    local = robot.find_bodies(list(FOOT_BODY_NAMES), preserve_order=True)[0]
    self.feet_ids = [int(robot.indexing.body_ids[i]) for i in local]

  def _forward(
    self, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray
  ) -> None:
    self.data.qpos[self.root_q[:3]] = root_pos
    self.data.qpos[self.root_q[3:]] = root_quat
    self.data.qpos[self.joint_q] = joint_pos
    mujoco.mj_kinematics(self.model, self.data)
    mujoco.mj_comPos(self.model, self.data)

  def feet(
    self, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray
  ) -> list[FootPose]:
    """World pose of both feet, left first."""
    self._forward(root_pos, root_quat, joint_pos)
    return [
      (self.data.xpos[b].copy(), self.data.xquat[b].copy()) for b in self.feet_ids
    ]

  def solve(
    self,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray,
    targets: list[FootPose],
  ) -> np.ndarray:
    """Leg joints that put both feet on targets, starting from joint_pos."""
    q = joint_pos.copy()
    jacp = np.zeros((3, self.model.nv))
    jacr = np.zeros((3, self.model.nv))
    rot_error = np.zeros(3)
    for _ in range(self.iterations):
      self._forward(root_pos, root_quat, q)
      errors, rows = [], []
      for body, (pos, quat) in zip(self.feet_ids, targets, strict=True):
        # subQuat gives the error in the body frame, the jacobian is in the world frame
        mujoco.mju_subQuat(rot_error, quat, self.data.xquat[body])
        errors += [
          pos - self.data.xpos[body],
          self.data.xmat[body].reshape(3, 3) @ rot_error,
        ]
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body)
        rows += [jacp[:, self.leg_dofs].copy(), jacr[:, self.leg_dofs].copy()]
      error = np.concatenate(errors)
      if np.abs(error).max() < 1e-6:
        break
      jac = np.vstack(rows)
      step = jac.T @ np.linalg.solve(
        jac @ jac.T + self.damping * np.eye(jac.shape[0]), error
      )
      q[self.leg] = np.clip(
        q[self.leg] + step, self.leg_range[:, 0], self.leg_range[:, 1]
      )
    return q


def _yaw_only(quat: np.ndarray) -> np.ndarray:
  w, x, y, z = quat
  yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
  dot = float(np.dot(a, b))
  if dot < 0.0:
    b, dot = -b, -dot
  if dot > 0.9995:
    q = a + t * (b - a)
    return q / np.linalg.norm(q)
  theta = math.acos(dot)
  return (math.sin((1.0 - t) * theta) * a + math.sin(t * theta) * b) / math.sin(theta)


def _min_jerk(t: float) -> float:
  """Smooth 0 to 1 ramp with zero velocity and acceleration at both ends."""
  t = min(max(t, 0.0), 1.0)
  return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


def standing_pose(
  pinner: FootPinner,
  feet: list[FootPose],
  heading: np.ndarray,
  default_root_pos: np.ndarray,
  default_joint_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Upright stance with the feet exactly on the given footprint.

  The robot's default knees bent pose, turned to the clip heading and moved over the
  footprint, then leg IK closes the remaining gap. Waist and arms keep their default.

  Returns root position and joint positions. Root orientation is heading.
  """
  identity = np.array([1.0, 0.0, 0.0, 0.0])
  default_feet = pinner.feet(default_root_pos, identity, default_joint_pos)
  pelvis_offset = default_root_pos - np.mean([pos for pos, _ in default_feet], axis=0)
  rotated = np.zeros(3)
  mujoco.mju_rotVecQuat(rotated, pelvis_offset, heading)
  root_pos = np.mean([pos for pos, _ in feet], axis=0) + rotated
  return root_pos, pinner.solve(root_pos, heading, default_joint_pos, feet)


def prepend_lead_in(
  motion: RawMotion,
  pinner: FootPinner,
  joint_names: list[str],
  default_root_pos: np.ndarray,
  default_joint_pos: np.ndarray,
  stand_s: float,
  lean_s: float,
) -> RawMotion:
  """Open the clip standing upright, then lean into its first frame.

  Frames, in order:

    stand   standing_pose on the first frame's footprint, held still
    lean    blend from the stand to the first frame. Arms start first (ARM_LEAD) so the
            hands are over the box before the torso tips. Root pose, waist and legs follow.
            Min jerk easing, so velocities are zero at both ends
    clip    the source, unchanged

  Every lead-in frame goes through leg IK, so the feet stay exactly where the first frame
  has them. A joint space blend alone would slide them.
  """
  stand_frames = int(round(stand_s * motion.fps))
  lean_frames = int(round(lean_s * motion.fps))
  if stand_frames + lean_frames <= 0:
    return motion

  root0 = motion.root_pos[0].cpu().numpy().astype(np.float64)
  quat0 = motion.root_quat[0].cpu().numpy().astype(np.float64)
  joints0 = motion.joint_pos[0].cpu().numpy().astype(np.float64)
  feet = pinner.feet(root0, quat0, joints0)

  heading = _yaw_only(quat0)
  stand_root, stand_joints = standing_pose(
    pinner, feet, heading, default_root_pos, default_joint_pos
  )
  arms = np.array([bool(ARM_JOINT_PATTERN.search(n)) for n in joint_names])

  root_pos, root_quat, joint_pos = [], [], []
  for i in range(stand_frames + lean_frames):
    k = max(i - stand_frames, 0) / max(lean_frames, 1)
    body = _min_jerk((k - ARM_LEAD) / (1.0 - ARM_LEAD))
    arm = _min_jerk(k / (1.0 - ARM_LEAD))

    pos = stand_root + body * (root0 - stand_root)
    quat = _slerp(heading, quat0, body)
    q = stand_joints + body * (joints0 - stand_joints)
    q[arms] = stand_joints[arms] + arm * (joints0[arms] - stand_joints[arms])

    root_pos.append(pos)
    root_quat.append(quat)
    joint_pos.append(pinner.solve(pos, quat, q, feet))

  def prepend(frames: list[np.ndarray], tensor: torch.Tensor) -> torch.Tensor:
    lead = torch.tensor(np.stack(frames), dtype=tensor.dtype, device=tensor.device)
    return torch.cat([lead, tensor], dim=0)

  return RawMotion(
    root_pos=prepend(root_pos, motion.root_pos),
    root_quat=prepend(root_quat, motion.root_quat),
    joint_pos=prepend(joint_pos, motion.joint_pos),
    fps=motion.fps,
  )


def box_from_corners(corners: np.ndarray, height: float) -> Box:
  """Turn four base corners and a height into the box the scene will hold.

  The obstacle is a rectangular prism at some yaw, so the corners give the centre, the two
  half extents and the angle exactly. It rests on the plane, so its centre sits at half its
  height and there is nothing else to say about z.
  """
  centre = corners.mean(axis=0)
  offsets = corners - centre
  ordered = offsets[np.argsort(np.arctan2(offsets[:, 1], offsets[:, 0]))]
  edge_x = ordered[1] - ordered[0]
  edge_y = ordered[2] - ordered[1]

  return Box(
    pos=(float(centre[0]), float(centre[1]), height / 2.0),
    half_size=(
      float(np.linalg.norm(edge_x) / 2.0),
      float(np.linalg.norm(edge_y) / 2.0),
      height / 2.0,
    ),
    yaw=float(np.arctan2(edge_x[1], edge_x[0])),
  )


def near_face(box: Box) -> float:
  """How far ahead of the robot's start the box begins, in metres.

  What a hand-over into this skill has to deliver: the clip opens with the robot standing
  this far from the face it is about to climb.
  """
  cos, sin = math.cos(box.yaw), math.sin(box.yaw)
  rotation = np.array([[cos, -sin], [sin, cos]])
  corners = np.array(
    [
      [sx * box.half_size[0], sy * box.half_size[1]]
      for sx in (-1.0, 1.0)
      for sy in (-1.0, 1.0)
    ]
  )
  return float((corners @ rotation.T + np.array(box.pos[:2]))[:, 0].min())


def box_from_manifest(directory: Path) -> Box:
  """The obstacle the converter measured for one motion, or the nominal one.

  Falls back rather than raising, because a task is registered at import and a checkout that
  has not run the converter yet still has to import.
  """
  manifest = directory / "manifest.json"
  if not manifest.exists():
    return NOMINAL_BOX

  entries = json.loads(manifest.read_text())
  if not entries or "box" not in entries[0]:
    return NOMINAL_BOX
  return Box.from_dict(entries[0]["box"])


def lead_in_from_manifest(directory: Path, fps: float = 50.0) -> int:
  """Frame where the source clip starts, after the stand and the lean.

  Falls back to the default lead-in, for the same reason box_from_manifest does.
  """
  manifest = directory / "manifest.json"
  if manifest.exists():
    entries = json.loads(manifest.read_text())
    if entries and "lead_in_steps" in entries[0]:
      return int(entries[0]["lead_in_steps"])
  return lead_in_steps(fps)


def convert_clip(
  sim: Simulation,
  scene: Scene,
  robot: Entity,
  joint_names: list[str],
  clip_path: Path,
  urdf_path: Path,
  crop: Crop,
  output_path: Path,
  output_fps: float,
  standing_height: float,
  stand_s: float,
  lean_s: float,
  input_fps: float = SOURCE_FPS,
) -> dict[str, Any]:
  # The clip is read before the obstacle, because how tall the obstacle is depends on what
  # the robot is standing on at the frame the window opens
  raw = load_qpos(clip_path, crop, joint_names, str(sim.device), input_fps)
  corners, height = load_obstacle(urdf_path, raw.root_pos[0, :2].cpu().numpy())
  motion, corners = canonicalize(raw, corners)
  pinner = FootPinner(sim.mj_model, robot)
  motion = prepend_lead_in(
    resample(motion, output_fps),
    pinner,
    joint_names,
    default_root_pos=robot.data.default_root_state[0, :3].cpu().numpy(),
    default_joint_pos=robot.data.default_joint_pos[0].cpu().numpy(),
    stand_s=stand_s,
    lean_s=lean_s,
  )
  lead_in = lead_in_steps(output_fps, stand_s, lean_s)
  root_lin_vel, root_ang_vel, joint_vel = velocities(motion)

  foot_ids = robot.find_bodies(list(FOOT_BODY_NAMES), preserve_order=True)[0]

  # First pass: find how far off the ground the retargeted clip sits.
  #
  # The baseline is measured over the lead-in alone, where both feet are pinned to the
  # opening footprint. A climb has no second standing phase to average with: the next time
  # both feet are still they are on top of the box
  probe = replay(sim, scene, robot, motion, root_lin_vel, root_ang_vel, joint_vel)
  probe_foot = probe["body_pos_w"][:, foot_ids, 2]
  stance_frames = max(1, lead_in)
  baseline = stance_baseline(probe_foot, stance_frames)
  z_shift = max(
    standing_height - baseline,
    (standing_height - MAX_GROUND_PENETRATION) - float(probe_foot.min()),
  )
  motion.root_pos[:, 2] += z_shift

  # The obstacle rests on the plane and is as tall above it as it was above the surface the
  # subject stood on. Not top plus the shift: the shift lands the opening stance at mjlab's
  # own standing foot height, which is not the height OmniRetarget's G1 stands at, and
  # carrying that difference onto the box top would leave the reference hovering over it by
  # exactly the amount the two models disagree
  box = box_from_corners(corners, height)

  # Second pass: the real one
  log = replay(sim, scene, robot, motion, root_lin_vel, root_ang_vel, joint_vel)
  shifted_baseline = stance_baseline(log["body_pos_w"][:, foot_ids, 2], stance_frames)

  described = describe_clip(log, foot_ids, shifted_baseline)
  payload: dict[str, Any] = {
    "fps": np.array([output_fps], dtype=np.float32),
    **log,
    **described,
  }

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output_path, **payload)  # ty: ignore[invalid-argument-type]

  root_z = log["body_pos_w"][:, 0, 2]
  lead_in_feet = log["body_pos_w"][: lead_in + 1, foot_ids]
  lead_in_drift = np.linalg.norm(lead_in_feet - lead_in_feet[:1], axis=-1).max()
  summary = {
    "name": output_path.stem,
    "file": output_path.name,
    "scene": crop.scene,
    "z_scale": crop.z_scale,
    "crop": [crop.start, crop.end],
    "frames": int(log["joint_pos"].shape[0]),
    "fps": output_fps,
    "distance": round(float(np.linalg.norm(described["goal_xy"])), 3),
    "box": box.as_dict(),
    # Above the plane, which after the shift is the surface the robot starts on, so this is
    # what it climbs. It is not the box's height in the source scene: that was measured over
    # a capture platform, and the two G1 models do not stand at quite the same foot height
    "box_height": round(box.height, 3),
    "near_face": round(near_face(box), 3),
    "rise": round(float(root_z.max() - root_z[0]), 3),
    "z_shift": round(z_shift, 4),
    "stance_float": round(shifted_baseline - standing_height, 4),
    "ground_penetration": round(
      standing_height - float(log["body_pos_w"][:, foot_ids, 2].min()), 4
    ),
    # Frame where the source clip starts, after the stand and the lean
    "lead_in_steps": lead_in,
    # How far either foot moves during the lead-in. Leg IK should keep this near zero
    "lead_in_foot_drift": round(float(lead_in_drift), 4),
  }
  print(
    f"  {summary['name']:<12} {summary['frames']:>4} frames  "
    f"box {summary['box_height']:.3f} m  near face {summary['near_face']:.2f} m  "
    f"yaw {math.degrees(box.yaw):+.1f} deg  rise {summary['rise']:.2f} m  "
    f"travel {summary['distance']:.2f} m  "
    f"float {summary['stance_float']:+.3f} m  "
    f"sink {summary['ground_penetration']:+.3f} m  "
    f"lead-in {lead_in} frames, foot drift {summary['lead_in_foot_drift']:.4f} m"
  )
  return summary


def convert(
  motions: dict[str, Crop],
  clip_dir: Path,
  source_dir: Path,
  output_fps: float,
  stand_s: float,
  lean_s: float,
  device: str,
  input_fps: float = SOURCE_FPS,
) -> None:
  """Convert OmniRetarget windows into mjlab motion npz files, one directory per motion.

  The scene is built once and every motion converted against it, so converting the whole
  table costs one startup rather than one each. The obstacle is not in that scene: the
  replay is forward kinematics, and what it measures is where the bodies are, not what they
  would have hit.
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

  for name, crop in motions.items():
    output_dir = clip_dir / name
    print(f"Converting {name} from {crop.clip} to {output_dir}:")
    summary = convert_clip(
      sim=sim,
      scene=scene,
      robot=robot,
      joint_names=joint_names,
      clip_path=omniretarget.fetch(crop.clip, source_dir),
      urdf_path=omniretarget.fetch_terrain(crop.scene, crop.z_scale, source_dir),
      crop=crop,
      output_path=output_dir / f"{name}.npz",
      output_fps=output_fps,
      standing_height=standing_height,
      stand_s=stand_s,
      lean_s=lean_s,
      input_fps=input_fps,
    )

    # One motion per directory, so the manifest is one entry and is rewritten whole
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps([summary], indent=2))
    print(f"  wrote {output_dir / (name + '.npz')} and {manifest_path}")


def main(
  motion: str | None = None,
  crop_scene: str | None = None,
  crop_z_scale: str | None = None,
  crop_start: int | None = None,
  crop_end: int | None = None,
  clip_dir: Path = CLIP_DIR,
  source_dir: Path = SOURCE_DIR,
  output_fps: float = 50.0,
  stand_s: float = STAND_S,
  lean_s: float = LEAN_S,
  device: str = "cuda:0",
) -> None:
  """Convert the climbing windows into mjlab motion npz files.

  The four crop arguments override what MOTIONS says about one motion, for trying a window
  out before writing it down. They need a motion, because there is one window per motion and
  nothing to override without one.

  Args:
    motion: Which entry of MOTIONS to convert. Every one of them when left out.
    crop_scene: Scene to cut from, instead of the one MOTIONS lists.
    crop_z_scale: Height variant, instead of the one MOTIONS lists.
    crop_start: First frame, instead of the one MOTIONS lists.
    crop_end: Last frame, instead of the one MOTIONS lists.
    clip_dir: Parent of the per motion output directories.
    source_dir: Where the downloader caches the archive and the obstacle models.
    output_fps: Should match the env control rate, 1 / (timestep * decimation).
    stand_s: How long the reference stands upright before it leans.
    lean_s: How long the lean from standing onto the box edge takes.
    device: Torch device for the replay.
  """
  if motion is None:
    if crop_scene or crop_z_scale or crop_start or crop_end:
      raise ValueError("A crop override needs a --motion to override")
    selected = MOTIONS
  else:
    if motion not in MOTIONS:
      raise ValueError(f"Unknown motion '{motion}'. Known: {', '.join(MOTIONS)}")
    listed = MOTIONS[motion]
    selected = {
      motion: Crop(
        scene=crop_scene or listed.scene,
        z_scale=crop_z_scale or listed.z_scale,
        start=crop_start or listed.start,
        end=crop_end or listed.end,
      )
    }

  convert(selected, clip_dir, source_dir, output_fps, stand_s, lean_s, device)


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
