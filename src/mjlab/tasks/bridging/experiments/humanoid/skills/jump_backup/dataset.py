"""Fetch ASAP's retargeted G1 jump clips and convert them into mjlab motions.

Run:

    uv run --with joblib python -m \
      mjlab.tasks.bridging.experiments.humanoid.skills.jump.dataset

    # the turning and sideways jumps too, which widen the goal space
    uv run --with joblib python -m \
      mjlab.tasks.bridging.experiments.humanoid.skills.jump.dataset --clips all

Raw pickles are cached in data/asap/raw, converted motions land in data/asap/motions, and
both steps skip files already there.

ASAP publishes its motions already retargeted to g1_29dof_anneal_23dof, as joblib pickles
holding a root trajectory plus 23 joint angles per frame. mjlab's tracking machinery wants
per-body world poses and velocities, produced by replaying the motion through the actual
MuJoCo model. This script downloads the few clips this experiment needs and does the
translation.

What happens to each clip, in order:

    0. Download the pickle from ASAP's repository, unless it is already cached.
    1. Scatter the 23 ASAP joints into mjlab's 29-joint G1 by name. The six wrist joints
       ASAP does not have stay at zero.
    2. Rotate and translate so the clip starts at the origin facing +x. Every
       clip was captured with a different world heading, and without this the jump
       direction is a different vector in every file.
    3. Shift the root vertically so the lowest foot over the clip sits exactly at standing
       foot height. The retargeting was fitted against ASAP's own XML, and a couple of
       centimetres of mismatch against mjlab's G1 is the difference between a jump and a
       stumble.
    4. Resample from 30 Hz to the control rate, finite difference the velocities, and
       replay through MuJoCo to log every body's world pose and velocity.
    5. Detect the flight phase from the foot heights and store the jump's goal
       (displacement, turn, apex) alongside the frames.

The resulting npz is a superset of what mjlab.scripts.csv_to_npz writes, so the files also
work with the stock tracking task.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

import mjlab
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_apply,
  quat_conjugate,
  quat_inv,
  quat_mul,
  quat_slerp,
  yaw_quat,
)

# The 23 joints ASAP retargets to, in the order its dof array uses
ASAP_JOINT_NAMES: tuple[str, ...] = (
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
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
)

FOOT_BODY_NAMES: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")

# How high a foot has to be above its standing height before it counts as flight
FLIGHT_FOOT_CLEARANCE = 0.04

# How far the reference may sink below the floor at the deepest frame
MAX_GROUND_PENETRATION = 0.03

# Where ASAP keeps its retargeted G1 clips, and where the raw downloads are cached. Only
# these few files are fetched: the repository is large and none of the rest is used here
ASAP_REPO = "LeCAR-Lab/ASAP"
ASAP_REF = "main"
ASAP_MOTION_PATH = (
  "humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles"
)

DATA_ROOT = Path("data") / "asap"
DEFAULT_CACHE_DIR = DATA_ROOT / "raw"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "motions"

_PREFIX = "0-motions_raw_tairantestbed_smpl_video_"

# The forward jumps are the goal-conditioning axis. level1 is a hop, level5 clears about
# two metres, and everything between is reachable by stretching one of them
FORWARD_CLIPS: dict[str, str] = {
  f"jump_forward_level{i}": f"{_PREFIX}jump_forward_level{i}_filter_amass.pkl"
  for i in range(1, 6)
}

# Optional extras, off by default. They widen the goal space rather than filling it in: the
# turning jumps make the goal's yaw component mean something, the side jumps its lateral
# component. Convert them with --clips all for a jump that is not straight ahead
EXTRA_CLIPS: dict[str, str] = {
  f"jump_degree_level{i}": f"{_PREFIX}jump_degree_level{i}_filter_amass.pkl"
  for i in range(1, 6)
} | {
  f"side_jump_level{i}": f"{_PREFIX}side_jump_level{i}_filter_amass.pkl"
  for i in range(1, 5)
}

CLIP_SETS: dict[str, dict[str, str]] = {
  "forward": FORWARD_CLIPS,
  "all": FORWARD_CLIPS | EXTRA_CLIPS,
}


@dataclass
class RawMotion:
  """An ASAP clip after name mapping, before resampling."""

  root_pos: torch.Tensor  # (T, 3)
  root_quat: torch.Tensor  # (T, 4) wxyz
  joint_pos: torch.Tensor  # (T, num_joints)
  fps: float


def _load_asap_pkl(path: Path, joint_names: list[str], device: str) -> RawMotion:
  # joblib is only needed to read ASAP's pickles, so it stays a local import: the
  # rest of the package must import cleanly without it
  try:
    import joblib
  except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
      "Reading ASAP's motion pickles needs joblib, which is not a project "
      "dependency. Either run this script with joblib injected:\n\n"
      "  uv run --with joblib python -m mjlab.tasks.bridging.experiments.parkour"
      ".jump.dataset\n\n"
      "or add joblib to the project and rerun the plain command."
    ) from exc

  data = joblib.load(path)
  key = next(iter(data))
  clip = data[key]

  root_pos = torch.tensor(np.asarray(clip["root_trans_offset"]), dtype=torch.float32)
  root_quat_xyzw = torch.tensor(np.asarray(clip["root_rot"]), dtype=torch.float32)
  root_quat = root_quat_xyzw[:, [3, 0, 1, 2]]
  root_quat = root_quat / root_quat.norm(dim=-1, keepdim=True)

  dof = torch.tensor(np.asarray(clip["dof"]), dtype=torch.float32)
  if dof.shape[1] != len(ASAP_JOINT_NAMES):
    raise ValueError(
      f"{path.name}: expected {len(ASAP_JOINT_NAMES)} ASAP joints, got {dof.shape[1]}"
    )

  joint_pos = torch.zeros(dof.shape[0], len(joint_names), dtype=torch.float32)
  for src_idx, name in enumerate(ASAP_JOINT_NAMES):
    if name not in joint_names:
      raise ValueError(f"ASAP joint '{name}' is missing from the mjlab G1 model")
    joint_pos[:, joint_names.index(name)] = dof[:, src_idx]

  return RawMotion(
    root_pos=root_pos.to(device),
    root_quat=root_quat.to(device),
    joint_pos=joint_pos.to(device),
    fps=float(clip["fps"]),
  )


def _canonicalize(motion: RawMotion, align: str = "displacement") -> RawMotion:
  """Move the clip to the origin and turn it to face +x.

  Only a rotation about the vertical axis is removed. Pitch and roll of the first frame are
  part of the motion, and heights are left alone.

  Which yaw to remove is a real choice. Removing the pelvis's initial heading is the obvious
  one, and every clip in this set then jumps 11 to 22 degrees off its own +x axis, a
  systematic offset between the retargeted pelvis frame and the direction the subject
  actually travelled. Removing the displacement direction instead makes every clip a jump
  straight down +x, which is what "jump this far forward" should mean, at the cost of the
  robot starting with a small constant yaw offset relative to the jump. Worth taking: the
  goal becomes a distance rather than a distance plus an unexplained sideways component.
  """
  if align == "displacement":
    delta = motion.root_pos[-1, :2] - motion.root_pos[0, :2]
    if float(torch.linalg.norm(delta)) > 0.2:
      theta = torch.atan2(delta[1], delta[0])
      half = -theta / 2.0
      correction = torch.stack(
        [
          torch.cos(half),
          torch.zeros_like(half),
          torch.zeros_like(half),
          torch.sin(half),
        ]
      ).unsqueeze(0)
    else:
      # Too short to have a direction; fall back to the pelvis heading
      correction = quat_inv(yaw_quat(motion.root_quat[0:1]))
  else:
    correction = quat_inv(yaw_quat(motion.root_quat[0:1]))
  correction_seq = correction.expand(motion.root_quat.shape[0], 4)

  origin = motion.root_pos[0].clone()
  origin[2] = 0.0
  root_pos = quat_apply(correction_seq, motion.root_pos - origin)
  root_quat = quat_mul(correction_seq, motion.root_quat)

  return RawMotion(
    root_pos=root_pos,
    root_quat=root_quat,
    joint_pos=motion.joint_pos,
    fps=motion.fps,
  )


def _slerp_sequence(
  quats: torch.Tensor, index_0: torch.Tensor, index_1: torch.Tensor, blend: torch.Tensor
) -> torch.Tensor:
  out = torch.zeros(index_0.shape[0], 4, dtype=quats.dtype, device=quats.device)
  for i in range(index_0.shape[0]):
    out[i] = quat_slerp(quats[index_0[i]], quats[index_1[i]], float(blend[i]))
  return out


def resample(motion: RawMotion, output_fps: float) -> RawMotion:
  """Resample to the control rate, lerping positions and slerping the root."""
  input_frames = motion.root_pos.shape[0]
  duration = (input_frames - 1) / motion.fps
  device = motion.root_pos.device

  times = torch.arange(0.0, duration, 1.0 / output_fps, device=device)
  phase = times / duration
  index_0 = (phase * (input_frames - 1)).floor().long()
  index_1 = torch.clamp(index_0 + 1, max=input_frames - 1)
  blend = (phase * (input_frames - 1) - index_0).unsqueeze(1)

  root_pos = motion.root_pos[index_0] * (1 - blend) + motion.root_pos[index_1] * blend
  joint_pos = (
    motion.joint_pos[index_0] * (1 - blend) + motion.joint_pos[index_1] * blend
  )
  root_quat = _slerp_sequence(motion.root_quat, index_0, index_1, blend.squeeze(1))

  return RawMotion(
    root_pos=root_pos, root_quat=root_quat, joint_pos=joint_pos, fps=output_fps
  )


def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
  """World-frame angular velocity of a quaternion sequence, central differenced."""
  q_prev, q_next = rotations[:-2], rotations[2:]
  q_rel = quat_mul(q_next, quat_conjugate(q_prev))
  omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
  return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def velocities(
  motion: RawMotion,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  dt = 1.0 / motion.fps
  root_lin_vel = torch.gradient(motion.root_pos, spacing=dt, dim=0)[0]
  joint_vel = torch.gradient(motion.joint_pos, spacing=dt, dim=0)[0]
  root_ang_vel = _so3_derivative(motion.root_quat, dt)
  return root_lin_vel, root_ang_vel, joint_vel


def standing_foot_height(robot: Entity, sim: Simulation, scene: Scene) -> float:
  """Height of the ankle roll link when the robot stands in its default pose."""
  robot.write_root_state_to_sim(robot.data.default_root_state.clone())
  robot.write_joint_state_to_sim(
    robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
  )
  sim.forward()
  scene.update(sim.mj_model.opt.timestep)
  foot_ids = robot.find_bodies(list(FOOT_BODY_NAMES), preserve_order=True)[0]
  return float(robot.data.body_link_pos_w[0, foot_ids, 2].min().item())


def replay(
  sim: Simulation,
  scene: Scene,
  robot: Entity,
  motion: RawMotion,
  root_lin_vel: torch.Tensor,
  root_ang_vel: torch.Tensor,
  joint_vel: torch.Tensor,
) -> dict[str, np.ndarray]:
  """Drive the model frame by frame and log every body's world state."""
  log: dict[str, list[np.ndarray]] = {
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }

  for t in range(motion.root_pos.shape[0]):
    root_state = robot.data.default_root_state.clone()
    root_state[:, 0:3] = motion.root_pos[t]
    root_state[:, 0:2] += scene.env_origins[:, :2]
    root_state[:, 3:7] = motion.root_quat[t]
    root_state[:, 7:10] = root_lin_vel[t]
    root_state[:, 10:13] = root_ang_vel[t]
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(
      motion.joint_pos[t].unsqueeze(0), joint_vel[t].unsqueeze(0)
    )

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
    log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
    log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())

  return {k: np.stack(v, axis=0) for k, v in log.items()}


def stance_baseline(foot_height: np.ndarray, frames: int) -> float:
  """The clip's own idea of a planted foot.

  Every clip opens with the subject standing still, so the median foot height over the first
  second is the height a foot has when it is on the ground in this clip. Retargeting error
  means that is not the same number for every clip, so measuring per clip beats assuming any
  of them is exact.
  """
  window = foot_height.min(axis=1)[: max(frames, 1)]
  return float(np.median(window))


def flight_window(foot_height: np.ndarray, baseline: float) -> tuple[int, int] | None:
  """First and last frame of the flight phase, if there is one.

  Of all the stretches where both feet are clear of the ground, the one containing the
  highest foot is the jump. Taking the longest stretch is wrong: retargeting drift leaves
  some clips standing a few centimetres too high for their whole tail, which is a longer
  airborne run than the real flight.
  """
  clearance = foot_height.min(axis=1)
  airborne = clearance > baseline + FLIGHT_FOOT_CLEARANCE
  if not airborne.any():
    return None

  peak = int(np.argmax(clearance))
  start, end = peak, peak
  while start > 0 and airborne[start - 1]:
    start -= 1
  while end < len(airborne) - 1 and airborne[end + 1]:
    end += 1
  return start, end


def yaw_of(quat: np.ndarray) -> float:
  w, x, y, z = quat
  return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def describe_clip(
  log: dict[str, np.ndarray], foot_ids: list[int], baseline: float
) -> dict[str, Any]:
  root_pos = log["body_pos_w"][:, 0]
  root_quat = log["body_quat_w"][:, 0]
  foot_height = log["body_pos_w"][:, foot_ids, 2]

  window = flight_window(foot_height, baseline)
  takeoff = int(window[0]) if window else -1
  land = int(window[1]) if window else -1

  displacement = root_pos[-1, :2] - root_pos[0, :2]
  turn = yaw_of(root_quat[-1]) - yaw_of(root_quat[0])
  turn = float(np.arctan2(np.sin(turn), np.cos(turn)))
  apex = float(root_pos[:, 2].max() - root_pos[0, 2])

  return {
    "goal_xy": displacement.astype(np.float32),
    "goal_yaw": np.float32(turn),
    "goal_apex": np.float32(apex),
    "takeoff_step": np.int64(takeoff),
    "land_step": np.int64(land),
  }


def convert_clip(
  sim: Simulation,
  scene: Scene,
  robot: Entity,
  joint_names: list[str],
  input_path: Path,
  output_path: Path,
  output_fps: float,
  standing_height: float,
  align: str,
) -> dict[str, Any]:
  device = str(sim.device)
  motion = _canonicalize(_load_asap_pkl(input_path, joint_names, device), align)
  motion = resample(motion, output_fps)
  root_lin_vel, root_ang_vel, joint_vel = velocities(motion)

  foot_ids = robot.find_bodies(list(FOOT_BODY_NAMES), preserve_order=True)[0]

  # First pass: find how far off the ground the retargeted clip sits.
  #
  # Aim the *standing* phase at the ground rather than the lowest frame of the whole
  # clip. The lowest frame is the bottom of the landing crouch, which in some clips
  # dips several centimetres below where the same foot sits while standing; anchoring
  # on it leaves the robot hovering through the entire run-up, and the run-up is what
  # decides whether the takeoff is physical.
  #
  # The bound keeps that correction from burying the landing frames in the floor. A
  # reference a couple of centimetres underground is unreachable but cheap; six
  # centimetres of it teaches the policy to slam down
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

  torch.testing.assert_close(
    torch.tensor(log["body_lin_vel_w"][:, 0]),
    root_lin_vel.cpu(),
    rtol=1e-3,
    atol=1e-3,
  )

  jump = describe_clip(log, foot_ids, shifted_baseline)
  payload: dict[str, Any] = {
    "fps": np.array([output_fps], dtype=np.float32),
    **log,
    **jump,
  }

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output_path, **payload)  # ty: ignore[invalid-argument-type]

  distance = float(np.linalg.norm(jump["goal_xy"]))
  summary = {
    "name": output_path.stem,
    "file": output_path.name,
    "frames": int(log["joint_pos"].shape[0]),
    "fps": output_fps,
    "distance": round(distance, 3),
    "goal_xy": [round(float(v), 3) for v in jump["goal_xy"]],
    "goal_yaw": round(float(jump["goal_yaw"]), 3),
    "goal_apex": round(float(jump["goal_apex"]), 3),
    "takeoff_step": int(jump["takeoff_step"]),
    "land_step": int(jump["land_step"]),
    "z_shift": round(z_shift, 4),
    # How far a planted foot still floats, and how far the deepest frame sinks.
    # Both are retargeting residue; watch them, they are the honest measure of how
    # well the reference matches this robot
    "stance_float": round(shifted_baseline - standing_height, 4),
    "ground_penetration": round(
      standing_height - float(log["body_pos_w"][:, foot_ids, 2].min()), 4
    ),
  }
  print(
    f"  {summary['name']:<22} {summary['frames']:>4} frames  "
    f"distance {summary['distance']:.2f} m  apex {summary['goal_apex']:.2f} m  "
    f"flight [{summary['takeoff_step']}, {summary['land_step']}]  "
    f"float {summary['stance_float']:+.3f} m  "
    f"sink {summary['ground_penetration']:+.3f} m"
  )
  return summary


def download_clip(filename: str, cache_dir: Path, ref: str = ASAP_REF) -> Path:
  """Fetch one retargeted pickle from ASAP's repository, caching it on disk.

  Only the handful of clips this experiment uses. Cloning ASAP for them would pull a whole
  framework, its meshes and every other motion, none of which is used here.
  """
  import urllib.request

  destination = cache_dir / filename
  if destination.exists():
    return destination

  url = (
    f"https://raw.githubusercontent.com/{ASAP_REPO}/{ref}/{ASAP_MOTION_PATH}/{filename}"
  )
  cache_dir.mkdir(parents=True, exist_ok=True)
  print(f"  downloading {filename}")
  try:
    # Written to a temporary name first, so an interrupted download cannot leave a
    # truncated file behind that later runs treat as cached
    partial = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, partial)
    partial.replace(destination)
  except Exception as exc:
    raise RuntimeError(f"Could not download {url}: {exc}") from exc
  return destination


def main(
  output_dir: Path = DEFAULT_OUTPUT_DIR,
  cache_dir: Path = DEFAULT_CACHE_DIR,
  output_fps: float = 50.0,
  clips: str = "forward",
  align: str = "displacement",
  ref: str = ASAP_REF,
  device: str = "cuda:0",
) -> None:
  """Download the ASAP jump clips and convert them into mjlab motion npz files.

  Args:
    output_dir: Where the npz files and the manifest are written.
    cache_dir: Where the downloaded ASAP pickles are kept.
    output_fps: Should match the env control rate, 1 / (timestep * decimation).
    clips: Which clip set to convert, "forward" or "all".
    align: Which yaw to remove, "displacement" or "heading". See `_canonicalize`.
    ref: Branch or tag of the ASAP repository to download from.
    device: Torch device for the replay.
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

  manifest = []
  clip_set = CLIP_SETS[clips]
  print(f"Converting {len(clip_set)} clips to {output_dir}:")
  for name, filename in clip_set.items():
    source = download_clip(filename, cache_dir, ref)
    manifest.append(
      convert_clip(
        sim=sim,
        scene=scene,
        robot=robot,
        joint_names=joint_names,
        input_path=source,
        output_path=output_dir / f"{name}.npz",
        output_fps=output_fps,
        standing_height=standing_height,
        align=align,
      )
    )

  manifest.sort(key=lambda entry: entry["distance"])
  manifest_path = output_dir / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2))
  print(f"\nWrote {len(manifest)} motions and {manifest_path}")
  if manifest:
    print(
      f"Goal distance coverage: "
      f"{manifest[0]['distance']:.2f} m to {manifest[-1]['distance']:.2f} m"
    )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
