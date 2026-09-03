"""Cut a strike out of a LAFAN1 fight performance and convert it to an mjlab motion.

LAFAN1 (Harvey et al., SIGGRAPH 2020) has a fight category, and Unitree publishes the whole
set retargeted to the 29 joint G1 as one CSV per performance: root position, root
quaternion in xyzw, then the joint angles, at 30 Hz. That is the same file the tracking
task's cropping tools read, so a strike here is a frame range into one of those
performances.

Source CSVs are cached in data/lafan1_g1, the converted motion lands in
data/lafan1_g1/front_kick, and both steps skip files already there.

What happens to the clip, in order:

    0. Download the performance, unless it is already cached.
    1. Slice the frame window and scatter the CSV joint columns into the model's own joint
       order by name.
    2. Rotate and translate so the clip starts at the origin facing +x. The pelvis heading
       is what is removed here, not the direction of travel, because a strike goes nowhere
       and has no direction of travel.
    3. Resample to the control rate and hold the first frame still for half a second, so
       the clip starts from a standstill the policy has to launch out of.
    4. Shift the root vertically so a planted foot sits at standing foot height.
    5. Replay through MuJoCo to log every body world pose and velocity, and record where
       the clip ends up.

Check the frame interval using the tracking task's manual_crop.py.

Run

1. Convert the default window.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.front_kick.dataset

2. Convert a different window, when the default lands on the wrong strike.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.front_kick.dataset       --crop.start 1180 --crop.end 1300

3. Pick a window by eye first.

    uv run python src/mjlab/tasks/tracking/scripts/datasets/lafan1/interactive_crop.py       --data-dir data/lafan1_g1 --motion fight1_subject2
"""

from __future__ import annotations

import json
import urllib.request
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
from mjlab.tasks.bridging.experiments.humanoid.skills.jump.dataset import (
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
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat

# Column layout of a Unitree generalized coordinate CSV, and the joint order of its last 29
# columns. Column 7 + k is the joint named k below, whatever order the compiled model uses
CSV_COLUMNS = 36
CSV_JOINT_NAMES: tuple[str, ...] = (
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

# Where the retargeted performances come from. Point --source-dir at a local copy if this
# ever moves; nothing else here depends on the download
LAFAN1_URL = (
  "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset"
  "/resolve/main/g1/{name}.csv"
)

SOURCE_DIR = Path("data") / "lafan1_g1"
CLIP_DIR = SOURCE_DIR / "clips"
"""Where the converter puts its output, one directory per clip."""

MOTION_DIR = CLIP_DIR / "front_kick"

# How long the reference stands still before the strike begins. Long enough that the policy
# has to hold a stance and then break out of it, short enough not to spend the episode on it
STILL_HOLD_S = 0.5


@dataclass(frozen=True)
class Crop:
  """One strike, as a frame window into a LAFAN1 performance.

  Frames are 1-indexed and inclusive, matching the tracking task's cropping tools.
  """

  source: str
  start: int
  end: int


# 121 frames, four seconds at 30 Hz: long enough to hold a stance, kick and recover, short
# enough that the subject has not moved on to something else
CLIP = Crop("fight1_subject2", start=1200, end=1320)


def download_clip(name: str, source_dir: Path) -> Path:
  """Fetch one retargeted performance, caching it on disk."""
  destination = source_dir / f"{name}.csv"
  if destination.exists():
    return destination

  source_dir.mkdir(parents=True, exist_ok=True)
  url = LAFAN1_URL.format(name=name)
  print(f"  downloading {name}.csv")
  try:
    # Written to a temporary name first, so an interrupted download cannot leave a
    # truncated file behind that later runs treat as cached
    partial = destination.with_suffix(".csv.part")
    urllib.request.urlretrieve(url, partial)
    partial.replace(destination)
  except Exception as exc:
    raise RuntimeError(f"Could not download {url}: {exc}") from exc
  return destination


def load_csv(path: Path, crop: Crop, joint_names: list[str], device: str) -> RawMotion:
  """Read one window of a performance into the model's joint order."""
  rows = np.loadtxt(
    path,
    delimiter=",",
    skiprows=crop.start - 1,
    max_rows=crop.end - crop.start + 1,
  )
  if rows.ndim != 2 or rows.shape[1] != CSV_COLUMNS:
    raise ValueError(f"{path.name}: expected {CSV_COLUMNS} columns, got {rows.shape}")

  motion = torch.tensor(rows, dtype=torch.float32, device=device)
  root_pos = motion[:, 0:3]
  root_quat = motion[:, 3:7][:, [3, 0, 1, 2]]  # xyzw to wxyz
  root_quat = root_quat / root_quat.norm(dim=-1, keepdim=True)

  joint_pos = torch.zeros(
    motion.shape[0], len(joint_names), dtype=torch.float32, device=device
  )
  for column, name in enumerate(CSV_JOINT_NAMES):
    if name not in joint_names:
      raise ValueError(f"Joint '{name}' is missing from the mjlab G1 model")
    joint_pos[:, joint_names.index(name)] = motion[:, 7 + column]

  return RawMotion(
    root_pos=root_pos, root_quat=root_quat, joint_pos=joint_pos, fps=SOURCE_FPS
  )


def canonicalize(motion: RawMotion) -> RawMotion:
  """Move the clip to the origin and turn its opening stance to face +x.

  The pelvis heading of the first frame is what is removed. The jump's converter removes
  the direction of travel instead, because there the goal is a displacement and the clip
  has to point along it. A strike stays where it is, so the only heading it has is the one
  it starts in, and that is the one the policy is asked to hold.
  """
  correction = quat_inv(yaw_quat(motion.root_quat[0:1]))
  correction_seq = correction.expand(motion.root_quat.shape[0], 4)

  origin = motion.root_pos[0].clone()
  origin[2] = 0.0

  return RawMotion(
    root_pos=quat_apply(correction_seq, motion.root_pos - origin),
    root_quat=quat_mul(correction_seq, motion.root_quat),
    joint_pos=motion.joint_pos,
    fps=motion.fps,
  )


def prepend_hold(motion: RawMotion, seconds: float) -> RawMotion:
  """Repeat the first frame, so the clip opens from a standstill.

  A crop out of a fight performance starts mid-bounce, with the subject already moving. The
  skill is meant to begin from a robot that is standing there, so the reference is given a
  stretch of held pose to begin with. Velocities are finite differenced after this, so the
  held frames carry no motion and the strike gets one frame of ramp into it.
  """
  frames = int(round(seconds * motion.fps))
  if frames <= 0:
    return motion

  def hold(tensor: torch.Tensor) -> torch.Tensor:
    return torch.cat([tensor[0:1].expand(frames, *tensor.shape[1:]), tensor], dim=0)

  return RawMotion(
    root_pos=hold(motion.root_pos),
    root_quat=hold(motion.root_quat),
    joint_pos=hold(motion.joint_pos),
    fps=motion.fps,
  )


def convert_clip(
  sim: Simulation,
  scene: Scene,
  robot: Entity,
  joint_names: list[str],
  input_path: Path,
  crop: Crop,
  output_path: Path,
  output_fps: float,
  standing_height: float,
  hold_s: float,
) -> dict[str, Any]:
  motion = canonicalize(load_csv(input_path, crop, joint_names, str(sim.device)))
  motion = prepend_hold(resample(motion, output_fps), hold_s)
  root_lin_vel, root_ang_vel, joint_vel = velocities(motion)

  foot_ids = robot.find_bodies(list(FOOT_BODY_NAMES), preserve_order=True)[0]

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

  described = describe_clip(log, foot_ids, shifted_baseline)
  payload: dict[str, Any] = {
    "fps": np.array([output_fps], dtype=np.float32),
    **log,
    **described,
  }

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output_path, **payload)  # ty: ignore[invalid-argument-type]

  drift = float(np.linalg.norm(described["goal_xy"]))
  summary = {
    "name": output_path.stem,
    "file": output_path.name,
    "frames": int(log["joint_pos"].shape[0]),
    "fps": output_fps,
    "distance": round(drift, 3),
    "goal_xy": [round(float(v), 3) for v in described["goal_xy"]],
    "goal_yaw": round(float(described["goal_yaw"]), 3),
    "goal_apex": round(float(described["goal_apex"]), 3),
    "takeoff_step": int(described["takeoff_step"]),
    "land_step": int(described["land_step"]),
    "z_shift": round(z_shift, 4),
    "stance_float": round(shifted_baseline - standing_height, 4),
    "ground_penetration": round(
      standing_height - float(log["body_pos_w"][:, foot_ids, 2].min()), 4
    ),
  }
  print(
    f"  {summary['name']:<16} {summary['frames']:>4} frames  "
    f"drift {summary['distance']:.2f} m  turn {summary['goal_yaw']:+.2f} rad  "
    f"rise {summary['goal_apex']:+.2f} m  "
    f"float {summary['stance_float']:+.3f} m  "
    f"sink {summary['ground_penetration']:+.3f} m"
  )
  return summary


def convert(
  name: str,
  crop: Crop,
  output_dir: Path,
  source_dir: Path,
  output_fps: float,
  hold_s: float,
  device: str,
) -> None:
  """Convert one LAFAN1 window into an mjlab motion npz, next to a manifest.

  This is the whole of a strike task's dataset script: bring up a one env scene, measure the
  model's standing foot height, convert, and write the manifest the motion library reads.
  Both strike tasks call it with a different window.
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

  print(f"Converting {name} to {output_dir}:")
  source = download_clip(crop.source, source_dir)
  summary = convert_clip(
    sim=sim,
    scene=scene,
    robot=robot,
    joint_names=joint_names,
    input_path=source,
    crop=crop,
    output_path=output_dir / f"{name}.npz",
    output_fps=output_fps,
    standing_height=standing_height,
    hold_s=hold_s,
  )

  manifest_path = output_dir / "manifest.json"
  manifest_path.write_text(json.dumps([summary], indent=2))
  print(f"\nWrote {output_dir / (name + '.npz')} and {manifest_path}")


def main(
  output_dir: Path = MOTION_DIR,
  source_dir: Path = SOURCE_DIR,
  crop: Crop = CLIP,
  output_fps: float = 50.0,
  hold_s: float = STILL_HOLD_S,
  device: str = "cuda:0",
) -> None:
  """Convert the front kick window into an mjlab motion npz file.

  Args:
    output_dir: Where the npz file and the manifest are written.
    source_dir: Where the downloaded LAFAN1 CSVs are kept.
    crop: The frame window to cut. Move it if the default lands on the wrong strike.
    output_fps: Should match the env control rate, 1 / (timestep * decimation).
    hold_s: How long the reference stands still before the kick.
    device: Torch device for the replay.
  """
  convert("front_kick", crop, output_dir, source_dir, output_fps, hold_s, device)


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
