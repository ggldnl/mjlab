"""Cut a martial arts motion out of a LAFAN1 fight performance and convert it.

One converter for every motion in this package. A motion is a name and a frame window into
a performance, listed in MOTIONS below, and adding one is adding a line there.

LAFAN1 (Harvey et al., SIGGRAPH 2020) has a fight category, and Unitree publishes the whole
set retargeted to the 29 joint G1 as one CSV per performance: root position, root
quaternion in xyzw, then the joint angles, at 30 Hz. That is the same file the tracking
task's cropping tools read, so a motion here is a frame range into one of those
performances.

Source CSVs are cached in data/lafan1_g1 by the tracking task's LAFAN1 downloader, which
this calls rather than fetching anything itself. Each converted motion lands in
data/lafan1_g1/clips/<name>. Both steps skip files already there.

What happens to a clip, in order:

    0. Download the performance through datasets/lafan1/download.py, unless it is cached.
    1. Slice the frame window and scatter the CSV joint columns into the model's own joint
       order by name.
    2. Rotate and translate so the clip starts at the origin facing +x. The pelvis heading
       is what is removed here, not the direction of travel, because these motions go
       nowhere and have no direction of travel.
    3. Resample to the control rate and hold the first frame still for half a second, so
       the clip starts from a standstill the policy has to launch out of.
    4. Shift the root vertically so a planted foot sits at standing foot height.
    5. Replay through MuJoCo to log every body world pose and velocity, and record where
       the clip ends up.

Check a frame interval using the tracking task's manual_crop.py.

Run

1. Convert every motion in MOTIONS.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.martial.dataset

2. Convert one of them.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.martial.dataset --motion front_kick

3. Convert one against a different window, when the one in MOTIONS lands on the wrong
   moment.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.martial.dataset --motion front_kick --crop-start 1180 --crop-end 1300

4. Pick a window by eye first.

    uv run python src/mjlab/tasks/tracking/scripts/datasets/lafan1/interactive_crop.py --data-dir data/lafan1_g1 --motion fight1_subject2
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import tyro

import mjlab
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
from mjlab.tasks.tracking.scripts.datasets.lafan1 import download as lafan1
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

SOURCE_DIR = lafan1.DEFAULT_DIR
"""Where the LAFAN1 downloader caches its performances.

The download is that script's job, not this one's, so every task that crops a performance
reads the same copy and none of them carries a URL. Point --source-dir at a local copy if
the release ever moves."""

CLIP_DIR = SOURCE_DIR / "clips"
"""Where the converter puts its output, one directory per clip."""


def motion_dir(name: str) -> Path:
  """Where one motion's npz and manifest live."""
  return CLIP_DIR / name


# How long the reference stands still before the strike begins. Long enough that the policy
# has to hold a stance and then break out of it, short enough not to spend the episode on it
STILL_HOLD_S = 0.5


@dataclass(frozen=True)
class Crop:
  """One motion, as a frame window into a LAFAN1 performance.

  Frames are 1-indexed and inclusive, matching the tracking task's cropping tools.
  """

  source: str
  start: int
  end: int


MOTIONS: dict[str, Crop] = {
  # 121 frames, four seconds at 30 Hz: long enough to hold a stance, kick and recover,
  # short enough that the subject has not moved on to something else
  "front_kick": Crop("fight1_subject2", start=1200, end=1320),
  # 131 frames, four and a half seconds. Longer than the front kick because a combination
  # is several strikes, and the window has to reach the end of the last one
  "punch_combo": Crop("fight1_subject2", start=1600, end=1730),
}
"""Motion name to the window it is cut from.

Add a motion by adding a line here. It gets a task, its own clip directory and a g1_<name>
log directory, and nothing else has to be said because the environment is the same one for
all of them.
"""


def load_csv(
  path: Path,
  crop: Crop,
  joint_names: list[str],
  device: str,
  input_fps: float = SOURCE_FPS,
) -> RawMotion:
  """Read one window of a performance into the model's joint order.

  input_fps is the rate the CSV was written at. LAFAN1's retargeted set is 30 Hz and that
  is the default, but the column layout is not LAFAN1's, it is Unitree's, so a clip
  retargeted from anywhere else reads here too and may well be at another rate. Getting it
  wrong does not fail, it stretches the motion: the resample below reads this as the input
  rate, so a 50 Hz clip declared as 30 comes out five thirds too slow.
  """
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
    root_pos=root_pos, root_quat=root_quat, joint_pos=joint_pos, fps=input_fps
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
  input_fps: float = SOURCE_FPS,
) -> dict[str, Any]:
  motion = canonicalize(
    load_csv(input_path, crop, joint_names, str(sim.device), input_fps)
  )
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
  motions: dict[str, Crop],
  clip_dir: Path,
  source_dir: Path,
  output_fps: float,
  hold_s: float,
  device: str,
  input_fps: float = SOURCE_FPS,
  fetch: Callable[[str, Path], Path] = lafan1.fetch,
) -> None:
  """Convert LAFAN1 windows into mjlab motion npz files, one directory per motion.

  This is the whole of the package's dataset script: bring up a one env scene, measure the
  model's standing foot height, and convert every motion asked for against it. The scene is
  built once, so converting the whole table costs one startup rather than one each.

  input_fps and fetch are the two seams a task with a different source uses. The columns of
  a Unitree CSV are the same wherever it came from, so the only things that change are the
  rate it was written at and where the file comes from. The kick retargets its own clip and
  passes both; every motion in this package takes the defaults and is unaffected.
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
    print(f"Converting {name} to {output_dir}:")
    summary = convert_clip(
      sim=sim,
      scene=scene,
      robot=robot,
      joint_names=joint_names,
      input_path=fetch(crop.source, source_dir),
      crop=crop,
      output_path=output_dir / f"{name}.npz",
      output_fps=output_fps,
      standing_height=standing_height,
      hold_s=hold_s,
      input_fps=input_fps,
    )

    # One motion per directory, so the manifest is one entry and is rewritten whole
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps([summary], indent=2))
    print(f"  wrote {output_dir / (name + '.npz')} and {manifest_path}")


def main(
  motion: str | None = None,
  crop_source: str | None = None,
  crop_start: int | None = None,
  crop_end: int | None = None,
  clip_dir: Path = CLIP_DIR,
  source_dir: Path = SOURCE_DIR,
  output_fps: float = 50.0,
  hold_s: float = STILL_HOLD_S,
  device: str = "cuda:0",
) -> None:
  """Convert the martial arts windows into mjlab motion npz files.

  The three crop arguments override what MOTIONS says about one motion, for trying a
  window out before writing it down. They need a motion, because there is one window per
  motion and nothing to override without one.

  Args:
    motion: Which entry of MOTIONS to convert. Every one of them when left out.
    crop_source: Performance to cut from, instead of the one MOTIONS lists.
    crop_start: First frame, instead of the one MOTIONS lists.
    crop_end: Last frame, instead of the one MOTIONS lists.
    clip_dir: Parent of the per motion output directories.
    source_dir: Where the downloaded LAFAN1 CSVs are kept.
    output_fps: Should match the env control rate, 1 / (timestep * decimation).
    hold_s: How long the reference stands still before the motion starts.
    device: Torch device for the replay.
  """
  if motion is None:
    if crop_source or crop_start or crop_end:
      raise ValueError("A crop override needs a --motion to override")
    selected = MOTIONS
  else:
    if motion not in MOTIONS:
      raise ValueError(f"Unknown motion '{motion}'. Known: {', '.join(MOTIONS)}")
    listed = MOTIONS[motion]
    selected = {
      motion: Crop(
        source=crop_source or listed.source,
        start=crop_start or listed.start,
        end=crop_end or listed.end,
      )
    }

  convert(selected, clip_dir, source_dir, output_fps, hold_s, device)


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
