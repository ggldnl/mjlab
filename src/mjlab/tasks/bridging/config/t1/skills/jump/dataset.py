"""Retarget ASAP's G1 jump clips onto the T1 and convert them into mjlab motions.

ASAP publishes its five forward jumps retargeted to the G1 only. The G1 jump trains on them
(see g1/skills/jump_continuous/dataset.py), and this puts the same five jumps on the T1.

What happens to each clip:

    0. Convert the G1 clip through the G1 dataset, unless data/asap/motions has it.
    1. Retarget onto the T1 frame by frame with inverse kinematics (mink, the solver GMR
       uses). Hips, knees and feet go where the G1's are, scaled by the ratio of the two
       leg lengths. Trunk and waist turn like the G1's torso and pelvis. Arms follow
       loosely, since the T1's are longer.
    2. The G1 conversion on the result: put the standing foot on the ground, replay through
       MuJoCo, store the jump goal.

Converted motions land in data/asap_t1/motions, in the G1 npz format, so the shared jump
command reads them unchanged.

Run

1. Retarget and convert. joblib reads ASAP's pickles when the G1 clips are not there yet.

    uv run --with joblib python -m mjlab.tasks.bridging.config.t1.skills.jump.dataset
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mink
import mujoco
import numpy as np
import torch
import tyro

import mjlab
from mjlab.asset_zoo.robots import get_g1_robot_cfg, get_t1_robot_cfg
from mjlab.asset_zoo.robots.booster_t1.t1_constants import HOME_QPOS
from mjlab.asset_zoo.robots.booster_t1.t1_constants import get_spec as get_t1_spec
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec as get_g1_spec
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.scripts import csv_to_npz
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.bridging.config.g1.skills.jump_continuous import dataset as g1_dataset
from mjlab.tasks.bridging.config.g1.skills.jump_continuous.dataset import (
  MAX_GROUND_PENETRATION,
  RawMotion,
  describe_clip,
  replay,
  stance_baseline,
  velocities,
)
from mjlab.terrains import TerrainEntityCfg

FOOT_BODY_NAMES: tuple[str, str] = ("left_foot_link", "right_foot_link")

# G1 body, T1 body, position cost, orientation cost. Every pair has the same frame at the
# zero pose on both robots, so a G1 orientation is a T1 one as is
TARGETS: tuple[tuple[str, str, float, float], ...] = (
  ("left_hip_pitch_link", "Hip_Pitch_Left", 5.0, 0.0),
  ("right_hip_pitch_link", "Hip_Pitch_Right", 5.0, 0.0),
  ("pelvis", "Waist", 0.0, 2.0),
  ("torso_link", "Trunk", 0.0, 2.0),
  ("left_knee_link", "Shank_Left", 3.0, 0.0),
  ("right_knee_link", "Shank_Right", 3.0, 0.0),
  ("left_ankle_roll_link", "left_foot_link", 10.0, 2.0),
  ("right_ankle_roll_link", "right_foot_link", 10.0, 2.0),
  ("left_shoulder_roll_link", "AL2", 0.5, 0.0),
  ("right_shoulder_roll_link", "AR2", 0.5, 0.0),
  ("left_elbow_link", "left_hand_link", 0.5, 0.0),
  ("right_elbow_link", "right_hand_link", 0.5, 0.0),
  ("left_wrist_yaw_link", "left_hand_sphere_link", 0.3, 0.0),
  ("right_wrist_yaw_link", "right_hand_sphere_link", 0.3, 0.0),
)

IK_DT = 0.05
IK_ITERATIONS = 10
IK_FIRST_FRAME_ITERATIONS = 200

DEFAULT_OUTPUT_DIR = Path("data") / "asap_t1" / "motions"


def _zero_pose(model: mujoco.MjModel) -> mujoco.MjData:
  data = mujoco.MjData(model)
  data.qpos[3] = 1.0
  mujoco.mj_kinematics(model, data)
  return data


def _body_z(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
  return float(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name), 2])


def scale_and_lift(g1: mujoco.MjModel, t1: mujoco.MjModel) -> tuple[float, float]:
  """How G1 positions map onto the T1: z_t1 = scale * z_g1 + lift, and xy scaled.

  The scale is the ratio of hip to ankle heights at the zero pose. The lift puts the T1's
  ankle where the scaled G1 ankle is, since the two sit at different heights over the sole.
  """
  g, t = _zero_pose(g1), _zero_pose(t1)
  g1_leg = _body_z(g1, g, "left_hip_pitch_link") - _body_z(
    g1, g, "left_ankle_roll_link"
  )
  t1_leg = _body_z(t1, t, "Hip_Pitch_Left") - _body_z(t1, t, "left_foot_link")
  scale = t1_leg / g1_leg
  lift = _body_z(t1, t, "left_foot_link") - scale * _body_z(
    g1, g, "left_ankle_roll_link"
  )
  return scale, lift


def retarget(clip: dict[str, np.ndarray], g1_bodies: list[str]) -> RawMotion:
  """Solve the T1 pose for every frame of a converted G1 clip."""
  g1_model = get_g1_spec().compile()
  model = get_t1_spec().compile()
  scale, lift = scale_and_lift(g1_model, model)

  configuration = mink.Configuration(model)
  data = configuration.data
  for name, value in HOME_QPOS.items():
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[joint]] = value
  data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
  data.qpos[2] = 0.7
  configuration.update(data.qpos)

  tasks = []
  frame_tasks = []
  for g1_name, t1_name, pos_cost, ori_cost in TARGETS:
    task = mink.FrameTask(
      t1_name, "body", position_cost=pos_cost, orientation_cost=ori_cost
    )
    frame_tasks.append((g1_bodies.index(g1_name), task))
    tasks.append(task)
  posture = mink.PostureTask(model, cost=1e-2)
  posture.set_target(data.qpos.copy())
  tasks.append(posture)
  limits = [mink.ConfigurationLimit(model)]

  joint_names = csv_to_npz.robot_joint_names("booster_t1")
  qpos_adr = [
    int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
    for n in joint_names
  ]

  pos_w, quat_w = clip["body_pos_w"], clip["body_quat_w"]
  frames = len(pos_w)
  root_pos = np.zeros((frames, 3))
  root_quat = np.zeros((frames, 4))
  joint_pos = np.zeros((frames, len(joint_names)))
  for t in range(frames):
    for body, task in frame_tasks:
      pos = pos_w[t, body] * scale
      pos[2] += lift
      task.set_target(
        mink.SE3.from_rotation_and_translation(mink.SO3(quat_w[t, body]), pos)
      )
    if t == 0:
      # Start from standing over the clip's first hip position
      configuration.data.qpos[:2] = pos_w[0, 0, :2] * scale
      configuration.update(configuration.data.qpos)
    for _ in range(IK_FIRST_FRAME_ITERATIONS if t == 0 else IK_ITERATIONS):
      velocity = mink.solve_ik(
        configuration, tasks, IK_DT, solver="daqp", damping=1e-3, limits=limits
      )
      configuration.integrate_inplace(velocity, IK_DT)
    q = configuration.q
    root_pos[t], root_quat[t], joint_pos[t] = q[0:3], q[3:7], q[qpos_adr]

  fps = float(np.asarray(clip["fps"]).reshape(-1)[0])
  return RawMotion(
    root_pos=torch.tensor(root_pos, dtype=torch.float32),
    root_quat=torch.tensor(root_quat, dtype=torch.float32),
    joint_pos=torch.tensor(joint_pos, dtype=torch.float32),
    fps=fps,
  )


def standing_foot_height(robot: Entity, sim: Simulation, scene: Scene) -> float:
  """Height of the foot link when the T1 stands in its default pose."""
  robot.write_root_state_to_sim(robot.data.default_root_state.clone())
  robot.write_joint_state_to_sim(
    robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
  )
  sim.forward()
  scene.update(sim.mj_model.opt.timestep)
  foot_ids = robot.find_bodies(list(FOOT_BODY_NAMES), preserve_order=True)[0]
  return float(robot.data.body_link_pos_w[0, foot_ids, 2].min().item())


def convert_clip(
  sim: Simulation,
  scene: Scene,
  robot: Entity,
  motion: RawMotion,
  output_path: Path,
  standing_height: float,
) -> dict[str, Any]:
  """The G1 convert_clip from step 3 on, with the T1's feet."""
  device = str(sim.device)
  motion = RawMotion(
    motion.root_pos.to(device),
    motion.root_quat.to(device),
    motion.joint_pos.to(device),
    motion.fps,
  )
  root_lin_vel, root_ang_vel, joint_vel = velocities(motion)
  foot_ids = robot.find_bodies(list(FOOT_BODY_NAMES), preserve_order=True)[0]

  # Same grounding as the G1: the stand on the floor, the deepest frame no further than
  # MAX_GROUND_PENETRATION into it. See the G1 dataset for why
  probe = replay(sim, scene, robot, motion, root_lin_vel, root_ang_vel, joint_vel)
  probe_foot = probe["body_pos_w"][:, foot_ids, 2]
  stance_frames = int(round(motion.fps))
  baseline = stance_baseline(probe_foot, stance_frames)
  z_shift = max(
    standing_height - baseline,
    (standing_height - MAX_GROUND_PENETRATION) - float(probe_foot.min()),
  )
  motion.root_pos[:, 2] += z_shift

  log = replay(sim, scene, robot, motion, root_lin_vel, root_ang_vel, joint_vel)
  foot = log["body_pos_w"][:, foot_ids, 2]
  shifted_baseline = stance_baseline(foot, stance_frames)
  jump = describe_clip(log, foot_ids, shifted_baseline)

  payload: dict[str, Any] = {
    "fps": np.array([motion.fps], dtype=np.float32),
    **log,
    **jump,
  }
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output_path, **payload)  # ty: ignore[invalid-argument-type]

  summary = {
    "name": output_path.stem,
    "file": output_path.name,
    "frames": int(log["joint_pos"].shape[0]),
    "fps": motion.fps,
    "distance": round(float(np.linalg.norm(jump["goal_xy"])), 3),
    "goal_xy": [round(float(v), 3) for v in jump["goal_xy"]],
    "goal_yaw": round(float(jump["goal_yaw"]), 3),
    "goal_apex": round(float(jump["goal_apex"]), 3),
    "takeoff_step": int(jump["takeoff_step"]),
    "land_step": int(jump["land_step"]),
    "z_shift": round(z_shift, 4),
    "stance_float": round(shifted_baseline - standing_height, 4),
    "ground_penetration": round(standing_height - float(foot.min()), 4),
  }
  print(
    f"  {summary['name']:<22} {summary['frames']:>4} frames  "
    f"distance {summary['distance']:.2f} m  apex {summary['goal_apex']:.2f} m  "
    f"flight [{summary['takeoff_step']}, {summary['land_step']}]  "
    f"float {summary['stance_float']:+.3f} m  sink {summary['ground_penetration']:+.3f} m"
  )
  return summary


def main(
  output_dir: Path = DEFAULT_OUTPUT_DIR,
  g1_dir: Path = g1_dataset.DEFAULT_OUTPUT_DIR,
  clips: str = "forward",
  device: str = "cuda:0",
) -> None:
  """Retarget the G1 jump clips onto the T1 and convert them.

  Args:
    output_dir: Where the T1 npz files and the manifest are written.
    g1_dir: Where the converted G1 clips are, written there first when missing.
    clips: Which ASAP clip set, "forward" or "all". See the G1 dataset.
    device: Torch device for the replay.
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARN] CUDA unavailable, falling back to CPU.")
    device = "cpu"

  names = g1_dataset.CLIP_SETS[clips]
  missing = [n for n in names if not (g1_dir / f"{n}.npz").is_file()]
  if missing:
    g1_dataset.convert(tuple(missing), output_dir=g1_dir, device=device)

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / 50.0
  scene = Scene(
    SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      entities={"robot": get_t1_robot_cfg()},
    ),
    device=device,
  )
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=scene.compile(), device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  scene.reset()
  robot: Entity = scene["robot"]
  standing_height = standing_foot_height(robot, sim, scene)
  print(f"Standing foot height in mjlab's T1: {standing_height:.4f} m")

  g1_bodies = list(Entity(get_g1_robot_cfg()).body_names)
  manifest = []
  print(f"Retargeting {len(names)} clips to {output_dir}:")
  for name in names:
    clip = dict(np.load(g1_dir / f"{name}.npz"))
    motion = retarget(clip, g1_bodies)
    manifest.append(
      convert_clip(
        sim, scene, robot, motion, output_dir / f"{name}.npz", standing_height
      )
    )

  manifest.sort(key=lambda entry: entry["distance"])
  (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
  print(f"\nWrote {len(manifest)} motions and {output_dir / 'manifest.json'}")
  print(
    f"Goal distance coverage: "
    f"{manifest[0]['distance']:.2f} m to {manifest[-1]['distance']:.2f} m"
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
