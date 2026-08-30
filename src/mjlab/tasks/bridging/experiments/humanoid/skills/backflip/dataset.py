"""Build a backflip reference for the G1 and convert it into an mjlab motion.

Run:

    uv run python -m \
      mjlab.tasks.bridging.experiments.humanoid.skills.backflip.dataset

    # a slower flip that travels further back
    uv run python -m \
      mjlab.tasks.bridging.experiments.humanoid.skills.backflip.dataset \
      --flight-s 0.65 --travel -0.5

The motion lands in data/backflip.

Every other skill here tracks a human clip. This one cannot: there is no retargeted G1
backflip in ASAP, in LAFAN1 or in any of the other motion sets the tracking task pulls from,
and a backflip is not something a crop of a walk contains. So the reference is written out
rather than recorded, from a handful of poses.

What that means in practice:

    Poses. Eight keyframes, given as joint angles and interpolated with a smoothstep:
    stand, crouch, extend, tuck, open, land, absorb, stand. Whatever is not named in a
    keyframe stays at the robot's default angle, so the arms and the waist only move where
    the flip needs them to.

    Height. The keyframes say nothing about where the pelvis is. It is measured: the clip is
    replayed once with a nominal height, and each grounded frame is then shifted so its
    lowest foot sits exactly at standing foot height. A crouch that deep with feet on the
    floor is a pelvis height, and this is how it is found rather than guessed.

    Flight. Ballistic between the height at takeoff and the height at touchdown, which the
    measurement above supplies. The rotation is a full turn about the pitch axis at a rate
    that ramps up and back down, so the reference is not asking for angular velocity out of
    nothing at the moment the feet leave the ground.

The result is a kinematically consistent reference, not a recording of a flip that happened.
It says a G1 shaped body could pass through these poses on this trajectory; whether this
robot's actuators can drive it there is what training answers. flight_s is the knob that
decides how hard the question is, since it sets both the takeoff speed and the turn rate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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
  RawMotion,
  describe_clip,
  replay,
  stance_baseline,
  standing_foot_height,
  velocities,
)
from mjlab.terrains import TerrainEntityCfg

MOTION_DIR = Path("data") / "backflip"

# Matches the simulation's own gravity. The flight arc is computed here and tracked there,
# so the two have to agree
GRAVITY = 9.81

# How long each phase on the ground lasts, in seconds. Flight is a command line argument
# because it is the one that decides whether the reference is reachable
STAND_S = 0.40
CROUCH_S = 0.40
LAUNCH_S = 0.16
FLIGHT_S = 0.55
ABSORB_S = 0.30
SETTLE_S = 0.44

# How far back the flip travels, in metres, and how much of the flight is spent ramping the
# turn rate up at one end and down at the other
TRAVEL = -0.35
TURN_EASE = 0.15


@dataclass(frozen=True)
class Keyframe:
  """One pose the flip passes through.

  Joint names are given without a side, so "knee" sets both knees. Anything not named keeps
  the robot's default angle.
  """

  name: str
  time: float
  pose: dict[str, float] = field(default_factory=dict)


def keyframes(flight_s: float) -> tuple[Keyframe, ...]:
  """The flip, as poses and the times they happen at.

  Ankles are held at minus the sum of hip and knee wherever the feet are on the ground,
  which is the angle that keeps a foot flat. This matters at takeoff: pushing off the toes
  would be a better flip and a worse reference, because the height correction below measures
  the ankle link and would read a robot up on its toes as one standing too high.
  """
  t_ready = STAND_S
  t_crouch = t_ready + CROUCH_S
  t_launch = t_crouch + LAUNCH_S
  t_tuck = t_launch + 0.45 * flight_s
  t_open = t_launch + 0.85 * flight_s
  t_land = t_launch + flight_s
  t_absorb = t_land + ABSORB_S
  t_end = t_absorb + SETTLE_S

  return (
    Keyframe("stand", 0.0),
    Keyframe("ready", t_ready),
    Keyframe(
      "crouch",
      t_crouch,
      {
        "hip_pitch": -0.85,
        "knee": 1.70,
        "ankle_pitch": -0.85,
        "shoulder_pitch": 0.90,
        "elbow": 0.30,
        "waist_pitch": 0.20,
      },
    ),
    Keyframe(
      "launch",
      t_launch,
      {
        "hip_pitch": -0.15,
        "knee": 0.25,
        "ankle_pitch": -0.10,
        # Arms thrown overhead. This is where a flip gets the angular momentum it spends
        # on the turn, and leaving it out is what makes a reference look like a hop
        "shoulder_pitch": -2.20,
        "elbow": 0.20,
        "waist_pitch": -0.15,
      },
    ),
    Keyframe(
      "tuck",
      t_tuck,
      {
        "hip_pitch": -1.90,
        "knee": 2.20,
        "ankle_pitch": 0.20,
        "shoulder_pitch": -1.20,
        "elbow": 1.20,
        "waist_pitch": 0.30,
      },
    ),
    Keyframe(
      "open",
      t_open,
      {
        "hip_pitch": -0.55,
        "knee": 0.85,
        "ankle_pitch": -0.30,
        "shoulder_pitch": -0.30,
        "elbow": 0.60,
      },
    ),
    Keyframe(
      "land",
      t_land,
      {
        "hip_pitch": -0.45,
        "knee": 0.80,
        "ankle_pitch": -0.35,
        "shoulder_pitch": 0.10,
      },
    ),
    Keyframe(
      "absorb",
      t_absorb,
      {
        "hip_pitch": -0.80,
        "knee": 1.40,
        "ankle_pitch": -0.60,
        "shoulder_pitch": 0.30,
      },
    ),
    Keyframe("stand", t_end),
  )


def _pose(
  base: np.ndarray, joint_names: list[str], overrides: dict[str, float]
) -> np.ndarray:
  """One keyframe's joint angles, on top of the robot's default pose."""
  pose = base.copy()
  for key, value in overrides.items():
    sided = [f"left_{key}_joint", f"right_{key}_joint"]
    targets = [name for name in sided if name in joint_names] or [f"{key}_joint"]
    for name in targets:
      if name not in joint_names:
        raise ValueError(f"Keyframe joint '{name}' is missing from the mjlab G1 model")
      pose[joint_names.index(name)] = value
  return pose


def joint_trajectory(
  frames: tuple[Keyframe, ...],
  base: np.ndarray,
  joint_names: list[str],
  times: np.ndarray,
) -> np.ndarray:
  """Interpolate the keyframes onto the frame times, easing in and out of each pose."""
  poses = list(
    zip(frames, [_pose(base, joint_names, f.pose) for f in frames], strict=True)
  )
  out = np.zeros((times.shape[0], base.shape[0]), dtype=np.float32)
  for (start, p0), (end, p1) in zip(poses, poses[1:], strict=False):
    segment = (times >= start.time) & (times <= end.time)
    u = ((times[segment] - start.time) / (end.time - start.time))[:, None]
    blend = u * u * (3.0 - 2.0 * u)
    out[segment] = p0 * (1.0 - blend) + p1 * blend
  return out


def pitch_trajectory(
  times: np.ndarray, takeoff: int, land: int, ease: float
) -> np.ndarray:
  """A full backward turn, spread over the flight with a rate that ramps at both ends.

  A constant rate would ask the reference to be turning at eleven radians a second in the
  frame after the feet leave the ground, and stopped in the frame after they touch again.
  The ramp puts the same turn under an angular velocity that starts and ends at zero: a
  trapezoid in `ease`, smoothed so its corners are not steps either.

  The turn is integrated over the closed flight window, so it is exactly zero on the takeoff
  frame and exactly a full circle on the landing one.
  """
  pitch = np.zeros_like(times)
  flight = slice(takeoff, land + 1)

  u = (times[flight] - times[takeoff]) / (times[land] - times[takeoff])
  ramp = np.clip(np.minimum(u / ease, (1.0 - u) / ease), 0.0, 1.0)
  rate = ramp * ramp * (3.0 - 2.0 * ramp)
  progress = np.cumsum(rate)
  progress = progress / progress[-1]

  pitch[flight] = -2.0 * np.pi * progress
  pitch[land:] = -2.0 * np.pi
  return pitch


def root_trajectory(
  times: np.ndarray,
  foot_shift: np.ndarray,
  default_z: float,
  takeoff: int,
  land: int,
  travel: float,
) -> np.ndarray:
  """Where the pelvis is: on the feet while grounded, ballistic while not.

  `foot_shift` comes from the probe replay and is how far each frame has to move for its
  lowest foot to sit on the floor. On the ground that is the answer. In the air the arc is
  the answer, and the two heights the arc has to meet are the grounded ones either side
  of it.

  The arc is the pelvis rather than the centre of mass, which drifts a few centimetres
  around it as the legs tuck and open. Worth the simplification: the pelvis is what the
  reference has to say something about, and the tracking terms compare bodies, not momenta.
  """
  root = np.zeros((times.shape[0], 3), dtype=np.float32)
  root[:, 2] = default_z + foot_shift
  root[land:, 0] = travel

  flight = slice(takeoff + 1, land)
  duration = times[land] - times[takeoff]
  tau = times[flight] - times[takeoff]

  z0, z1 = root[takeoff, 2], root[land, 2]
  climb = (z1 - z0 + 0.5 * GRAVITY * duration**2) / duration

  root[flight, 2] = z0 + climb * tau - 0.5 * GRAVITY * tau**2
  root[flight, 0] = travel * tau / duration
  return root


def quat_from_pitch(pitch: np.ndarray) -> np.ndarray:
  """Rotation about the pitch axis as a wxyz quaternion, one per frame."""
  half = 0.5 * pitch
  quat = np.zeros((pitch.shape[0], 4), dtype=np.float32)
  quat[:, 0] = np.cos(half)
  quat[:, 2] = np.sin(half)
  return quat


def build_reference(
  sim: Simulation,
  scene: Scene,
  robot: Entity,
  joint_names: list[str],
  output_fps: float,
  flight_s: float,
  travel: float,
  standing_height: float,
) -> tuple[RawMotion, int, int]:
  """The two passes: measure the standing heights, then place the flip on top of them."""
  frames = keyframes(flight_s)
  device = str(sim.device)

  num_frames = int(round(frames[-1].time * output_fps)) + 1
  times = np.arange(num_frames, dtype=np.float32) / output_fps
  takeoff = int(round(next(f.time for f in frames if f.name == "launch") * output_fps))
  land = int(round(next(f.time for f in frames if f.name == "land") * output_fps))

  base = robot.data.default_joint_pos[0].cpu().numpy()
  default_z = float(robot.data.default_root_state[0, 2].item())

  joint_pos = joint_trajectory(frames, base, joint_names, times)
  quat = quat_from_pitch(pitch_trajectory(times, takeoff, land, TURN_EASE))

  def to_motion(root_pos: np.ndarray) -> RawMotion:
    return RawMotion(
      root_pos=torch.tensor(root_pos, dtype=torch.float32, device=device),
      root_quat=torch.tensor(quat, dtype=torch.float32, device=device),
      joint_pos=torch.tensor(joint_pos, dtype=torch.float32, device=device),
      fps=output_fps,
    )

  # First pass: hold the pelvis at its standing height and read off where the feet end up.
  # Only forward kinematics runs here, so a frame whose feet are through the floor is fine;
  # it is exactly the number the second pass needs
  nominal = np.zeros((num_frames, 3), dtype=np.float32)
  nominal[:, 2] = default_z
  zeros_root = torch.zeros(num_frames, 3, device=device)
  zeros_joint = torch.zeros(num_frames, len(joint_names), device=device)
  probe = replay(
    sim, scene, robot, to_motion(nominal), zeros_root, zeros_root, zeros_joint
  )

  foot_ids = robot.find_bodies(list(FOOT_BODY_NAMES), preserve_order=True)[0]
  foot_shift = standing_height - probe["body_pos_w"][:, foot_ids, 2].min(axis=1)

  root_pos = root_trajectory(times, foot_shift, default_z, takeoff, land, travel)
  return to_motion(root_pos), takeoff, land


def main(
  output_dir: Path = MOTION_DIR,
  output_fps: float = 50.0,
  flight_s: float = FLIGHT_S,
  travel: float = TRAVEL,
  device: str = "cuda:0",
) -> None:
  """Write the backflip reference as an mjlab motion npz.

  Args:
    output_dir: Where the npz file and the manifest are written.
    output_fps: Should match the env control rate, 1 / (timestep * decimation).
    flight_s: How long the robot is airborne. Longer is a higher, slower flip, and a
      takeoff speed and turn rate the robot may not have.
    travel: How far back the flip goes, in metres. Negative is backwards.
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

  motion, takeoff, land = build_reference(
    sim=sim,
    scene=scene,
    robot=robot,
    joint_names=joint_names,
    output_fps=output_fps,
    flight_s=flight_s,
    travel=travel,
    standing_height=standing_height,
  )
  root_lin_vel, root_ang_vel, joint_vel = velocities(motion)
  log = replay(sim, scene, robot, motion, root_lin_vel, root_ang_vel, joint_vel)

  torch.testing.assert_close(
    torch.tensor(log["body_lin_vel_w"][:, 0]), root_lin_vel.cpu(), rtol=1e-3, atol=1e-3
  )

  foot_ids = robot.find_bodies(list(FOOT_BODY_NAMES), preserve_order=True)[0]
  baseline = stance_baseline(
    log["body_pos_w"][:, foot_ids, 2], int(round(output_fps * STAND_S))
  )
  described = describe_clip(log, foot_ids, baseline)

  # The flight window is known here, so it is written rather than detected. The detector is
  # still run, because a disagreement means the feet did not go where the arc says they did
  detected = (int(described["takeoff_step"]), int(described["land_step"]))
  described["takeoff_step"] = np.int64(takeoff)
  described["land_step"] = np.int64(land)

  payload: dict[str, Any] = {
    "fps": np.array([output_fps], dtype=np.float32),
    **log,
    **described,
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / "backflip.npz"
  np.savez(output_path, **payload)  # ty: ignore[invalid-argument-type]

  summary = {
    "name": output_path.stem,
    "file": output_path.name,
    "frames": int(log["joint_pos"].shape[0]),
    "fps": output_fps,
    "distance": round(float(np.linalg.norm(described["goal_xy"])), 3),
    "goal_xy": [round(float(v), 3) for v in described["goal_xy"]],
    "goal_yaw": round(float(described["goal_yaw"]), 3),
    "goal_apex": round(float(described["goal_apex"]), 3),
    "takeoff_step": takeoff,
    "land_step": land,
    "detected_flight": list(detected),
    "takeoff_speed": round(float(log["body_lin_vel_w"][takeoff + 1, 0, 2]), 3),
    "turn_rate": round(2.0 * np.pi / flight_s, 2),
    "ground_penetration": round(
      standing_height - float(log["body_pos_w"][:, foot_ids, 2].min()), 4
    ),
  }
  (output_dir / "manifest.json").write_text(json.dumps([summary], indent=2))

  print(
    f"  {summary['name']:<10} {summary['frames']:>4} frames  "
    f"flight [{takeoff}, {land}] detected {detected}  "
    f"apex {summary['goal_apex']:+.2f} m  "
    f"takeoff {summary['takeoff_speed']:+.2f} m/s  "
    f"turn {summary['turn_rate']:.1f} rad/s  "
    f"sink {summary['ground_penetration']:+.3f} m"
  )
  print(f"\nWrote {output_path} and its manifest")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
