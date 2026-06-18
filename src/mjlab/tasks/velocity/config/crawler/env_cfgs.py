"""Crawler velocity environment: gait-guided reinforcement learning (Option 2).

Instead of cloning the open-loop gait, we use it as a dense *reference reward*.
The gait's central pattern generator gives, for each foot, a target position in
the base frame as a closed-form function of ``(phase, vx, vy, wz)`` -- no inverse
kinematics, no precomputed table, fully vectorized on the GPU. The reward is the
closeness of the robot's actual feet to that reference, so PPO learns the joint
actions that make the feet follow the trot (it discovers the IK implicitly,
closed-loop). On top of that it optimizes the real velocity-tracking task, so the
policy can *exceed* the open-loop gait and -- with pushes and domain
randomization during training -- learn the disturbance robustness that pure
imitation cannot give.

Why this works where from-scratch PPO failed: the foot reference is a dense,
phase-locked signal available from the first step, which bootstraps the gait and
sidesteps the exploration dead-ends (standing, spinning) that a sparse
velocity-only reward fell into.

Observations stay deployable: IMU (projected gravity, gyro) + joint encoders, the
command, and a gait clock so the policy can phase-lock to the reference. The
foot-reference reward reads sim foot positions, but those are training-only -- the
policy never observes feet, so it deploys from IMU + encoders alone.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from mjlab.asset_zoo.robots.crawler.actuators import ACTION_SCALE
from mjlab.asset_zoo.robots.crawler.collisions import (
  BASE_NAME,
  FOOT_COLLISION_NAMES,
  FOOT_SITE_NAMES,
)
from mjlab.asset_zoo.robots.crawler.crawler_constants import get_crawler_robot_cfg
from mjlab.asset_zoo.robots.crawler.gait import GaitController, GaitParams
from mjlab.asset_zoo.robots.crawler.sensors import FEET_GROUND, IMU
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg, is_alive
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# Stride frequency of the reference gait (Hz). The gait-clock observation and the
# foot-reference reward both derive their phase from this.
GAIT_FREQUENCY = 2.5

# Command envelope, matched to the gait's verified physical reach.
LIN_VEL_X = (-0.10, 0.10)
LIN_VEL_Y = (-0.05, 0.05)
ANG_VEL_Z = (-0.6, 0.6)


def gait_clock(env: ManagerBasedRlEnv, frequency: float) -> torch.Tensor:
  """Sin/cos of the trot phase, per env, derived from episode time."""
  t = env.episode_length_buf.float() * env.step_dt
  phase = 2.0 * math.pi * frequency * t
  return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)


# Lazily-built, device-keyed reference parameters (nominal foot stance + gait
# shape), read from the GaitController so the reference matches the open-loop gait
# exactly. Built once per device on first reward evaluation (not at import).
_REF_CACHE: dict[str, dict] = {}


def _gait_reference(device: torch.device | str) -> dict:
  key = str(device)
  if key not in _REF_CACHE:
    g = GaitController(GaitParams(frequency=GAIT_FREQUENCY))
    _REF_CACHE[key] = {
      "nominal": torch.tensor(
        g.nominal_foot_base, dtype=torch.float32, device=device
      ),  # (4, 3)
      "offsets": torch.tensor(
        np.asarray(g.params.phase_offsets), dtype=torch.float32, device=device
      ),  # (4,)
      "duty": float(g.params.duty),
      "swing": float(g.params.swing_height),
      "max_stride": float(g.params.max_stride),
      "freq": float(g.params.frequency),
    }
  return _REF_CACHE[key]


def _foot_reference(
  phase: torch.Tensor, command: torch.Tensor, ref: dict
) -> torch.Tensor:
  """Reference foot positions in the base frame, shape (B, 4, 3).

  Vectorized copy of ``GaitController._foot_targets_base``: stance feet sweep from
  +half to -half (pushing the body along +(v + wz x r)); swing feet return with a
  sinusoidal lift.
  """
  nominal = ref["nominal"]  # (4, 3)
  r = nominal[:, :2]  # (4, 2)
  duty, swing, max_stride, freq = (
    ref["duty"],
    ref["swing"],
    ref["max_stride"],
    ref["freq"],
  )
  vx, vy, wz = command[:, 0:1], command[:, 1:2], command[:, 2:3]  # (B, 1)
  reach_x = vx - wz * r[:, 1][None, :]  # (B, 4)
  reach_y = vy + wz * r[:, 0][None, :]  # (B, 4)
  half = torch.stack((reach_x, reach_y), dim=-1) * (duty / (2.0 * freq))  # (B, 4, 2)
  norm = torch.linalg.norm(half, dim=-1, keepdim=True)
  half = half * torch.clamp(max_stride / norm.clamp(min=1e-9), max=1.0)

  s = (phase[:, None] + ref["offsets"][None, :] / (2.0 * math.pi)) % 1.0  # (B, 4)
  stance = s < duty
  prog_st = (s / duty).clamp(0.0, 1.0)
  prog_sw = ((s - duty) / (1.0 - duty)).clamp(0.0, 1.0)
  xy_st = half * (1.0 - 2.0 * prog_st).unsqueeze(-1)
  xy_sw = half * (2.0 * prog_sw - 1.0).unsqueeze(-1)
  xy = torch.where(stance.unsqueeze(-1), xy_st, xy_sw)  # (B, 4, 2)
  z = torch.where(
    stance, torch.zeros_like(s), swing * torch.sin(math.pi * prog_sw)
  )  # (B, 4)
  return nominal[None] + torch.cat((xy, z.unsqueeze(-1)), dim=-1)  # (B, 4, 3)


def gait_foot_tracking(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Dense reward: match the feet to the gait's reference foot trajectory."""
  asset: Entity = env.scene[asset_cfg.name]
  ref = _gait_reference(env.device)
  phase = (env.episode_length_buf.float() * env.step_dt * ref["freq"]) % 1.0
  command = env.command_manager.get_command(command_name)
  assert command is not None
  target = _foot_reference(phase, command[:, :3], ref)  # (B, 4, 3)

  foot_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]  # (B, 4, 3)
  base_pos = asset.data.root_link_pos_w[:, None, :]  # (B, 1, 3)
  base_quat = asset.data.root_link_quat_w[:, None, :].expand(-1, target.shape[1], -1)
  foot_b = quat_apply_inverse(base_quat, foot_w - base_pos)  # (B, 4, 3)

  err = torch.sum((foot_b - target) ** 2, dim=(1, 2))  # (B,)
  return torch.exp(-err / std**2)


def crawler_velocity_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat-terrain crawler env trained by gait-guided RL (foot-reference reward)."""
  cfg = make_velocity_env_cfg()

  ##
  # Simulation: fine timestep for the small, stiff (50 Hz) servos; decimation 10
  # @ 0.002 s keeps 50 Hz control.
  ##
  cfg.sim.nconmax = 35
  cfg.sim.njmax = 300
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.mujoco.timestep = 0.002
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.mujoco.impratio = 10
  cfg.decimation = 10

  ##
  # Robot + flat terrain. IMU for observations, foot-ground contact for slip.
  ##
  cfg.scene.entities = {"robot": get_crawler_robot_cfg()}
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = (*IMU, FEET_GROUND)

  ##
  # Actions: per-joint scale, centered on the neutral stance.
  ##
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = ACTION_SCALE
  joint_pos_action.use_default_offset = True

  ##
  # Observations: deployable sensor set (IMU + encoders) + command + gait clock.
  # No velocimeter, no contact sensing, no height scan. The policy never observes
  # feet -- the foot reference is a reward signal only.
  ##
  terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "twist"}
    ),
    "gait_clock": ObservationTermCfg(
      func=gait_clock, params={"frequency": GAIT_FREQUENCY}
    ),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=dict(terms), concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms=dict(terms), concatenate_terms=True, enable_corruption=False
    ),
  }

  ##
  # Commands.
  ##
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = LIN_VEL_X
  twist.ranges.lin_vel_y = LIN_VEL_Y
  twist.ranges.ang_vel_z = ANG_VEL_Z
  twist.ranges.heading = (-math.pi, math.pi)
  twist.heading_control_stiffness = 0.25
  twist.rel_standing_envs = 0.1
  twist.viz.z_offset = 0.1

  ##
  # Rewards. The foot reference bootstraps the gait; velocity tracking is the
  # task and lets the policy exceed the open-loop gait; the rest keep it upright
  # and smooth.
  ##
  foot_asset = SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES)
  cfg.rewards = {
    "gait_foot_tracking": RewardTermCfg(
      func=gait_foot_tracking,
      weight=2.0,
      params={"std": 0.03, "command_name": "twist", "asset_cfg": foot_asset},
    ),
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": 0.1},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=0.5,
      params={"command_name": "twist", "std": 0.5},
    ),
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=0.5,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_NAME,)),
      },
    ),
    "alive": RewardTermCfg(func=is_alive, weight=0.25),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    # Anti-gaming: the foot reference is in the base frame, so a policy could
    # satisfy it by oscillating/rocking the base and slipping the feet instead of
    # walking. Planting the stance feet (no-slip) makes the reference's backward
    # foot sweep translate the base, and the flat-orientation penalty kills the
    # rocking. Together they turn "follow the reference" into real locomotion.
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-2.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.02,
        "asset_cfg": foot_asset,
      },
    ),
    "flat_orientation": RewardTermCfg(
      func=mdp.flat_orientation_l2,
      weight=-0.5,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_NAME,))},
    ),
  }

  ##
  # Events: resets + disturbances/domain randomization. Unlike the open-loop gait,
  # RL can learn to reject these, which is the whole point of training a policy.
  ##
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.3, 0.3),
    "y": (-0.3, 0.3),
    "z": (0.0, 0.01),
    "yaw": (-math.pi, math.pi),
  }
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)
  cfg.events["push_robot"].interval_range_s = (2.0, 5.0)
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.1, 0.1),
    "y": (-0.1, 0.1),
    "z": (-0.05, 0.05),
    "roll": (-0.2, 0.2),
    "pitch": (-0.2, 0.2),
    "yaw": (-0.3, 0.3),
  }
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_COLLISION_NAMES
  cfg.events["foot_friction"].params["ranges"] = (0.6, 1.4)
  cfg.events["base_com"].params["asset_cfg"].body_names = (BASE_NAME,)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.005, 0.005),
    1: (-0.005, 0.005),
    2: (-0.005, 0.005),
  }

  ##
  # Terminations: time-out + tipping on flat ground.
  ##
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(60.0)

  cfg.curriculum = {}

  ##
  # Viewer: close in on the small robot.
  ##
  cfg.viewer.body_name = BASE_NAME
  cfg.viewer.distance = 0.6
  cfg.viewer.elevation = -10.0
  cfg.viewer.azimuth = 90.0

  cfg.episode_length_s = 20.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg
