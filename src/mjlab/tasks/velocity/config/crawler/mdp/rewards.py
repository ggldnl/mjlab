"""
A flat dictionary of RewardTermCfg entries, each with a func, a scalar weight, and params.
Positive weights encourage behaviors; negative weights penalize them.

No penalty term should ever produce a per-episode magnitude larger
than the primary reward term's weight. The primary reward is track_linear_velocity
at weight 5.0. Any penalty that routinely produces -10, -40 etc. will always win
the gradient competition.
"""

import torch

from mjlab.envs.mdp import action_rate_l2, joint_pos_limits
from mjlab.managers import RewardTermCfg, SceneEntityCfg
from mjlab.tasks.velocity.mdp import (
  feet_slip,
  feet_air_time,
  self_collision_cost,
  track_angular_velocity,
  track_linear_velocity,
  variable_posture,
  feet_swing_height,
  soft_landing,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab.asset_zoo.robots.crawler.collisions import BASE_NAME, FOOT_SITE_NAMES
from mjlab.asset_zoo.robots.crawler.actuators import COXA_JOINT_REGEX, FEMUR_JOINT_REGEX, TIBIA_JOINT_REGEX
from mjlab.asset_zoo.robots.crawler.actuators import LEG_PHASE_OFFSETS


def phase_contact_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.02,
) -> torch.Tensor:
  """
  Reward contact state matching the trot clock.
  When cos(phase) > 0 the leg should be in stance; when cos(phase) < 0 it
  should be in swing. Agreement is 1.0 when all four legs match the schedule,
  0.25 when only one matches (e.g. body-rocking with three static feet).
  This directly breaks the standing-still local minimum without any penalty.
  Gated on command magnitude so the robot is not forced to trot in place
  when commanded to stand.
  """

  # _phase_clock is initialized by gait_phase_clock in observations.py,
  # which always runs before rewards in the RL loop.
  if not hasattr(env, "_phase_clock"):
    return torch.zeros(env.num_envs, device=env.device)

  offsets = LEG_PHASE_OFFSETS.to(env.device)
  phases = env._phase_clock.unsqueeze(1) + offsets  # [B, 4]

  # Stance when cos > 0, swing when cos < 0
  desired_contact = (torch.cos(phases) > 0).float()  # [B, 4]

  sensor = env.scene.sensors[sensor_name]
  actual_contact = sensor.data.found.reshape(env.num_envs, -1).float()  # [B, 4]

  agreement = 1.0 - torch.abs(desired_contact - actual_contact).mean(dim=1)  # [B]

  command = env.command_manager.get_command(command_name)
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  scale = (total_command > command_threshold).float()

  return agreement * scale


def flat_orientation(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
  """
  Reward upright base orientation.
  Projects gravity into the body frame: zero x/y components = perfectly upright.
  Returns values in (0, 1]: 1.0 when flat, decaying with tilt.
  """
  asset = env.scene["robot"]
  base_id = asset.find_bodies(BASE_NAME)[0]
  body_quat_w = asset.data.body_link_quat_w[:, base_id, :]  # [B, 4]
  gravity_w = asset.data.gravity_vec_w  # [3]
  projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
  xy_squared = torch.sum(projected_gravity_b[:, :2] ** 2, dim=1)
  return torch.exp(-xy_squared / std**2)


def base_height(env: ManagerBasedRlEnv, target_height: float, std: float) -> torch.Tensor:
  """
  Reward staying close to a target base height.
  Returns values in (0, 1]: 1.0 at target, decaying with distance.
  """
  asset = env.scene["robot"]
  base_id = asset.find_bodies(BASE_NAME)[0]
  height = asset.data.body_link_pos_w[:, base_id, 2].squeeze(-1)  # [B]
  height_error_sq = (height - target_height) ** 2
  return torch.exp(-height_error_sq / std**2)


def base_stability(env: ManagerBasedRlEnv, std: float = 0.5) -> torch.Tensor:
  """
  Penalize roll and pitch angular velocity.
  Returns values in (0, 1]: 1.0 when still, decaying with wobble.
  NOTE: activate only after a stable gait exists; walking naturally
  produces roll/pitch oscillations that this term will fight early
  in training.
  """
  asset = env.scene["robot"]
  roll_pitch_vel = asset.data.root_link_vel_w[:, 3:5]  # wx, wy [B, 2]
  wobble_sq = torch.sum(roll_pitch_vel ** 2, dim=1)
  return torch.exp(-wobble_sq / std**2)


def nonfeet_ground_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str = "legs_ground_contact",
) -> torch.Tensor:
  """Penalize any non-foot body part touching the terrain."""
  sensor = env.scene.sensors[sensor_name]
  found = sensor.data.found.reshape(env.num_envs, -1)
  return found.float().sum(dim=1)

def all_feet_airborne(
        env: ManagerBasedRlEnv,
        sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """
  Penalize simultaneous loss of all ground contact (jumping).
  Returns 1.0 when every foot is in the air, 0.0 when any foot is grounded.
  """
  sensor = env.scene.sensors[sensor_name]
  contact = sensor.data.found.reshape(env.num_envs, -1)  # [B, n_feet]
  any_grounded = contact.any(dim=1).float()
  return 1.0 - any_grounded


rewards = {

  # Primary task

  "track_linear_velocity": RewardTermCfg(
    func=track_linear_velocity,
    weight=5.0,
    params={"command_name": "twist", "std": 0.25},
  ),

  "track_angular_velocity": RewardTermCfg(
    func=track_angular_velocity,
    weight=2.5,
    params={"command_name": "twist", "std": 0.25},
  ),

  "phase_contact": RewardTermCfg(
    func=phase_contact_reward,
    weight=2.0,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
    },
  ),

  # Stability

  "base_stability": RewardTermCfg(
    func=base_stability,
    weight=0.0,  # curriculum
    params={"std": 0.5},
  ),

  "base_height": RewardTermCfg(
    func=base_height,
    weight=0.0,  # curriculum
    params={"target_height": 0.03, "std": 0.25},
  ),

  "upright": RewardTermCfg(
    func=flat_orientation,
    weight=0.0,  # curriculum
    params={"std": 0.5},
  ),
  
  # Pose
  
  "pose": RewardTermCfg(
      func=variable_posture,
      weight=0.5,
      params={
          "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=(
              COXA_JOINT_REGEX,
              FEMUR_JOINT_REGEX,
              TIBIA_JOINT_REGEX
            ),  # (".*",)
          ),
          "command_name": "twist",
          "std_standing": {
              # When standing still: tight tolerance, robot should hold default pose closely.
              COXA_JOINT_REGEX:  0.05,
              FEMUR_JOINT_REGEX: 0.1,
              TIBIA_JOINT_REGEX: 0.1,
          },
          "std_walking": {
              # When walking: open up the tolerance significantly to not fight the gait.
              COXA_JOINT_REGEX:  0.1,
              FEMUR_JOINT_REGEX: 0.3,
              TIBIA_JOINT_REGEX: 0.5,
          },
          "std_running": {
              # At higher speeds allow even more deviation.
              COXA_JOINT_REGEX:  0.25,
              FEMUR_JOINT_REGEX: 0.5,
              TIBIA_JOINT_REGEX: 0.6,
          },
          "walking_threshold": 0.05,   # m/s
          "running_threshold": 0.25,   # m/s
      },
  ),

  # Foot

  # Structural contact guard: prevents dragging limbs on the ground.
  "nonfeet_ground_contact": RewardTermCfg(
    func=nonfeet_ground_contact,
    weight=-1.0,
    params={"sensor_name": "nonfeet_ground_contact"},
  ),

  "foot_slip": RewardTermCfg(
    func=feet_slip,
    weight=0.0,  # curriculum
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.02,
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
    },
  ),

  # Foot clearance: penalize feet that are in swing but not reaching target height.
  # Uses foot_height_scan to measure actual foot height above terrain.
  "foot_swing_height": RewardTermCfg(
    func=feet_swing_height,
    weight=0.0,  # curriculum
    params={
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "foot_height_scan",
      "target_height": 0.02,  # 2 cm
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  ),

  "feet_air_time": RewardTermCfg(
    func=feet_air_time,
    weight=0.0,  # curriculum
    params={
      "sensor_name": "feet_ground_contact",
      "threshold_min": 0.1,
      "threshold_max": 0.5,
      "command_name": "twist",
      "command_threshold": 0.02,
    },
  ),

  # Polish

  "action_rate_l2": RewardTermCfg(
    func=action_rate_l2,
    weight=-0.0,  # curriculum
  ),

  "self_collisions": RewardTermCfg(
    func=self_collision_cost,
    weight=-0.2,
    params={
      "sensor_name": "self_collision",
      "force_threshold": 2.5
    },
  ),

  "dof_pos_limits": RewardTermCfg(
    func=joint_pos_limits,
    weight=-1.0
  ),

  "soft_landing": RewardTermCfg(
    func=soft_landing,
    weight=-1e-5,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  ),
}