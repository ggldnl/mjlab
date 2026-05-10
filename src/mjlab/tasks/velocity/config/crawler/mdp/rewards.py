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
from mjlab.managers import RewardTermCfg
from mjlab.tasks.velocity.mdp import (
  feet_air_time,
  feet_swing_height,
  self_collision_cost,
  track_angular_velocity,
  track_linear_velocity,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab.asset_zoo.robots.crawler.collisions import BASE_NAME


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


rewards = {

  # Primary task.
  "track_linear_velocity": RewardTermCfg(
    func=track_linear_velocity,
    weight=3.0,
    params={"command_name": "twist", "std": 0.25},
  ),

  "track_angular_velocity": RewardTermCfg(
    func=track_angular_velocity,
    weight=1.5,
    params={"command_name": "twist", "std": 0.25},
  ),

  # Stability

  "base_stability": RewardTermCfg(
    func=base_stability,
    weight=0.2,
    params={"std": 0.5},
  ),

  "upright": RewardTermCfg(
    func=flat_orientation,
    weight=0.2,
    params={"std": 0.5},
  ),

  # Foot

  # Structural contact guard: prevents dragging limbs on the ground.
  "nonfeet_ground_contact": RewardTermCfg(
    func=nonfeet_ground_contact,
    weight=-0.5,
    params={"sensor_name": "legs_ground_contact"},
  ),

  # Gait structure: reward lifting feet off the ground long enough to constitute a stride.
  # threshold_min=0.1s avoids rewarding micro-hops; threshold_max=1.0s avoids rewarding
  # standing on three legs.
  "feet_air_time": RewardTermCfg(
    func=feet_air_time,
    weight=0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "threshold_min": 0.1,
      "threshold_max": 0.5,
      "command_name": "twist",
      "command_threshold": 0.05,
    },
  ),

  # Polish

  "dof_pos_limits": RewardTermCfg(
    func=joint_pos_limits,
    weight=-1.0,
  ),

  "self_collisions": RewardTermCfg(
    func=self_collision_cost,
    weight=-0.1,
    params={"sensor_name": "self_collision", "force_threshold": 2.5},
  ),
}

"""
"action_rate_l2": RewardTermCfg(
  func=action_rate_l2,
  weight=-0.05,
),
  
# Foot clearance: penalize feet that are in swing but not reaching target height.
# Uses foot_height_scan to measure actual foot height above terrain.
"foot_swing_height": RewardTermCfg(
  func=feet_swing_height,
  weight=-0.5,
  params={
    "sensor_name": "feet_ground_contact",
    "height_sensor_name": "foot_height_scan",
    "target_height": 0.01,
    "command_name": "twist",
    "command_threshold": 0.05,
  },
),

"is_terminated": RewardTermCfg(
  func=is_terminated,
  weight=0.0,
),

"joint_vel_l2": RewardTermCfg(
  func=joint_vel_l2,
  weight=0.0,
),

"foot_slip": RewardTermCfg(
  func=feet_slip,
  weight=-0.1,
  params={
    "sensor_name": "feet_ground_contact",
    "command_name": "twist",
    "command_threshold": 0.05,
    "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
  },
),
"""