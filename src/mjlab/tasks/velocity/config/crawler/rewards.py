"""
A flat dictionary of RewardTermCfg entries, each with a func, a scalar weight, and params.
Positive weights encourage behaviors; negative weights penalize them. Reward tuning is
iterative and messy, so having it on a dedicated file could be useful.

The general rule is: no penalty term should ever produce a per-episode magnitude larger
than the primary reward term's weight. The primary reward is track_linear_velocity
at weight 4.0. Any penalty that routinely produces -10, -40 etc. will always win
the gradient competition.
"""

import math
import torch

from mjlab.envs.mdp import action_rate_l2, joint_vel_l2
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import (
  feet_slip,
  is_terminated,
  self_collision_cost,
  track_angular_velocity,
  track_linear_velocity,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse

from experiments.robots.crawler.constants import (
  CRAWLER_BASE_NAME,
  CRAWLER_FOOT_SITE_NAMES,
)


# This is the only coupling between the two files.
_LEG_PHASE_OFFSETS = torch.tensor([0.0, math.pi, 0.0, math.pi])


def phase_contact_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.05,
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

  # _phase_clock is initialised by gait_phase_clock in observations.py,
  # which always runs before rewards in the RL loop.
  if not hasattr(env, "_phase_clock"):
    return torch.zeros(env.num_envs, device=env.device)

  offsets = _LEG_PHASE_OFFSETS.to(env.device)
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
  base_id = asset.find_bodies(CRAWLER_BASE_NAME)[0]

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
  base_id = asset.find_bodies(CRAWLER_BASE_NAME)[0]

  height = asset.data.body_link_pos_w[:, base_id, 2].squeeze(-1)  # Z coordinate [B]
  height_error_sq = (height - target_height) ** 2

  return torch.exp(-height_error_sq / std ** 2)


def base_stability(env: ManagerBasedRlEnv, std: float = 0.5) -> torch.Tensor:
  """
  Penalize roll and pitch angular velocity.
  Returns values in (0, 1]: 1.0 when still, decaying with wobble.
  Use with a negative curriculum weight so wobble is penalized.
  """

  asset = env.scene["robot"]
  roll_pitch_vel = asset.data.root_link_vel_w[:, 3:5]  # [B, 2]
  wobble_sq = torch.sum(roll_pitch_vel ** 2, dim=1)
  return torch.exp(-wobble_sq / std ** 2)


def nonfeet_ground_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str = "nonfeet_ground_contact",
) -> torch.Tensor:
  """Penalize any non-foot body part touching the terrain."""
  sensor = env.scene.sensors[sensor_name]
  found = sensor.data.found.reshape(env.num_envs, -1)
  return found.float().sum(dim=1)


rewards = {

  # --- Core terms (active from step 0, never disabled) ---

  # Primary task. std=0.25 m/s gives meaningful gradient even at low speeds.
  "track_linear_velocity": RewardTermCfg(
    func=track_linear_velocity,
    weight=5.0,
    params={"command_name": "twist", "std": 0.25},
  ),

  # Structural fix for the standing-still local minimum.
  # Agreement score breaks the body-rocking + single-leg exploit directly:
  # three static feet will never score above 0.25 on this term.
  "phase_contact": RewardTermCfg(
    func=phase_contact_reward,
    weight=2.0,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
    },
  ),

  # Hard constraint. Large enough that no combination of other terms justifies dying.
  "is_terminated": RewardTermCfg(
    func=is_terminated,
    weight=-200.0,
  ),

  # Tiny smoothness floor: prevents high-frequency noise, does not suppress exploration.
  "action_rate_l2": RewardTermCfg(
    func=action_rate_l2,
    weight=-0.02,
  ),

  # --- Phase 2: behavioural constraints (curriculum) ---

  "nonfeet_ground_contact": RewardTermCfg(
    func=nonfeet_ground_contact,
    weight=0.0,
    params={"sensor_name": "nonfeet_ground_contact"},
  ),

  # --- Phase 3: posture (curriculum) ---

  "upright": RewardTermCfg(
    func=flat_orientation,
    weight=0.0,
    params={"std": 0.5},
  ),

  "base_height": RewardTermCfg(
    func=base_height,
    weight=0.0,
    params={"target_height": 0.035, "std": 0.025},
  ),

  "foot_slip": RewardTermCfg(
    func=feet_slip,
    weight=0.0,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", site_names=CRAWLER_FOOT_SITE_NAMES),
    },
  ),

  # --- Phase 4: polish (curriculum) ---

  "base_stability": RewardTermCfg(
    func=base_stability,
    weight=0.0,
    params={"std": 0.5},
  ),

  "joint_vel_l2": RewardTermCfg(
    func=joint_vel_l2,
    weight=0.0,
  ),

  "self_collisions": RewardTermCfg(
    func=self_collision_cost,
    weight=0.0,
    params={"sensor_name": "self_collision", "force_threshold": 2.5},
  ),
}