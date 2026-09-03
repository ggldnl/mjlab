"""Reward terms for the jump.

The tracking terms mjlab ships (anchor position and orientation, relative body position and
orientation, body velocities) cover most of ASAP's reward set and are used directly from the
env config. Added here:

    joint-space tracking     ASAP's teleop_joint_position and teleop_joint_velocity
    the feet terms           what decides whether a landing holds
    the goal term            where the jump has to end up
    landing impact           stops the robot reaching the target by falling onto it
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import JumpCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _cmd(env: ManagerBasedRlEnv, command_name: str) -> JumpCommand:
  return cast(JumpCommand, env.command_manager.get_term(command_name))


##
# Joint-space tracking. ASAP's teleop_joint_position and teleop_joint_velocity.
##


def motion_joint_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _cmd(env, command_name)
  error = torch.square(command.joint_pos - command.robot_joint_pos).mean(dim=-1)
  return torch.exp(-error / std**2)


def motion_joint_velocity_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _cmd(env, command_name)
  error = torch.square(command.joint_vel - command.robot_joint_vel).mean(dim=-1)
  return torch.exp(-error / std**2)


##
# The goal.
##


def jump_goal_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Reward landing where the goal said, paid from touchdown onward.

  Zero before touchdown. The jump is not over, and paying for proximity mid-flight would
  reward drifting toward the target rather than jumping to it.
  """
  command = _cmd(env, command_name)
  error = torch.square(command.goal_pos_error)
  return torch.exp(-error / std**2) * command.has_landed.float()


def jump_goal_success(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  """A sparse bonus for a landing inside the tolerance, once the clip is over.

  Paid on the final step only, so it is a per-episode bonus rather than something the policy
  can farm by sitting at the target.
  """
  command = _cmd(env, command_name)
  return ((command.goal_pos_error < threshold) & command.motion_done).float()


##
# Feet.
##


def feet_orientation_penalty(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Penalize feet that are not level.

  A humanoid landing on the edge of a foot does not stay standing. ASAP found this term
  necessary for exactly the phase this task is about.
  """
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
  gravity = asset.data.gravity_vec_w[:, None, :].expand(-1, quat.shape[1], -1)
  projected = quat_apply_inverse(quat, gravity)
  return torch.sum(torch.square(projected[..., :2]), dim=(-1, -2))


def feet_slip_penalty(
  env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Penalize horizontal foot velocity while the foot is loaded.

  Unlike the velocity task's version, not gated on a commanded speed. The reference decides
  when the feet should move, and a foot in contact should never slide whatever the jump is
  doing.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  in_contact = (sensor.data.found > 0).float()
  foot_vel_xy = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
  return torch.sum(torch.square(foot_vel_xy).sum(-1) * in_contact, dim=1)


def landing_impact_penalty(
  env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float = 400.0
) -> torch.Tensor:
  """Penalize contact force above what a controlled landing needs.

  Only the excess. A jump has to push off and has to absorb, and charging for all contact
  force teaches the policy not to leave the ground.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  magnitude = torch.norm(sensor.data.force, dim=-1)
  return torch.sum(torch.clamp(magnitude - force_threshold, min=0.0), dim=-1)
