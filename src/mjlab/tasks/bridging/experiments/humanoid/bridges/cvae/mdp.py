"""Observations and terminations for the CVAE bridge."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.command import (
  CvaeCommand,
)
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms

_ROBOT = SceneEntityCfg("robot")


def command(env: ManagerBasedRlEnv, name: str) -> CvaeCommand:
  term = env.command_manager.get_term(name)
  if not isinstance(term, CvaeCommand):
    raise TypeError(f"'{name}' is not a CvaeCommand")
  return term


def endpoint(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return command(env, command_name).command


def posterior_path(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return command(env, command_name).posterior_path()


def root_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return height[:, None]


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  if not isinstance(sensor, ContactSensor) or sensor.data.found is None:
    raise TypeError(f"'{sensor_name}' is not a contact sensor with found data")
  return (sensor.data.found > 0).float()


def teacher_motion(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  bridge = command(env, command_name)
  return torch.cat((bridge.motion.joint_pos, bridge.motion.joint_vel), dim=-1)


def teacher_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  bridge = command(env, command_name)
  target_pos, target_quat = bridge.teacher_anchor()
  position, _ = subtract_frame_transforms(
    bridge.motion.robot_anchor_pos_w,
    bridge.motion.robot_anchor_quat_w,
    target_pos,
    target_quat,
  )
  return position


def teacher_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  bridge = command(env, command_name)
  target_pos, target_quat = bridge.teacher_anchor()
  _, orientation = subtract_frame_transforms(
    bridge.motion.robot_anchor_pos_w,
    bridge.motion.robot_anchor_quat_w,
    target_pos,
    target_quat,
  )
  matrix = matrix_from_quat(orientation)
  return matrix[..., :2].reshape(matrix.shape[0], -1)


def deadline(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return command(env, command_name).deadline


def target_error(
  env: ManagerBasedRlEnv, command_name: str, channel: int
) -> torch.Tensor:
  return command(env, command_name).target_errors()[:, channel]


def target_success(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  bridge = command(env, command_name)
  return (bridge.target_errors() <= bridge.tolerances).all(dim=-1).float()


def fell_over(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.2
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b[:, 2] > -threshold
