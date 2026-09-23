from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.oracle.command import (
  MotionSetCommand,
)
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_error_magnitude,
  subtract_frame_transforms,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> MotionSetCommand:
  return cast(MotionSetCommand, env.command_manager.get_term(command_name))


def _expand_anchor(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  shape = (value.shape[0],) + (1,) * (target.ndim - 2) + (value.shape[-1],)
  return value.reshape(shape).expand(target.shape[:-1] + (value.shape[-1],))


def _body_pose_in_anchor(
  command: MotionSetCommand, positions: torch.Tensor, orientations: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  return subtract_frame_transforms(
    _expand_anchor(command.robot_anchor_pos_w, positions),
    _expand_anchor(command.robot_anchor_quat_w, orientations),
    positions,
    orientations,
  )


def _rotation_6d(quaternions: torch.Tensor) -> torch.Tensor:
  return matrix_from_quat(quaternions)[..., :2, :].flatten(-2)


def _future_robot(value: torch.Tensor, frames: int) -> torch.Tensor:
  return value[:, None].expand((value.shape[0], frames) + value.shape[1:])


def _controlled_joints(command: MotionSetCommand) -> torch.Tensor:
  """Joints controlled by the 23-DoF oracle, excluding fixed wrists."""
  return torch.tensor(
    ["wrist" not in name for name in command.robot.joint_names],
    device=command.device,
    dtype=torch.bool,
  )


def _upper_joints(command: MotionSetCommand) -> torch.Tensor:
  return torch.tensor(
    [
      "wrist" not in name and ("shoulder" in name or "elbow" in name)
      for name in command.robot.joint_names
    ],
    device=command.device,
    dtype=torch.bool,
  )


def oracle_proprioception(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Privileged current robot state in the robot anchor frame."""
  command = _command(env, command_name)
  body_pos_b, body_quat_b = _body_pose_in_anchor(
    command, command.robot_body_pos_w, command.robot_body_quat_w
  )
  anchor_quat = _expand_anchor(
    command.robot_anchor_quat_w, command.robot_body_lin_vel_w
  )
  body_lin_vel_b = quat_apply_inverse(anchor_quat, command.robot_body_lin_vel_w)
  body_ang_vel_b = quat_apply_inverse(anchor_quat, command.robot_body_ang_vel_w)
  controlled = _controlled_joints(command)
  return torch.cat(
    (
      body_pos_b.flatten(1),
      _rotation_6d(body_quat_b).flatten(1),
      body_lin_vel_b.flatten(1),
      body_ang_vel_b.flatten(1),
      command.robot_joint_pos[:, controlled],
      command.robot_joint_vel[:, controlled],
      env.action_manager.action,
    ),
    dim=-1,
  )


def oracle_goal(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Five consecutive future frames relative to the current robot state."""
  command = _command(env, command_name)
  frames = len(command.oracle_cfg.future_steps)
  robot_pos = _future_robot(command.robot_body_pos_w, frames)
  robot_quat = _future_robot(command.robot_body_quat_w, frames)
  robot_lin_vel = _future_robot(command.robot_body_lin_vel_w, frames)
  robot_ang_vel = _future_robot(command.robot_body_ang_vel_w, frames)
  robot_joint_pos = _future_robot(command.robot_joint_pos, frames)
  robot_joint_vel = _future_robot(command.robot_joint_vel, frames)

  robot_pos_b, _ = _body_pose_in_anchor(command, robot_pos, robot_quat)
  future_pos_b, _ = _body_pose_in_anchor(
    command, command.future_body_pos_w, command.future_body_quat_w
  )
  _, body_quat_error = subtract_frame_transforms(
    robot_pos,
    robot_quat,
    command.future_body_pos_w,
    command.future_body_quat_w,
  )
  anchor_quat = _expand_anchor(command.robot_anchor_quat_w, robot_lin_vel)
  lin_vel_error_b = quat_apply_inverse(
    anchor_quat, command.future_body_lin_vel_w - robot_lin_vel
  )
  ang_vel_error_b = quat_apply_inverse(
    anchor_quat, command.future_body_ang_vel_w - robot_ang_vel
  )
  body_pos = future_pos_b - robot_pos_b
  body_ori = _rotation_6d(body_quat_error).flatten(2)
  body_lin_vel = lin_vel_error_b.flatten(2)
  body_ang_vel = ang_vel_error_b.flatten(2)
  joint_pos = command.future_joint_pos - robot_joint_pos
  joint_vel = command.future_joint_vel - robot_joint_vel
  controlled = _controlled_joints(command)

  root = 0
  feet = command._foot_indexes
  per_frame = torch.cat(
    (
      body_pos.flatten(2),
      body_ori,
      body_lin_vel,
      body_ang_vel,
      joint_pos[:, :, controlled],
      joint_vel[:, :, controlled],
      body_pos[:, :, root],
      _rotation_6d(body_quat_error[:, :, root]),
      lin_vel_error_b[:, :, root],
      ang_vel_error_b[:, :, root],
      body_pos[:, :, feet].flatten(2),
      _rotation_6d(body_quat_error[:, :, feet]).flatten(2),
      lin_vel_error_b[:, :, feet].flatten(2),
      ang_vel_error_b[:, :, feet].flatten(2),
    ),
    dim=-1,
  )
  return per_frame.flatten(1)


def _tracking_reward(error: torch.Tensor, tight: float, broad: float) -> torch.Tensor:
  """Keep a broad learning signal while rewarding handoff precision."""
  return 0.5 * torch.exp(-torch.square(error / tight)) + 0.5 * torch.exp(
    -torch.square(error / broad)
  )


def root_position_tracking(
  env: ManagerBasedRlEnv, command_name: str, tight: float, broad: float
) -> torch.Tensor:
  command = _command(env, command_name)
  error = torch.linalg.vector_norm(
    command.goal_body_pos_w[:, 0] - command.robot_body_pos_w[:, 0], dim=-1
  )
  return _tracking_reward(error, tight, broad)


def root_orientation_tracking(
  env: ManagerBasedRlEnv, command_name: str, tight: float, broad: float
) -> torch.Tensor:
  command = _command(env, command_name)
  error = quat_error_magnitude(
    command.goal_body_quat_w[:, 0], command.robot_body_quat_w[:, 0]
  )
  return _tracking_reward(error, tight, broad)


def root_linear_velocity_tracking(
  env: ManagerBasedRlEnv, command_name: str, tight: float, broad: float
) -> torch.Tensor:
  command = _command(env, command_name)
  error = torch.linalg.vector_norm(
    command.goal_body_lin_vel_w[:, 0] - command.robot_body_lin_vel_w[:, 0],
    dim=-1,
  )
  return _tracking_reward(error, tight, broad)


def root_angular_velocity_tracking(
  env: ManagerBasedRlEnv, command_name: str, tight: float, broad: float
) -> torch.Tensor:
  command = _command(env, command_name)
  error = torch.linalg.vector_norm(
    command.goal_body_ang_vel_w[:, 0] - command.robot_body_ang_vel_w[:, 0],
    dim=-1,
  )
  return _tracking_reward(error, tight, broad)


def body_position_tracking(
  env: ManagerBasedRlEnv, command_name: str, tight: float, broad: float
) -> torch.Tensor:
  command = _command(env, command_name)
  robot_pos_b, _ = _body_pose_in_anchor(
    command, command.robot_body_pos_w, command.robot_body_quat_w
  )
  goal_pos_b, _ = _body_pose_in_anchor(
    command, command.goal_body_pos_w, command.goal_body_quat_w
  )
  error = torch.linalg.vector_norm(goal_pos_b - robot_pos_b, dim=-1).mean(dim=-1)
  return _tracking_reward(error, tight, broad)


def body_orientation_tracking(
  env: ManagerBasedRlEnv, command_name: str, tight: float, broad: float
) -> torch.Tensor:
  command = _command(env, command_name)
  error = quat_error_magnitude(
    command.goal_body_quat_w, command.robot_body_quat_w
  ).mean(dim=-1)
  return _tracking_reward(error, tight, broad)


def body_linear_velocity_tracking(
  env: ManagerBasedRlEnv, command_name: str, tight: float, broad: float
) -> torch.Tensor:
  command = _command(env, command_name)
  error = torch.linalg.vector_norm(
    command.goal_body_lin_vel_w - command.robot_body_lin_vel_w, dim=-1
  ).mean(dim=-1)
  return _tracking_reward(error, tight, broad)


def body_angular_velocity_tracking(
  env: ManagerBasedRlEnv, command_name: str, tight: float, broad: float
) -> torch.Tensor:
  command = _command(env, command_name)
  error = torch.linalg.vector_norm(
    command.goal_body_ang_vel_w - command.robot_body_ang_vel_w, dim=-1
  ).mean(dim=-1)
  return _tracking_reward(error, tight, broad)


def joint_position_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  tight: float,
  broad: float,
  upper: bool,
) -> torch.Tensor:
  command = _command(env, command_name)
  selected = _upper_joints(command)
  selected = selected if upper else _controlled_joints(command) & ~selected
  error = (command.goal_joint_pos - command.robot_joint_pos)[:, selected].abs().amax(-1)
  return _tracking_reward(error, tight, broad)


def joint_velocity_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  tight: float,
  broad: float,
  upper: bool,
) -> torch.Tensor:
  command = _command(env, command_name)
  selected = _upper_joints(command)
  selected = selected if upper else _controlled_joints(command) & ~selected
  error = (command.goal_joint_vel - command.robot_joint_vel)[:, selected].abs().amax(-1)
  return _tracking_reward(error, tight, broad)


def foot_position_tracking(
  env: ManagerBasedRlEnv, command_name: str, tight: float, broad: float
) -> torch.Tensor:
  command = _command(env, command_name)
  feet = command._foot_indexes
  error = torch.linalg.vector_norm(
    (command.goal_body_pos_w[:, feet] - command.robot_body_pos_w[:, feet]).flatten(1),
    dim=-1,
  )
  return _tracking_reward(error, tight, broad)


def foot_orientation_tracking(
  env: ManagerBasedRlEnv, command_name: str, tight: float, broad: float
) -> torch.Tensor:
  command = _command(env, command_name)
  feet = command._foot_indexes
  error = torch.linalg.vector_norm(
    quat_error_magnitude(
      command.goal_body_quat_w[:, feet], command.robot_body_quat_w[:, feet]
    ),
    dim=-1,
  )
  return _tracking_reward(error, tight, broad)


def foot_linear_velocity_tracking(
  env: ManagerBasedRlEnv, command_name: str, tight: float, broad: float
) -> torch.Tensor:
  command = _command(env, command_name)
  feet = command._foot_indexes
  error = torch.linalg.vector_norm(
    (
      command.goal_body_lin_vel_w[:, feet] - command.robot_body_lin_vel_w[:, feet]
    ).flatten(1),
    dim=-1,
  )
  return _tracking_reward(error, tight, broad)


def foot_angular_velocity_tracking(
  env: ManagerBasedRlEnv, command_name: str, tight: float, broad: float
) -> torch.Tensor:
  command = _command(env, command_name)
  feet = command._foot_indexes
  error = torch.linalg.vector_norm(
    (
      command.goal_body_ang_vel_w[:, feet] - command.robot_body_ang_vel_w[:, feet]
    ).flatten(1),
    dim=-1,
  )
  return _tracking_reward(error, tight, broad)


def bad_goal_anchor_pos_z(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = _command(env, command_name)
  anchor = command.motion_anchor_body_index
  return (
    command.goal_body_pos_w[:, anchor, 2] - command.robot_anchor_pos_w[:, 2]
  ).abs() > threshold


def bad_goal_anchor_ori(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = _command(env, command_name)
  anchor = command.motion_anchor_body_index
  gravity = command.robot.data.gravity_vec_w
  goal_gravity = quat_apply_inverse(command.goal_body_quat_w[:, anchor], gravity)
  robot_gravity = quat_apply_inverse(command.robot_anchor_quat_w, gravity)
  return (goal_gravity[:, 2] - robot_gravity[:, 2]).abs() > threshold


def bad_goal_body_pos_z(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  command = _command(env, command_name)
  indexes = [command.cfg.body_names.index(name) for name in body_names]
  goal_pos_relative_w, _ = command._relative_goal_body_pose()
  error = (
    goal_pos_relative_w[:, indexes, 2] - command.robot_body_pos_w[:, indexes, 2]
  ).abs()
  return torch.any(error > threshold, dim=-1)
