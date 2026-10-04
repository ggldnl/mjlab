"""Push observations, rewards, terminations and resets, shared by every robot.

Sensors are named after the box so a scene with several boxes gets one pair each:

    hands_<box>   the hands against the box, one column per hand
    robot_<box>   the whole robot subtree against the box, hands included

Illegal contact is the difference of the two counts.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.bridging.config.t1.skills.push.command import PushCommand
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  sample_uniform,
  yaw_quat,
)


def command(env: ManagerBasedRlEnv) -> PushCommand:
  term = env.command_manager.get_term("push")
  assert isinstance(term, PushCommand)
  return term


def _found(env: ManagerBasedRlEnv, prefix: str) -> torch.Tensor:
  sensor = env.scene[f"{prefix}_{command(env).box_name}"]
  assert isinstance(sensor, ContactSensor) and sensor.data.found is not None
  return sensor.data.found


##
# Observations
##


def push_command(env: ManagerBasedRlEnv) -> torch.Tensor:
  return command(env).command


def hand_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Whether each hand touches the box, (num_envs, 2)"""
  return (_found(env, "hands") > 0).float()


def illegal_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  """One while anything but a hand touches the box, (num_envs,)"""
  hands = _found(env, "hands").sum(-1)
  return (_found(env, "robot").sum(-1) > hands).float()


def body_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  return illegal_contact(env)[:, None]


def box_features(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Box position, velocity, forward and up axes, angular velocity, heading frame"""
  robot = env.scene["robot"]
  box = command(env).box
  heading = yaw_quat(robot.data.root_link_quat_w)
  quat = box.data.root_link_quat_w
  forward = quat_apply(quat, quat.new_tensor([1.0, 0.0, 0.0]).expand(env.num_envs, -1))
  up = quat_apply(quat, quat.new_tensor([0.0, 0.0, 1.0]).expand(env.num_envs, -1))
  vectors = (
    box.data.root_link_pos_w - robot.data.root_link_pos_w,
    box.data.root_link_lin_vel_w,
    forward,
    up,
    box.data.root_link_ang_vel_w,
  )
  return torch.cat([quat_apply_inverse(heading, v) for v in vectors], dim=-1)


def hand_offsets(env: ManagerBasedRlEnv, hands_cfg: SceneEntityCfg) -> torch.Tensor:
  """Hand geoms relative to the box center, heading frame, (num_envs, 6)"""
  robot = env.scene["robot"]
  offsets = (
    robot.data.geom_pos_w[:, hands_cfg.geom_ids]
    - command(env).box.data.root_link_pos_w[:, None]
  )
  heading = yaw_quat(robot.data.root_link_quat_w)[:, None].expand(-1, 2, -1)
  return quat_apply_inverse(heading, offsets).flatten(1)


##
# Rewards
##


def box_velocity(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
  """Box tracks the reference speed along the axis and does not slide sideways"""
  term = command(env)
  velocity = term.box.data.root_link_lin_vel_w[:, :2]
  along = (velocity * term.direction_w).sum(-1)
  side = (velocity * term.left_w).sum(-1)
  error = (along - term.speed).square() + side.square()
  return torch.exp(-error / std**2)


def box_position(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
  return torch.exp(-command(env).position_error.square() / std**2)


def hands_on_box(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Fraction of the hands on the box while there is still pushing to do"""
  return hand_contact(env).mean(-1) * (command(env).speed > 0).float()


def body_clearance(
  env: ManagerBasedRlEnv, min_clearance: float, half_size: float = 0.5
) -> torch.Tensor:
  """Cost of a robot root closer than min_clearance to the box rear face"""
  term = command(env)
  offset = (
    term.box.data.root_link_pos_w[:, :2]
    - env.scene["robot"].data.root_link_pos_w[:, :2]
  )
  clearance = (offset * term.direction_w).sum(-1) - half_size
  return ((min_clearance - clearance).clamp(min=0.0) / min_clearance).square()


def flight_phase(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """One while no foot is on the ground, the signature of a hop"""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor) and sensor.data.found is not None
  return (sensor.data.found == 0).all(-1).float()


def box_lane_error(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = command(env)
  return (term.lateral_displacement / term.cfg.lateral_tolerance).square()


def box_tilt_error(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = command(env)
  return (term.tilt / term.cfg.tilt_tolerance).square()


def box_yaw_error(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = command(env)
  return (term.yaw_error / term.cfg.yaw_tolerance).square()


def box_angular_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
  return command(env).box.data.root_link_ang_vel_w.square().sum(-1)


##
# Terminations
##


def invalid_box(env: ManagerBasedRlEnv, half_size: float = 0.5) -> torch.Tensor:
  """Box out of its lane, tipping, lifted, turned or pushed past the target"""
  term = command(env)
  height = term.box.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return (
    (term.lateral_displacement.abs() > term.cfg.max_lateral_displacement)
    | (term.tilt > term.cfg.max_tilt)
    | (height > half_size + 0.15)
    | (term.yaw_error.abs() > term.cfg.max_yaw)
    | (term.remaining < -term.cfg.max_overshoot)
  )


##
# Events
##


def reset_box(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  pose_range: dict[str, tuple[float, float]],
  cell: tuple[float, float] = (1.5, 0.5),
  half_size: float = 0.5,
) -> None:
  """Put the box in its cell ahead of the robot, with a small placement error"""
  box = env.scene["box"]
  n = len(env_ids)
  root = box.data.default_root_state[env_ids].clone()
  root[:, 0] = cell[0] + sample_uniform(*pose_range.get("x", (0.0, 0.0)), n, env.device)
  root[:, 1] = cell[1] + sample_uniform(*pose_range.get("y", (0.0, 0.0)), n, env.device)
  root[:, 2] = half_size
  root[:, :3] += env.scene.env_origins[env_ids]
  yaw = sample_uniform(*pose_range.get("yaw", (0.0, 0.0)), n, env.device)
  root[:, 3:7] = 0.0
  root[:, 3] = torch.cos(0.5 * yaw)
  root[:, 6] = torch.sin(0.5 * yaw)
  root[:, 7:] = 0.0
  box.write_root_state_to_sim(root, env_ids)


##
# Metrics
##


def hands_contact_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  return hand_contact(env).mean(-1)


def body_contact_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  return illegal_contact(env)


def flight_rate(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  return flight_phase(env, sensor_name)
