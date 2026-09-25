"""Observations, rewards, metrics, and terminations for the learned tracker."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  ROOT_STATE_DIM,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker.command import (
  TrackerCommand,
)
from mjlab.utils.lab_api.math import quat_apply_inverse

_ROBOT = SceneEntityCfg("robot")


def tracker(env: ManagerBasedRlEnv, command_name: str) -> TrackerCommand:
  term = env.command_manager.get_term(command_name)
  if not isinstance(term, TrackerCommand):
    raise TypeError(f"'{command_name}' is not a TrackerCommand")
  return term


def root_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  return (robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2])[:, None]


def _history(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return tracker(env, command_name).actual_history


def history_root_height(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  states = _history(env, command_name)
  height = states[..., 2:3] - env.scene.env_origins[:, None, 2:3]
  return height.flatten(1)


def history_base_lin_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  states = _history(env, command_name)
  return quat_apply_inverse(states[..., 3:7].contiguous(), states[..., 7:10]).flatten(1)


def history_base_ang_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  states = _history(env, command_name)
  return quat_apply_inverse(states[..., 3:7].contiguous(), states[..., 10:13]).flatten(
    1
  )


def history_projected_gravity(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  states = _history(env, command_name)
  gravity = states.new_zeros((*states.shape[:2], 3))
  gravity[..., 2] = -1.0
  return quat_apply_inverse(states[..., 3:7].contiguous(), gravity).flatten(1)


def history_joint_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  states = _history(env, command_name)
  robot: Entity = env.scene[_ROBOT.name]
  default = robot.data.default_joint_pos
  assert default is not None
  joints = robot.data.joint_pos.shape[1]
  q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + joints)
  return (states[..., q] - default[:, None]).flatten(1)


def history_joint_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  states = _history(env, command_name)
  robot: Entity = env.scene[_ROBOT.name]
  default = robot.data.default_joint_vel
  assert default is not None
  joints = robot.data.joint_pos.shape[1]
  qd = slice(ROOT_STATE_DIM + joints, ROOT_STATE_DIM + 2 * joints)
  return (states[..., qd] - default[:, None]).flatten(1)


def action_history(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Current and two preceding raw residual actions."""
  return torch.cat(
    (
      env.action_manager.action,
      env.action_manager.prev_action,
      env.action_manager.prev_prev_action,
    ),
    dim=-1,
  )


def tracking_errors(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = tracker(env, command_name)
  from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
    channel_errors,
  )

  errors = channel_errors(
    command.state_now(), command.reference_now(), command.upper_body
  )
  return errors / command.tolerances


def endpoint_errors(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = tracker(env, command_name)
  return command.target_errors() / command.tolerances


def trajectory_tracking(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return tracker(env, command_name).tracking_score()


def endpoint_focus(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return tracker(env, command_name).endpoint_score()


def terminal_target(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  value = tracker(env, command_name).terminal_reward()
  return value / env.step_dt if env.cfg.scale_rewards_by_dt else value


def deadline(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return tracker(env, command_name).deadline


def fell_over(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.7
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  return robot.data.projected_gravity_b[:, 2] > -threshold


def target_error(
  env: ManagerBasedRlEnv, command_name: str, channel: int
) -> torch.Tensor:
  return tracker(env, command_name).target_errors()[:, channel]


def target_success(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = tracker(env, command_name)
  return (
    command.deadline & (command.target_errors() <= command.tolerances).all(dim=-1)
  ).float()


def route_score(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return tracker(env, command_name).tracking_score()
