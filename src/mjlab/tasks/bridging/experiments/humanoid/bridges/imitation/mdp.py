"""Rewards and terminations for the time-aligned imitation bridge."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
  ImitationCommand,
)


def command(env: ManagerBasedRlEnv, name: str) -> ImitationCommand:
  term = env.command_manager.get_term(name)
  assert isinstance(term, ImitationCommand)
  return term


def trajectory_tracking(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Dense reward for matching the demonstrated state at this clock tick."""
  return command(env, command_name).tracking_score()


def terminal_target(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """One target reward at the exact deadline."""
  value = command(env, command_name).terminal_reward()
  return value / env.step_dt if env.cfg.scale_rewards_by_dt else value


def deadline(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return command(env, command_name).deadline


def fell_over(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.7
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b[:, 2] > -threshold
