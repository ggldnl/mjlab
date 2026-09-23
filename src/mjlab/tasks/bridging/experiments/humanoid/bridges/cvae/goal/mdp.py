"""Observations and terminal diagnostics for the goal CVAE."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae import mdp as legacy_mdp
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.command import (
  GoalCommand,
)


def command(env: ManagerBasedRlEnv, name: str) -> GoalCommand:
  term = env.command_manager.get_term(name)
  if not isinstance(term, GoalCommand):
    raise TypeError(f"{name} is not a GoalCommand")
  return term


def endpoint(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return command(env, command_name).command


def initial(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return command(env, command_name).initial_condition()


def posterior(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return command(env, command_name).posterior_path()


def route(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return command(env, command_name).route_label()


def target_error(
  env: ManagerBasedRlEnv, command_name: str, channel: int
) -> torch.Tensor:
  return command(env, command_name).target_errors()[:, channel]


def foot_error(env: ManagerBasedRlEnv, command_name: str, channel: int) -> torch.Tensor:
  return command(env, command_name).foot_errors()[:, channel]


def target_success(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  bridge = command(env, command_name)
  state_ok = (bridge.target_errors() <= bridge.tolerances).all(dim=-1)
  foot_limits = bridge.cfg.foot_tolerances
  foot_ok = (
    bridge.foot_errors() <= torch.tensor(foot_limits, device=bridge.device)
  ).all(dim=-1)
  return (state_ok & foot_ok).float()


def deadline(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return command(env, command_name).deadline


fell_over = legacy_mdp.fell_over
