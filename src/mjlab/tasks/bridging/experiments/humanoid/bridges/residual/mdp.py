"""Action, observations and rewards for terminal residual correction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.residual.command import (
  ResidualCommand,
)


def command(env: ManagerBasedRlEnv, name: str) -> ResidualCommand:
  term = env.command_manager.get_term(name)
  if not hasattr(term, "residual_weight"):
    raise TypeError(f"Command '{name}' does not support residual correction")
  return cast(ResidualCommand, term)


def blend_actions(
  base: torch.Tensor,
  residual: torch.Tensor,
  weight: torch.Tensor,
  scale: float,
) -> torch.Tensor:
  """Add a gated residual to a coarse bridge action."""
  if base.shape != residual.shape or weight.shape != base.shape[:1]:
    raise ValueError("base, residual and weight shapes do not match")
  return base + weight[:, None] * scale * residual.clamp(-1.0, 1.0)


class ResidualJointPositionAction(JointPositionAction):
  """Run the frozen imitation policy and add the learned correction."""

  cfg: ResidualJointPositionActionCfg

  def __init__(
    self, cfg: ResidualJointPositionActionCfg, env: ManagerBasedRlEnv
  ) -> None:
    super().__init__(cfg, env)
    self._base_policy: Callable[[TensorDict], torch.Tensor] | None = None
    self._effective_action = torch.zeros_like(self._raw_actions)

  @property
  def effective_action(self) -> torch.Tensor:
    return self._effective_action

  def set_base_policy(self, policy: Callable[[TensorDict], torch.Tensor]) -> None:
    self._base_policy = policy

  def process_actions(self, actions: torch.Tensor) -> None:
    if self._base_policy is None:
      raise RuntimeError(
        "The frozen imitation policy was not attached by ResidualRunner"
      )
    self._raw_actions[:] = actions
    bridge = command(self._env, self.cfg.command_name)
    enabled = getattr(bridge, "active", True)
    if isinstance(enabled, bool):
      enabled = torch.full(
        (self._env.num_envs,), enabled, dtype=torch.bool, device=actions.device
      )
    if not bool(enabled.any()):
      self._effective_action[:] = actions
      self._processed_actions = actions * self._scale + self._offset
      return
    observations = self._env.observation_manager.compute()
    with torch.no_grad():
      base = self._base_policy(
        TensorDict(
          observations,  # ty: ignore[invalid-argument-type]
          batch_size=[self._env.num_envs],
        )
      )
    combined = blend_actions(
      base, actions, bridge.residual_weight, self.cfg.residual_scale
    )
    self._effective_action[:] = torch.where(enabled[:, None], combined, actions)
    self._processed_actions = self._effective_action * self._scale + self._offset
    if self.cfg.clip is not None:
      self._processed_actions = torch.clamp(
        self._processed_actions, self._clip[..., 0], self._clip[..., 1]
      )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    self._effective_action[env_ids] = 0.0


@dataclass(kw_only=True)
class ResidualJointPositionActionCfg(JointPositionActionCfg):
  command_name: str = "bridge"
  residual_scale: float = 0.25
  imitation_checkpoint: Path | None = None

  def build(self, env: ManagerBasedRlEnv) -> ResidualJointPositionAction:
    if self.residual_scale <= 0.0:
      raise ValueError("residual_scale must be positive")
    return ResidualJointPositionAction(self, env)


def effective_action(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  term = env.action_manager.get_term(action_name)
  assert isinstance(term, ResidualJointPositionAction)
  bridge = command(env, term.cfg.command_name)
  first = env.episode_length_buf == 0
  return torch.where(first[:, None], bridge.entering_action, term.effective_action)


def progress(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  value = command(env, command_name).advance()
  return value / env.step_dt if env.cfg.scale_rewards_by_dt else value


def capture(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  value = command(env, command_name).new_capture
  return value / env.step_dt if env.cfg.scale_rewards_by_dt else value


def residual_effort(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  action = env.action_manager.get_term(action_name).raw_action
  return torch.mean(torch.square(action), dim=-1)


def captured(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  bridge = command(env, command_name)
  bridge.advance()
  return bridge.captured


def missed_deadline(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  bridge = command(env, command_name)
  bridge.advance()
  return bridge.deadline


def fell_over(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.7
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b[:, 2] > -threshold
