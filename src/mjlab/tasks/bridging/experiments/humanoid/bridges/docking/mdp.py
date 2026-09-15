"""Actions, rewards and terminations for docking bridge training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.docking.command import (
  DockingCommand,
)


def command(env: ManagerBasedRlEnv, name: str) -> DockingCommand:
  term = env.command_manager.get_term(name)
  assert isinstance(term, DockingCommand)
  return term


class DockingJointPositionAction(JointPositionAction):
  """Use direct actions on approach and residual actions while docking."""

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    cfg = self.cfg
    assert isinstance(cfg, DockingJointPositionActionCfg)
    bridge = command(self._env, cfg.command_name)
    residual = (
      bridge.entering_action
      + (1.0 - bridge.blend[:, None]) * cfg.residual_scale * actions
    )
    effective = torch.where(bridge.docking[:, None], residual, actions)
    self._processed_actions = effective * self._scale + self._offset
    if self.cfg.clip is not None:
      self._processed_actions = torch.clamp(
        self._processed_actions, self._clip[..., 0], self._clip[..., 1]
      )


@dataclass(kw_only=True)
class DockingJointPositionActionCfg(JointPositionActionCfg):
  command_name: str
  residual_scale: float = 0.25

  def build(self, env: ManagerBasedRlEnv) -> DockingJointPositionAction:
    return DockingJointPositionAction(self, env)


def progress(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  bridge = command(env, command_name)
  value = bridge.advance()
  return value / env.step_dt if env.cfg.scale_rewards_by_dt else value


def capture(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  bridge = command(env, command_name)
  value = bridge.new_capture
  return value / env.step_dt if env.cfg.scale_rewards_by_dt else value


def handoff(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  bridge = command(env, command_name)
  bridge.advance()
  return bridge.handoff


def missed_deadline(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  bridge = command(env, command_name)
  bridge.advance()
  return bridge.deadline


def strayed(
  env: ManagerBasedRlEnv, command_name: str, margin: float = 1.5
) -> torch.Tensor:
  bridge = command(env, command_name)
  distance = torch.linalg.vector_norm(
    bridge.robot.data.root_link_pos_w - bridge.target[:, 0:3], dim=-1
  )
  return distance > bridge.start_distance + margin


def fell_over(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.7
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b[:, 2] > -threshold
