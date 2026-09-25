"""Reference-centred residual actions for the learned tracker."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  ROOT_STATE_DIM,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker.command import (
  TrackerCommand,
)


@dataclass(kw_only=True)
class ReferenceJointPositionActionCfg(JointPositionActionCfg):
  """Joint targets equal the next reference pose plus the policy residual."""

  command_name: str
  lookahead: int = 1

  def build(self, env: ManagerBasedRlEnv) -> ReferenceJointPositionAction:
    if self.use_default_offset:
      raise ValueError("Reference-centred actions cannot use the default joint offset")
    if self.lookahead < 0:
      raise ValueError("lookahead cannot be negative")
    return ReferenceJointPositionAction(self, env)


class ReferenceJointPositionAction(JointPositionAction):
  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    cfg = self.cfg
    if not isinstance(cfg, ReferenceJointPositionActionCfg):
      raise TypeError("Reference action has the wrong configuration")
    command = self._env.command_manager.get_term(cfg.command_name)
    if not isinstance(command, TrackerCommand):
      raise TypeError("Reference action requires TrackerCommand")
    reference = command.reference_at(cfg.lookahead)
    joint_reference = reference[:, ROOT_STATE_DIM : ROOT_STATE_DIM + command.num_joints]
    self._processed_actions += joint_reference[:, self._target_ids]
