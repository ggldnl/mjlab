"""Reward terms for the step-over task.

``abstraction_signal`` surfaces a dense signal from the step-over abstraction
(``clearance`` / ``cross``) as a reward term. The ``crossed_*`` terms express the
"end standing still on the far side" objective: they are gated on ``is_beyond``
(the trunk has passed the barrier), so they cannot be farmed by standing on the
near side without crossing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.stepover.mdp.abstractions import StepOverAbstraction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _step(env: ManagerBasedRlEnv, abstraction_name: str) -> StepOverAbstraction:
  return cast(StepOverAbstraction, env.abstraction_manager.get_term(abstraction_name))


def abstraction_signal(
  env: ManagerBasedRlEnv, abstraction_name: str, signal_name: str
) -> torch.Tensor:
  """Return a named signal from an abstraction, shape ``(num_envs,)``."""
  return _step(env, abstraction_name).get_signal(signal_name)


def crossed_upright(
  env: ManagerBasedRlEnv,
  abstraction_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward an upright base, only once the trunk is beyond the barrier."""
  asset: Entity = env.scene[asset_cfg.name]
  xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  reward = torch.exp(-xy_squared / std**2)
  return reward * _step(env, abstraction_name).is_beyond


def crossed_posture(
  env: ManagerBasedRlEnv,
  abstraction_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the default joint pose, only once the trunk is beyond the barrier."""
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  joint_ids = asset_cfg.joint_ids
  error = torch.square(
    asset.data.joint_pos[:, joint_ids] - default_joint_pos[:, joint_ids]
  )
  reward = torch.exp(-torch.mean(error, dim=1) / std**2)
  return reward * _step(env, abstraction_name).is_beyond
