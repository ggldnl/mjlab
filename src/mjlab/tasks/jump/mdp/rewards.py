"""Reward terms for the jump task.

Two families:

- ``abstraction_signal`` surfaces a signal produced by the jump abstraction
  (takeoff / tracking / landing) as a reward term.
- ``landed_*`` express stability rewards (upright, posture) for the *terminal*
  position only: they are gated to the LANDED phase **and** weighted by
  proximity to the landing target. This prevents two reward-hacking loops:
  standing still at the start (never LANDED -> no reward) and hopping in place
  to reach LANDED without crossing the gap (lands far from the target ->
  proximity ~0 -> no reward). Start-uprightness is instead enforced by the
  ``fell_over`` termination, which cannot be farmed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.jump.mdp.abstractions import JumpAbstraction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _jump(env: ManagerBasedRlEnv, abstraction_name: str) -> JumpAbstraction:
  return cast(JumpAbstraction, env.abstraction_manager.get_term(abstraction_name))


def abstraction_signal(
  env: ManagerBasedRlEnv, abstraction_name: str, signal_name: str
) -> torch.Tensor:
  """Return a named signal from an abstraction, shape ``(num_envs,)``."""
  return _jump(env, abstraction_name).get_signal(signal_name)


def termination_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1.0 on failure terminations (falling), 0 on time-outs and survival.

  Use with a negative weight. Without this, every per-step reward is <= 0 until
  a jump completes, so ending the episode early is optimal under discounting -
  the robot learns to tip over and die rather than stand or jump. Penalizing
  the fall makes "tip and die" strictly worse than standing, which is in turn
  worse than a real jump.
  """
  return env.termination_manager.terminated.float()


def _settle_gate(
  env: ManagerBasedRlEnv, abstraction_name: str, proximity_std: float
) -> torch.Tensor:
  """Gate that is non-zero only after a successful jump near the target.

  ``is_landed * target_proximity``, shape ``(num_envs,)``.
  """
  jump = _jump(env, abstraction_name)
  return jump.is_landed * jump.target_proximity(proximity_std)


def landed_upright(
  env: ManagerBasedRlEnv,
  abstraction_name: str,
  std: float,
  proximity_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward an upright base, only once landed near the target."""
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity_b = asset.data.projected_gravity_b
  xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  reward = torch.exp(-xy_squared / std**2)
  return reward * _settle_gate(env, abstraction_name, proximity_std)


def landed_posture(
  env: ManagerBasedRlEnv,
  abstraction_name: str,
  std: float,
  proximity_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the default joint pose, only once landed near the target."""
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  joint_ids = asset_cfg.joint_ids
  error = torch.square(
    asset.data.joint_pos[:, joint_ids] - default_joint_pos[:, joint_ids]
  )
  reward = torch.exp(-torch.mean(error, dim=1) / std**2)
  return reward * _settle_gate(env, abstraction_name, proximity_std)
