"""Termination terms for the jump task."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.jump.mdp.abstractions import JumpAbstraction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def fell_in_gap(
  env: ManagerBasedRlEnv,
  minimum_height: float = -0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate (failure) when the base drops into the pit."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] < minimum_height


def grounded_bad_orientation(
  env: ManagerBasedRlEnv,
  abstraction_name: str,
  limit_angle: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate on excessive tilt, but only while grounded.

  Mid-flight orientation is unconstrained so the robot can tuck and reorient;
  the tilt limit is enforced at takeoff and after landing.
  """
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity = asset.data.projected_gravity_b
  tilt = torch.acos(-projected_gravity[:, 2].clamp(-1.0, 1.0)).abs()
  bad = tilt > limit_angle
  grounded = (
    cast(
      JumpAbstraction, env.abstraction_manager.get_term(abstraction_name)
    ).is_grounded
    > 0.5
  )
  return bad & grounded
