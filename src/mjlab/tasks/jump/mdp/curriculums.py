"""Curriculum terms for the jump task."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.jump.mdp.abstractions import LANDED, JumpAbstraction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def jump_terrain_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  abstraction_name: str,
  success_distance: float = 0.5,
) -> dict[str, torch.Tensor]:
  """Promote/demote envs across gap-difficulty rows based on jump outcome.

  Evaluated at episode reset, reading the terminal state of the just-finished
  episode:

  - **Promote** (wider gap) envs that landed within ``success_distance`` of the
    target.
  - **Demote** (narrower gap) envs that took off but failed (missed the target
    or fell in the gap), so they practice an easier jump.
  - Envs that never jumped stay put (already at the easiest level).
  """
  terrain = env.scene.terrain
  assert terrain is not None
  assert terrain.cfg.terrain_generator is not None

  jump = cast(JumpAbstraction, env.abstraction_manager.get_term(abstraction_name))
  base_pos_w = jump.robot.data.body_link_pos_w[:, jump.base_body_id]
  distance = torch.norm(
    base_pos_w[env_ids, :2] - jump.target_pos_w[env_ids, :2], dim=-1
  )

  landed = jump.phase[env_ids] == LANDED
  took_off = jump.has_taken_off[env_ids]
  success = landed & (distance < success_distance)

  move_up = success
  move_down = took_off & ~success

  terrain.update_env_origins(env_ids, move_up, move_down)

  levels = terrain.terrain_levels.float()
  return {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }
