"""Curriculum terms for the step-over task."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.stepover.mdp.abstractions import StepOverAbstraction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def barrier_terrain_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  abstraction_name: str,
) -> dict[str, torch.Tensor]:
  """Promote/demote envs across barrier-height rows based on the outcome.

  Evaluated at episode reset, reading the terminal state of the just-finished
  episode: promote (taller barrier) envs that got both feet across; demote
  (shorter barrier) the rest. Levels are clamped, so envs that fail at the flat
  row simply stay there.
  """
  terrain = env.scene.terrain
  assert terrain is not None
  assert terrain.cfg.terrain_generator is not None

  step = cast(StepOverAbstraction, env.abstraction_manager.get_term(abstraction_name))
  success = step.crossed[env_ids].all(dim=1)

  terrain.update_env_origins(env_ids, success, ~success)

  levels = terrain.terrain_levels.float()
  return {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }
