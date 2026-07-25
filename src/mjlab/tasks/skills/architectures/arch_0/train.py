"""Training stub for arch_0: the no-bridge baseline has nothing to train.

arch_0 hands over directly, so there are no networks to fit and nothing to save.
This module exists only so every architecture exposes the same `train(...)` entry
point, letting an experiment train any architecture by id without special-casing it.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.architectures.arch_0 import Arch0
from mjlab.tasks.skills.skill import SkillPool

SuccessFn = Callable[[ManagerBasedRlEnv], torch.Tensor]


def train(
  env: ManagerBasedRlEnv,
  pool: SkillPool,
  entity_name: str,
  meta: Arch0,
  success_fns: dict[int, SuccessFn],
) -> Arch0:
  """Return `meta` unchanged: the direct hand-off baseline has nothing to learn."""
  del env, pool, entity_name, success_fns
  return meta
