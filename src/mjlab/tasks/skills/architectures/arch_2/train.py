"""Training for arch_2: the single universal bridge.

Not implemented yet. The stub keeps arch_2 on the same `train(...)` entry point as
the other architectures, so an experiment can already dispatch to it by id; calling
it just reports that the architecture is unfinished.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.architectures.arch_2 import Arch2
from mjlab.tasks.skills.skill import SkillPool

SuccessFn = Callable[[ManagerBasedRlEnv], torch.Tensor]


def train(
  env: ManagerBasedRlEnv,
  pool: SkillPool,
  entity_name: str,
  meta: Arch2,
  success_fns: dict[int, SuccessFn],
) -> Arch2:
  del env, pool, entity_name, meta, success_fns
  raise NotImplementedError("arch_2 training is not implemented yet.")
