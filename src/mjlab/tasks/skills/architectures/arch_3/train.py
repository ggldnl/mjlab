"""Training for arch_3: residuals.

Not implemented yet. The stub keeps arch_3 on the same `train(...)` entry point as the
other architectures, so an experiment can already dispatch to it by id; calling it just
reports that the architecture is unfinished.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.architectures.arch_3 import Arch3
from mjlab.tasks.skills.skill import SkillPool
from mjlab.tasks.skills.windows import WindowPlan


def train(
  env: ManagerBasedRlEnv,
  pool: SkillPool,
  entity_name: str,
  meta: Arch3,
  success_fns: Mapping[int, Any],
  windows: WindowPlan,
  training: Any,
) -> Arch3:
  del env, pool, entity_name, meta, success_fns, windows, training
  raise NotImplementedError("arch_3 training is not implemented yet.")
