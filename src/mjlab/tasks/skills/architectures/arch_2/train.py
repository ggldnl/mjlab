"""Training for arch_2: arch_1's two phases, judged on survival alone.

Phase 1 is untouched (it never had an outcome signal to begin with). Phase 2 runs the
same Double-DQN loop with the outcome below in place of arch_1's hand-written oracle, so
the only thing a hand-over is asked is whether the episode was still going.

`success_fns` is accepted and ignored, so an experiment can dispatch to any architecture
by id without special-casing which ones want an oracle.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.architectures.arch_1.config import BridgeTraining
from mjlab.tasks.skills.architectures.arch_1.outcomes import (
  OutcomeSource,
  SuccessFn,
)
from mjlab.tasks.skills.architectures.arch_1.train import train_bridging
from mjlab.tasks.skills.architectures.arch_2 import Arch2
from mjlab.tasks.skills.skill import SkillPool
from mjlab.tasks.skills.windows import WindowPlan


class SurvivalOutcome(OutcomeSource):
  """Nothing broke, so it went well.

  The caller already requires survival of every hand-over it counts, so this adds
  nothing on top: it is the deliberate absence of a judgement, and that is the whole
  point of arch_2. On a task whose failure mode is a termination (the diffdrive tips, a
  robot falls) it is exactly the right signal and costs no hand-written function. On a
  task that never terminates it makes every hand-over look equally good, and the decider
  will learn nothing beyond "commit before the window runs out".
  """

  label = "survival"

  def begin(self, num_envs: int, device: str) -> None:
    self._ones = torch.ones(num_envs, dtype=torch.bool, device=device)

  def verdict(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    del env
    return self._ones


def train(
  env: ManagerBasedRlEnv,
  pool: SkillPool,
  entity_name: str,
  meta: Arch2,
  success_fns: Mapping[int, SuccessFn],
  windows: WindowPlan,
  training: BridgeTraining,
) -> Arch2:
  """arch_2: the switch-decider learns from terminations and nothing else."""
  del success_fns  # arch_2 deliberately does without one
  outcomes: Mapping[int, OutcomeSource] = {
    i: SurvivalOutcome() for i in range(len(pool))
  }
  train_bridging(env, pool, entity_name, meta, outcomes, windows, training)
  return meta
