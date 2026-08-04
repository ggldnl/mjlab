"""Training for arch_2: arch_1's two phases, judged on survival alone.

Phase 1 is untouched. Phase 2 runs the same loop with `always_ok` in place of arch_1's
hand-written test, so the only thing a hand-over is asked is whether the episode was
still going -- which the caller already requires of every hand-over it counts.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.architectures.arch_1.config import BridgeTraining
from mjlab.tasks.skills.architectures.arch_1.switch import always_ok
from mjlab.tasks.skills.architectures.arch_1.train import train_bridging
from mjlab.tasks.skills.architectures.arch_2 import Arch2
from mjlab.tasks.skills.experiment import Experiment


def train(
  env: ManagerBasedRlEnv,
  exp: Experiment,
  meta: Arch2,
  cfg: BridgeTraining,
) -> Arch2:
  """arch_2: the decider learns from terminations and nothing else."""
  survival = {i: always_ok for i in range(len(exp.pool))}
  train_bridging(env, exp, meta, cfg, survival, label="survival")
  return meta
