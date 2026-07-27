"""Training for arch_3: arch_1's two phases, judged by the environment's own reward.

Phase 1 is untouched. Phase 2 runs the same Double-DQN loop with the outcome below in
place of arch_1's hand-written oracle: the target skill's post-hand-over reward against
what that same skill earns starting from its own reset, measured once before training.

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
from mjlab.tasks.skills.architectures.arch_3 import Arch3
from mjlab.tasks.skills.skill import Skill, SkillPool
from mjlab.tasks.skills.windows import WindowPlan


class RewardOutcome(OutcomeSource):
  """The target skill went on to earn what it usually earns.

  The bar is measured, not chosen: before training, the target skill is rolled from its
  own reset and the reward it collects per step is recorded. A hand-over passes if the
  reward collected after it is within `margin` standard deviations of that reference.
  Using the spread rather than a fraction keeps it well-behaved whatever sign the
  rewards happen to have, and means there is no threshold to invent per experiment,
  which was the whole complaint about a hand-written oracle.
  """

  label = "reward"

  def __init__(self, margin: float = 1.0) -> None:
    self.margin = margin
    self.threshold = 0.0
    self.reference = 0.0
    self._total = torch.zeros(0)
    self._steps = torch.zeros(0)

  def prepare(
    self, env: ManagerBasedRlEnv, target_skill: Skill, eval_steps: int
  ) -> None:
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    obs, _ = env.reset()
    target_skill.reset(active)
    total = torch.zeros(env.num_envs, device=env.device)
    for _ in range(eval_steps):
      obs, reward, _, _, _ = env.step(target_skill.act(obs, active))
      total += reward
    per_step = total / eval_steps
    self.reference = float(per_step.mean())
    self.threshold = self.reference - self.margin * float(per_step.std())
    print(
      f"[outcome] '{target_skill.name}' from its own reset earns "
      f"{self.reference:+.4f} per step; a hand-over passes above {self.threshold:+.4f}"
    )

  def begin(self, num_envs: int, device: str) -> None:
    self._total = torch.zeros(num_envs, device=device)
    self._steps = torch.zeros(num_envs, device=device)

  def record(self, reward: torch.Tensor, counting: torch.Tensor) -> None:
    self._total += reward * counting
    self._steps += counting

  def verdict(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    del env
    earned = self._total / self._steps.clamp_min(1.0)
    return (earned >= self.threshold) & (self._steps > 0)

  def summary(self) -> str:
    earned = self._total / self._steps.clamp_min(1.0)
    counted = self._steps > 0
    if not bool(counted.any()):
      return "earned   n/a"
    return f"earned {float(earned[counted].mean()):+6.4f}"


def train(
  env: ManagerBasedRlEnv,
  pool: SkillPool,
  entity_name: str,
  meta: Arch3,
  success_fns: Mapping[int, SuccessFn],
  windows: WindowPlan,
  training: BridgeTraining,
  reward_margin: float = 1.0,
) -> Arch3:
  """arch_3: the switch-decider learns from the task reward.

  `reward_margin` is how far below the target skill's own reference a hand-over may earn
  and still pass, in standard deviations of that reference. Larger is more forgiving.
  """
  del success_fns  # arch_3 deliberately does without one
  outcomes: Mapping[int, OutcomeSource] = {
    i: RewardOutcome(reward_margin) for i in range(len(pool))
  }
  train_bridging(env, pool, entity_name, meta, outcomes, windows, training)
  return meta
