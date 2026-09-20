"""DAgger loss for the residual conditional VAE."""

from __future__ import annotations

from typing import cast

import torch
from rsl_rl.algorithms import Distillation

from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.model import (
  ResidualCvaeModel,
)


class CvaeDistillation(Distillation):
  """Train the posterior on teacher actions and match the prior to it."""

  def __init__(
    self,
    *args,
    kl_beta_start: float = 1.0e-4,
    kl_beta_end: float = 1.0e-2,
    kl_schedule_updates: int = 5000,
    action_continuity_weight: float = 0.1,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    if not 0 <= kl_beta_start <= kl_beta_end:
      raise ValueError("KL weights must satisfy 0 <= start <= end")
    if kl_schedule_updates < 1:
      raise ValueError("kl_schedule_updates must be positive")
    self.kl_beta_start = kl_beta_start
    self.kl_beta_end = kl_beta_end
    self.kl_schedule_updates = kl_schedule_updates
    if action_continuity_weight < 0:
      raise ValueError("action_continuity_weight must be nonnegative")
    self.action_continuity_weight = action_continuity_weight

  @property
  def kl_beta(self) -> float:
    progress = min(self.num_updates / self.kl_schedule_updates, 1.0)
    return self.kl_beta_start + progress * (self.kl_beta_end - self.kl_beta_start)

  def update(self) -> dict[str, float]:
    self.num_updates += 1
    behavior_total = 0.0
    kl_total = 0.0
    continuity_total = 0.0
    updates = 0
    student = cast(ResidualCvaeModel, self.student)
    self.optimizer.zero_grad()

    for _ in range(self.num_learning_epochs):
      for batch in self.storage.generator():
        assert batch.observations is not None
        assert batch.privileged_actions is not None
        actions, kl = student.reconstruct(batch.observations)
        behavior = self.loss_fn(actions, batch.privileged_actions)
        handoff = cast(torch.Tensor, batch.observations["handoff"])
        mask = handoff[:, -1]
        prior_action = student(batch.observations)
        action_difference = torch.nn.functional.smooth_l1_loss(
          prior_action, handoff[:, :-1], reduction="none"
        ).mean(dim=-1)
        continuity = (action_difference * mask).sum() / mask.sum().clamp(min=1)
        loss = (
          behavior
          + self.kl_beta * kl.mean()
          + self.action_continuity_weight * continuity
        )
        loss.backward()
        behavior_total += behavior.item()
        kl_total += kl.mean().item()
        continuity_total += continuity.item()
        updates += 1

        if updates % self.gradient_length == 0:
          self._step()
          self.optimizer.zero_grad()

    if updates % self.gradient_length:
      self._step()
    self.storage.clear()
    return {
      "behavior": behavior_total / updates,
      "kl": kl_total / updates,
      "action_continuity": continuity_total / updates,
      "kl_beta": self.kl_beta,
    }

  def _step(self) -> None:
    if self.is_multi_gpu:
      self.reduce_parameters()
    if self.max_grad_norm:
      torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
    self.optimizer.step()
