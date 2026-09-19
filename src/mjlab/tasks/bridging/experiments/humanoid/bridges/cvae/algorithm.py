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

  @property
  def kl_beta(self) -> float:
    progress = min(self.num_updates / self.kl_schedule_updates, 1.0)
    return self.kl_beta_start + progress * (self.kl_beta_end - self.kl_beta_start)

  def update(self) -> dict[str, float]:
    self.num_updates += 1
    behavior_total = 0.0
    kl_total = 0.0
    updates = 0
    accumulated: torch.Tensor | None = None
    student = cast(ResidualCvaeModel, self.student)

    for _ in range(self.num_learning_epochs):
      for batch in self.storage.generator():
        assert batch.observations is not None
        assert batch.privileged_actions is not None
        actions, kl = student.reconstruct(batch.observations)
        behavior = self.loss_fn(actions, batch.privileged_actions)
        loss = behavior + self.kl_beta * kl.mean()
        accumulated = loss if accumulated is None else accumulated + loss
        behavior_total += behavior.item()
        kl_total += kl.mean().item()
        updates += 1

        if updates % self.gradient_length == 0:
          assert accumulated is not None
          self._step(accumulated)
          accumulated = None

    if accumulated is not None:
      self._step(accumulated)
    self.storage.clear()
    return {
      "behavior": behavior_total / updates,
      "kl": kl_total / updates,
      "kl_beta": self.kl_beta,
    }

  def _step(self, loss: torch.Tensor) -> None:
    self.optimizer.zero_grad()
    loss.backward()
    if self.is_multi_gpu:
      self.reduce_parameters()
    if self.max_grad_norm:
      torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
    self.optimizer.step()
