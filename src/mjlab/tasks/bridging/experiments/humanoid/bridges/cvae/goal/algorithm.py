"""DAgger imitation with route and demonstrated-foot supervision."""

from __future__ import annotations

from typing import cast

import torch
from rsl_rl.algorithms import Distillation

from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.model import (
  GoalCvaeModel,
)


class GoalDistillation(Distillation):
  """Fit oracle actions and a route-aware latent on student rollouts."""

  def __init__(
    self,
    *args,
    kl_beta_start: float = 1.0e-4,
    kl_beta_end: float = 1.0e-2,
    kl_schedule_updates: int = 5000,
    route_weight: float = 1.0,
    foot_path_weight: float = 0.1,
    upper_action_weight: float = 0.3,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.kl_beta_start = kl_beta_start
    self.kl_beta_end = kl_beta_end
    self.kl_schedule_updates = kl_schedule_updates
    self.route_weight = route_weight
    self.foot_path_weight = foot_path_weight
    self.upper_action_weight = upper_action_weight
    self.action_weights = torch.ones(
      cast(GoalCvaeModel, self.student).output_dim, device=self.device
    )

  def update(self) -> dict[str, float]:
    self.num_updates += 1
    student = cast(GoalCvaeModel, self.student)
    beta = self.kl_beta_start + min(
      self.num_updates / self.kl_schedule_updates, 1.0
    ) * (self.kl_beta_end - self.kl_beta_start)
    totals = torch.zeros(4)
    updates = 0
    self.optimizer.zero_grad()
    for _ in range(self.num_learning_epochs):
      for batch in self.storage.generator():
        assert batch.observations is not None
        assert batch.privileged_actions is not None
        action, kl, route, foot_path = student.reconstruct(batch.observations)
        behavior = (
          torch.nn.functional.smooth_l1_loss(
            action, batch.privileged_actions, reduction="none"
          )
          * self.action_weights
        ).mean()
        loss = (
          behavior
          + beta * kl
          + self.route_weight * route
          + self.foot_path_weight * foot_path
        )
        loss.backward()
        totals += torch.tensor(
          (behavior.item(), kl.item(), route.item(), foot_path.item())
        )
        updates += 1
        if updates % self.gradient_length == 0:
          self._step()
          self.optimizer.zero_grad()
    if updates % self.gradient_length:
      self._step()
    self.storage.clear()
    values = (totals / updates).tolist()
    return dict(zip(("behavior", "kl", "route", "foot_path"), values, strict=True)) | {
      "kl_beta": beta
    }

  def _step(self) -> None:
    if self.is_multi_gpu:
      self.reduce_parameters()
    if self.max_grad_norm:
      torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
    self.optimizer.step()
