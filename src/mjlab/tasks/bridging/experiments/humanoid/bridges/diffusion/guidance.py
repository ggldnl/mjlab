"""What a crossing is asked for, applied while the window is being sampled.

The model knows nothing about targets. Everything the bridge wants is said here, at
inference, by correcting the denoiser's prediction at every step of the ladder. One frozen
model therefore answers questions nobody trained it on, which is the claim BeyondMimic
makes and the reason this architecture is worth having beside the imitation bridge.

Arrival is the only cost written, because reaching a dynamic state by a deadline is the
whole problem. Another one goes here beside it: anything with an `apply` that takes a
predicted window and returns a corrected one is a cost, and a waypoint, a speed limit or a
signed distance field to an obstacle all fit that shape.

Why a blend rather than a gradient step

Classifier guidance nudges the sample by the gradient of a cost. Arrival's cost is a
squared distance to a fixed target, so its gradient is proportional to the gap itself, and
a preconditioned step of size a is exactly

    x  <-  (1 - a) x + a target

which is a blend. Writing it this way means the step size is a fraction rather than a
number with units, a is scale free across channels measured in metres and radians per
second alike, and the two ends of its range are the two things anyone wants: zero is an
unguided sample and one is hard inpainting of the target column, the boundary value problem
stated exactly. A cost that is not quadratic has to do the gradient itself and can still
live behind the same `apply`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.data import (
  Layout,
  Normalizer,
  encode,
)


class Cost(Protocol):
  """Anything that corrects a predicted window toward what is being asked for."""

  def apply(self, window: torch.Tensor) -> torch.Tensor:
    """(N, T, F) -> (N, T, F), in normalized feature units."""
    ...


def state_mask(layout: Layout, device: torch.device | str) -> torch.Tensor:
  """Which features a target dynamic state actually names. (F,) in {0, 1}.

  The root and the joints, and nothing else. An action is not part of a state, and body
  positions are forward kinematics the target was never recorded with, so pulling either
  toward a zero would be guiding the sample toward a pose nobody asked for.
  """
  mask = torch.zeros(layout.width, device=device)
  mask[layout.root_pos] = 1.0
  mask[layout.root_ori] = 1.0
  mask[layout.root_lin_vel] = 1.0
  mask[layout.root_ang_vel] = 1.0
  mask[layout.joint_pos] = 1.0
  mask[layout.joint_vel] = 1.0
  return mask


@dataclass
class Arrival:
  """Be in this dynamic state at this column, and stay there afterwards.

  Built fresh every time a plan is drawn, because the deadline column moves one tick closer
  on every control step.
  """

  target: torch.Tensor
  """(N, 1, F) the target state, encoded against the same anchor as the window and
  normalized the same way."""

  weight: torch.Tensor
  """(N, T, F) how far each feature of each column is moved toward the target, in [0, 1]."""

  def apply(self, window: torch.Tensor) -> torch.Tensor:
    return window + self.weight * (self.target - window)


def aim(
  layout: Layout,
  normalizer: Normalizer,
  target: torch.Tensor,
  anchor: torch.Tensor,
  deadline: torch.Tensor,
  columns: int,
  history: int,
  strength: float = 1.0,
  hold: float = 0.5,
) -> Arrival:
  """Build the cost for one plan.

  Args:
    target: (N, 13 + 2J) where the crossing has to end up, in world coordinates.
    anchor: (N, 13 + 2J) the state the plan is drawn from, which is the same tick the
      window is encoded against.
    deadline: (N,) control ticks left until the target is due, counted from the anchor.
    columns, history: the window this cost is applied to.
    strength: how hard the deadline column is pulled. One pins it outright.
    hold: how hard the columns past the deadline are pulled. A crossing that arrives and
      keeps going has not arrived, and the corpus never shows what follows a hand-over, so
      this is a demand rather than something the model would produce on its own.

  A deadline past the end of the window is clamped to the last column, which asks for the
  target as late as the model can be asked for it. A deadline already spent is clamped to
  the first predicted column, which asks for it now. Both are honest answers and neither
  raises: at inference the window ends where it ends, and a crossing that is late is still
  being driven.
  """
  device = target.device

  encoded = encode(layout, target.unsqueeze(1), anchor)
  pinned = normalizer(encoded)

  column = (history - 1 + deadline).clamp(min=history, max=columns - 1)
  index = torch.arange(columns, device=device).unsqueeze(0)
  at = (index == column.unsqueeze(-1)).float()
  after = (index > column.unsqueeze(-1)).float()

  mask = state_mask(layout, device)
  weight = (strength * (at + hold * after)).clamp(0.0, 1.0)
  return Arrival(target=pinned, weight=weight.unsqueeze(-1) * mask.view(1, 1, -1))
