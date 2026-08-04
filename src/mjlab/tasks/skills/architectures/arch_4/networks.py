"""What arch_4's policy reads, and how its reward is scored.

There is no network of arch_4's own here. The bridge is an ordinary rsl_rl actor built
the way the whole family builds one (`build_actor` in architectures/common/nets.py), so
what this module owns is the two things that are specific to arch_4: the vector that
goes in, and the tracking score that comes out.

The vector is proprioception plus the two context windows plus where in the gap the
policy is. Note what is not in it: the reference the policy is being rewarded against.
That absence is the architecture. A tracking policy shown its reference learns to follow
it; a policy shown only the two ends of a hole and rewarded for what was inside has to
learn what usually goes in such holes, which is the thing that transfers to a hole no
clip ever contained.
"""

from __future__ import annotations

import torch

from mjlab.tasks.skills.architectures.arch_4.frames import Groups


def masked_input(
  state: torch.Tensor,
  past: torch.Tensor,
  future: torch.Tensor,
  progress: torch.Tensor,
  length: torch.Tensor,
) -> torch.Tensor:
  """The bridge's whole question, as one vector.

  `state` is the standardized proprioceptive observation (the experiment's state view,
  see view.py). `past` and `future` are standardized frame windows, (N, steps, frame).
  `progress` is how far through the gap this step is, on a 0-to-1 scale, and `length`
  the gap's own length relative to the longest one trained on.

  Both scalars have to be here. Without `progress` the policy sees the same state early
  in a gap, where it should still be leaving the old motion, and late in one, where it
  should already have arrived at the new one, and averages the two. Without `length` it
  cannot tell a gap it has half a second to cross from one it has a second and a
  quarter, and the answer is not the same motion at all.
  """
  return torch.cat(
    [
      state,
      past.flatten(start_dim=1),
      future.flatten(start_dim=1),
      progress.unsqueeze(-1),
      length.unsqueeze(-1),
    ],
    dim=-1,
  )


def input_dim(
  state_dim: int, frame_dim: int, past_steps: int, future_steps: int
) -> int:
  return state_dim + (past_steps + future_steps) * frame_dim + 2


class TrackingScore:
  """How closely a frame reproduces another, group by group.

  One exponential kernel per group of the frame vector (see frames.py), summed with
  weights. Kernels rather than a plain negative distance for the usual reason a tracking
  reward uses them: an error of ten metres per second and an error of twenty are both
  simply wrong, and a policy that cannot yet track should not be handed a gradient that
  is dominated by how spectacularly it is failing.

  Both signs of that choice are visible in the training log: `score` is bounded above by
  the sum of the weights, so a rollout scoring near it is tracking well, and a rollout
  scoring near zero is not tracking at all rather than tracking badly. The termination
  penalty is what keeps the second case from being resolved by falling over on purpose.
  """

  def __init__(self, groups: Groups, weights: dict[str, float], stds: dict[str, float]):
    self.groups = groups
    self.weights = weights
    self.stds = stds
    self.total_weight = sum(weights.values())

  def terms(
    self, frame: torch.Tensor, reference: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    """Per group, the reward for this frame against that reference, (N,) each.

    Both are raw frames, not standardized ones: the length scales in the config are in
    the units a person can reason about (radians, metres per second), and standardizing
    first would silently reinterpret every one of them.
    """
    out: dict[str, torch.Tensor] = {}
    for name, channels in self.groups.named():
      error = torch.square(frame[..., channels] - reference[..., channels]).mean(dim=-1)
      std = self.stds[name]
      out[name] = self.weights[name] * torch.exp(-error / (std * std))
    return out

  def score(self, frame: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    terms = self.terms(frame, reference)
    total = next(iter(terms.values()))
    for name, value in list(terms.items())[1:]:
      del name
      total = total + value
    return total
