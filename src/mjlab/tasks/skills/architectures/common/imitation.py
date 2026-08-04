"""The copying game: a referee that tells a transition from the real thing.

arch_1, arch_2 and arch_3 all train the same way. Something rolls out, the target skill's
own recording is the "real" half, and a discriminator is trained to tell them apart while
the policy is trained on its verdict.

Three pieces:

    DiscBatch          one half of an update, or a reward query
    AIRLDiscriminator  the referee, split into a reward net `g` and a potential net `h`
    RewardScaler       what sits between the referee and PPO

Everything arriving here is already standardized/normalized by spaces.py. The referee
used to normalize internally, on statistics fitted from the target skill alone, which
meant it and the actor disagreed about what a number meant.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mjlab.tasks.skills.architectures.common.nets import mlp


@dataclass
class DiscBatch:
  """One half (policy or expert) of a discriminator update, or a reward query.

  `obs`/`next_obs` are standardized and `actions` normalized. `log_prob` takes no part in
  any decision; it is carried for the training log only (see `AIRLDiscriminator.logits`),
  and a caller with nothing to report passes `no_log_prob`.
  """

  obs: torch.Tensor
  actions: torch.Tensor
  next_obs: torch.Tensor
  done: torch.Tensor
  log_prob: torch.Tensor


def no_log_prob(rows: int, device: str) -> torch.Tensor:
  """The likelihood term, switched off, for a caller that has nothing to report."""
  return torch.zeros(rows, device=device)


def called_real(discriminator: AIRLDiscriminator, batch: DiscBatch) -> float:
  """Fraction of a batch the referee calls "expert"."""
  with torch.no_grad():
    return float((discriminator.logits(batch) > 0).float().mean())


class AIRLDiscriminator(nn.Module):
  """The referee: reward net `g` plus potential net `h`.

  `f(s, a, done, s') = g(s, a) + gamma * (1 - done) * h(s') - h(s)` is the learned
  reward. Splitting into g and h keeps two questions apart: `h` asks "is this the kind of
  state the target skill would be in?", `g` asks "given the state, is this the move it
  would make?". A single net blends them into one, less informative, number.
  """

  def __init__(
    self,
    obs_dim: int,
    action_dim: int,
    hidden_dims: tuple[int, ...],
    gamma: float,
  ) -> None:
    super().__init__()
    self.gamma = gamma
    self.g = mlp(obs_dim + action_dim, hidden_dims, 1)  # Does obs+action look right?
    self.h = mlp(obs_dim, hidden_dims, 1)  # How good is this obs?

  def f(
    self,
    obs: torch.Tensor,
    actions: torch.Tensor,
    done: torch.Tensor,
    next_obs: torch.Tensor,
  ) -> torch.Tensor:
    """The learned reward: does this transition look like the target skill's own?"""
    g = self.g(torch.cat([obs, actions], dim=-1)).squeeze(-1)
    h = self.h(obs).squeeze(-1)
    h_next = self.h(next_obs).squeeze(-1)
    return g + self.gamma * (1.0 - done.float()) * h_next - h

  def logits(self, batch: DiscBatch) -> torch.Tensor:
    """The verdict: `f` alone, judged on the movement and nothing else.

    Textbook AIRL puts the action's log-probability here too, as `f - log_pi`, which is
    what makes its optimum recover a reward that transfers across dynamics. It is dropped
    deliberately, because that term does not survive contact with a many-jointed robot.

    A diagonal Gaussian over `d` joints has

        -log_pi = sum over joints of [ (a-mu)^2 / 2 sigma^2 + log sigma + 0.919 ]
               ~= d * (1.419 + log sigma)

    a sum over joints, while `f` is one network output of whatever size it likes. So the
    balance is set by the robot, not the problem: negligible on a cart-pole (d=1),
    dominant on a humanoid (d=29). Once it dominates, the referee separates the halves
    without looking at the movement at all -- the target's actions are unlikely under the
    policy's habits simply because they are different actions -- and a referee that wins
    for free teaches nothing. Worse, it gets easier the more consistent the policy
    becomes, so improving is punished. Measured on the two-wheeled robot: +22 against an
    `f` gap of -1.4.

    Without it this is GAIL with an AIRL-shaped referee. The `g`/`h` split stays, which is
    what makes the reward smooth; the transferability of `f` goes, and no experiment here
    used it. The exploration the term used to supply is PPO's `entropy_coef`, a dial we
    set rather than one the joint count sets for us.
    """
    return self.f(batch.obs, batch.actions, batch.done, batch.next_obs)

  def split_logits(self, batch: DiscBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """`f` and the log-probability, for the training log. Diagnostic only.

    The second takes no part in any decision. It is worth watching because it is the
    quantity that broke this twice; a run drifting back toward it should be visible.
    """
    return self.f(batch.obs, batch.actions, batch.done, batch.next_obs), batch.log_prob

  def reward(self, batch: DiscBatch) -> torch.Tensor:
    """The per-step reward before scaling: the learned reward `f`, raw and unbounded.

    `f` is a network output and its scale drifts with the state of the game, so it is
    scaled by a `RewardScaler` rather than bounded here. Clipping it here, which is what
    this used to do, destroys the thing that makes it a reward: once `f` sits below the
    bound for every transition, every transition scores exactly the bound.
    """
    return self.f(batch.obs, batch.actions, batch.done, batch.next_obs)

  def loss(self, policy_batch: DiscBatch, expert_batch: DiscBatch) -> torch.Tensor:
    """Binary cross-entropy: policy rollouts are fake, recorded windows real."""
    policy_loss = -F.logsigmoid(-self.logits(policy_batch)).mean()
    expert_loss = -F.logsigmoid(self.logits(expert_batch)).mean()
    return policy_loss + expert_loss


class RewardScaler:
  """Keeps the imitation reward centered and at unit size, whatever the referee does.

  Two jobs, both of which turned out to matter.

  Size. `f` is unbounded and its scale drifts; it reached -20 while the two halves were
  far apart. PPO standardizes advantages so the policy is indifferent, but the critic
  fits raw returns, and -20 a step over a 64-step rollout is a target in the hundreds for
  a small MLP. That is the critic error climbing into the thousands in every log we have.

  Center. A reward that is negative *everywhere* pays the policy to end the episode:
  every extra step costs, so crashing truncates the sum and scores better than surviving.
  Measured: with the reward pinned at the old bound of -10, terminations climbed from 2%
  to 31% over eighty iterations. Subtracting the running mean removes the incentive in
  both directions, and what a fall actually costs is then said once, explicitly, by the
  trainer's `termination_penalty`.

  Deliberately not a module and deliberately not saved: it shapes what the policy is told
  during training and has no meaning at inference.
  """

  # How fast the running estimates follow the referee. Slow enough to be a stable divisor
  # across an iteration, fast enough to track a referee that is still moving.
  MOMENTUM = 0.01

  def __init__(self, device: str, clip: float = 5.0) -> None:
    self.mean = torch.zeros((), device=device)
    self.var = torch.ones((), device=device)
    self.clip = clip
    self._started = False

  @torch.no_grad()
  def __call__(self, reward: torch.Tensor) -> torch.Tensor:
    """Update the running estimates from this batch, and return the scaled reward."""
    batch_mean = reward.mean()
    batch_var = reward.var(unbiased=False)
    if not self._started:
      self.mean.copy_(batch_mean)
      self.var.copy_(batch_var)
      self._started = True
    else:
      self.mean.lerp_(batch_mean, self.MOMENTUM)
      self.var.lerp_(batch_var, self.MOMENTUM)
    scaled = (reward - self.mean) / self.var.clamp_min(1e-8).sqrt()
    # In standard deviations rather than the referee's units, so it catches a genuine
    # outlier and never binds on ordinary transitions the way a bound in raw units did.
    return scaled.clamp(-self.clip, self.clip)
