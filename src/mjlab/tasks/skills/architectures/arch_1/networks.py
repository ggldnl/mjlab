"""The networks arch_1 needs that rsl_rl does not provide.

Two are genuinely new (rsl_rl has no adversarial-imitation or off-policy piece):
- `AIRLDiscriminator`: the referee of the copy-catch game, split into a reward net
  `g` and a potential net `h`.
- `SwitchQNetwork` / `DoubleDQN`: the yes/no "hand over now?" decider and its
  Double-DQN training wrapper.

The bridge's actor is a plain rsl_rl `MLPModel`; `build_bridge_actor` is the single
place it is constructed so the Arch1 meta policy (which holds it for inference) and
train.py (which trains it) cannot drift on how it was built.

Nothing here converts units. Every tensor arriving at any network in this file is
already in the bridge's own units: observations standardized and actions normalized by
the `StateSpace` / `ActionSpace` in spaces.py. That used to be done inside the
discriminator, on statistics fitted from the target skill alone, which meant the actor
and the discriminator disagreed about what a number meant and the log-probability
appearing in the AIRL logit was left in raw units entirely. Keeping the conversion in
one place, upstream of everything, is what makes the quantities in `logits` below
comparable.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.models import MLPModel
from tensordict import TensorDict

# The obs groups a bridge actor/critic read, and the actor's action distribution.
# Shared by build_bridge_actor and train.py's critic so they stay in lock-step.
OBS_GROUPS = {"actor": ["actor"], "critic": ["critic"]}


def bridge_obs_td(state: torch.Tensor) -> TensorDict:
  """Pack an already-standardized state into the TensorDict a bridge's nets read.

  The actor and the throwaway critic PPO needs are built against the groups in
  `OBS_GROUPS`, so both entries hold the same vector: the experiment's state view of
  the observation (see view.py), standardized by the run's `StateSpace`. There is no
  privileged critic input here on purpose -- the bridge is being taught to match a
  distribution the discriminator judges on this exact vector, and a critic seeing more
  than the discriminator does would be valuing something the reward cannot express.
  """
  return TensorDict({"actor": state, "critic": state}, batch_size=[int(state.shape[0])])


def _mlp(
  input_dim: int, hidden_dims: tuple[int, ...], output_dim: int
) -> nn.Sequential:
  dims = (input_dim, *hidden_dims, output_dim)
  layers: list[nn.Module] = []
  for i in range(len(dims) - 1):
    layers.append(nn.Linear(dims[i], dims[i + 1]))
    if i < len(dims) - 2:
      layers.append(nn.ReLU())
  return nn.Sequential(*layers)


def build_bridge_actor(
  obs_td: TensorDict,
  action_dim: int,
  hidden_dims: tuple[int, ...],
  device: str,
  init_std: float = 0.4,
) -> MLPModel:
  """Build a bridge's actor the one way arch_1 ever builds it.

  `init_std` is in *normalized* action units (see spaces.py), where the range the pool's
  skills command spans roughly [-1, 1]. It is therefore directly readable as "how much
  of the useful action range does this policy explore": 0.4 is a fifth of that range per
  step, which is a lot but not so much that the robot is being driven at random.

  This number used to be 1.0 while the actor emitted raw wheel velocities spanning some
  forty units, i.e. two percent exploration, which is why the bridge never found the
  braking behavior it was being asked for.
  """
  return MLPModel(
    obs_td,
    OBS_GROUPS,
    "actor",
    action_dim,
    hidden_dims=hidden_dims,
    activation="tanh",
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": init_std,
      "std_type": "log",
    },
  ).to(device)


def _std_parameter(actor: MLPModel) -> tuple[torch.Tensor, bool]:
  """The actor's learnable std, and whether it is held in log space.

  rsl_rl's Gaussian keeps this under one of two names depending on how it was
  parameterized; `build_bridge_actor` always asks for the log one, but reading both
  means the three helpers below keep working if that choice ever changes.
  """
  distribution = actor.distribution
  log_std = getattr(distribution, "log_std_param", None)
  if isinstance(log_std, torch.Tensor):
    return log_std, True
  std = getattr(distribution, "std_param", None)
  if isinstance(std, torch.Tensor):
    return std, False
  raise AttributeError(
    f"{type(distribution).__name__} has neither `log_std_param` nor `std_param`, so "
    f"its exploration cannot be read or bounded."
  )


def action_std(actor: MLPModel) -> float:
  """The actor's mean exploration, in normalized action units. For logging."""
  parameter, is_log = _std_parameter(actor)
  with torch.no_grad():
    return float(parameter.exp().mean() if is_log else parameter.mean())


@torch.no_grad()
def set_action_std(actor: MLPModel, std: float) -> None:
  """Set the actor's exploration to `std`, in normalized action units.

  `build_bridge_actor` picks a default when the network is constructed, which is before
  any training config has been seen; phase 1 calls this so that the number the
  experiment actually declared (`BridgePhase.action_init_std`) is the one that applies.
  """
  parameter, is_log = _std_parameter(actor)
  parameter.fill_(math.log(std) if is_log else std)


@torch.no_grad()
def clamp_action_std(actor: MLPModel, max_std: float) -> None:
  """Hold the actor's exploration below `max_std`, in normalized action units.

  rsl_rl's Gaussian keeps `log_std` as a free parameter and nothing bounds it, so the
  entropy bonus in PPO's objective pushes it up without limit whenever the surrogate
  term is weaker than the bonus. That is not a hypothetical trade-off here: a bridge is
  rewarded by a discriminator that a poor bridge cannot move much, so the surrogate is
  small from the start, and `log_std` then grows by roughly a percent an iteration
  forever. Over a humanoid-sized budget of a few thousand iterations that is `exp` of a
  large number: the std overflows to inf, the next gradient is NaN, and PPO dies inside
  `Normal` with "expects all elements of std >= 0.0" -- a NaN, not a negative number.

  Clamping rather than lowering `entropy_coef` alone because the two do different jobs.
  The coefficient sets how hard the run is pushed toward exploring; this sets the point
  past which more exploration is not exploration at all. In normalized units the pool's
  whole commanded range is about [-1, 1], so a std of 1 already means the actor's own
  mean barely matters.
  """
  parameter, is_log = _std_parameter(actor)
  parameter.clamp_max_(math.log(max_std) if is_log else max_std)


@dataclass
class DiscBatch:
  """One half (policy or expert) of a discriminator update, or a reward query.

  `obs` and `next_obs` are standardized, `actions` are normalized, and `log_prob` is the
  actor's log-probability of those *normalized* actions. All three conditions matter;
  see `AIRLDiscriminator.logits`.
  """

  obs: torch.Tensor
  actions: torch.Tensor
  next_obs: torch.Tensor
  done: torch.Tensor
  log_prob: torch.Tensor


class AIRLDiscriminator(nn.Module):
  """AIRL discriminator: reward-shaping net `g` plus potential net `h`.

  `f(s, a, done, s') = g(s, a) + gamma * (1 - done) * h(s') - h(s)` is the learned
  reward. The discriminator logit is `f - log_pi(a|s)`; training it to separate the
  bridge's rollouts from the target skill's own transition-window rollouts pulls `f`
  toward a reward that reproduces the target skill's behavior.

  Splitting into g and h keeps two questions apart: `h` asks "is this the kind of state
  the target skill would be in?", `g` asks "given the state, is this the move it would
  make?". A single discriminator would blend them into one, less informative, number.

  Inputs are expected pre-converted (see the module docstring). This class deliberately
  holds no normalization of its own: it used to, and having the conversion live here
  while the log-probability in the logit stayed in raw units is precisely the imbalance
  described in `logits`.
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
    self.g = _mlp(obs_dim + action_dim, hidden_dims, 1)  # obs+action look right?
    self.h = _mlp(obs_dim, hidden_dims, 1)  # how good is this obs?

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

  def split_logits(self, batch: DiscBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """The two halves of the AIRL logit, kept apart so they can be compared.

    AIRL asks "is this transition better explained by the target skill's reward, or by
    the bridge's own habits?", and answers it by subtracting one from the other. That
    only works while the two are the same order of magnitude.

    If the likelihood term is much the larger of the two, it answers the question by
    itself: the expert's actions are unlikely under the bridge's policy simply because
    they are different actions, so the referee gets the right answer for free, its loss
    saturates, and no gradient ever reaches `g` and `h`. The bridge is then rewarded by
    a network that never learned anything. This is not hypothetical: with raw wheel
    velocities the likelihood term reached the hundreds while `f` sat near zero.

    Normalized actions keep the likelihood term at a few units, which is where `f` also
    lives. train.py logs both means side by side every time it prints, so a run drifting
    back into that state is visible rather than silent.
    """
    learned_reward = self.f(batch.obs, batch.actions, batch.done, batch.next_obs)
    policy_likelihood = batch.log_prob
    return learned_reward, policy_likelihood

  def logits(self, batch: DiscBatch) -> torch.Tensor:
    learned_reward, policy_likelihood = self.split_logits(batch)
    return learned_reward - policy_likelihood

  def reward(self, batch: DiscBatch) -> torch.Tensor:
    """PPO reward for the bridge: `-logsigmoid(-logit)` (== `softplus(logit)`).

    Always positive, which is worth keeping in mind: on its own this reward can never
    punish anything, so a bridge that crashes the robot pays nothing for it here. The
    crash penalty train.py subtracts is what supplies the other sign.
    """
    return F.softplus(self.logits(batch))

  def loss(self, policy_batch: DiscBatch, expert_batch: DiscBatch) -> torch.Tensor:
    """Binary cross-entropy: bridge rollouts are fake, target-window rollouts real."""
    policy_loss = -F.logsigmoid(-self.logits(policy_batch)).mean()
    expert_loss = -F.logsigmoid(self.logits(expert_batch)).mean()
    return policy_loss + expert_loss


class SwitchQNetwork(nn.Module):
  """state -> hidden -> 2 Q-network for the hand-over decision.

  Reads the same standardized state the bridge actor does. Action 0 is "stay" (keep
  bridging), action 1 is "switch" (hand control to the target skill now).
  """

  def __init__(self, obs_dim: int, hidden_dims: tuple[int, ...]) -> None:
    super().__init__()
    self.q = _mlp(obs_dim, hidden_dims, 2)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.q(obs)


class DoubleDQN:
  """Double-DQN training wrapper around a given `SwitchQNetwork`.

  Trains the passed-in `q` in place (so the caller keeps the same net for inference),
  holding its own target copy for the Bellman update.
  """

  def __init__(
    self, q: SwitchQNetwork, gamma: float, learning_rate: float, device: str
  ) -> None:
    self.q = q.to(device)
    # A frozen copy of the online net, synced periodically via `sync_target`.
    self.q_target = copy.deepcopy(self.q).to(device)
    self.q_target.eval()
    for p in self.q_target.parameters():
      p.requires_grad_(False)
    self.gamma = gamma
    self.optimizer = torch.optim.Adam(self.q.parameters(), lr=learning_rate)

  def act(self, obs: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Epsilon-greedy actions, shaped (N,) int64: 0 (stay) or 1 (switch)."""
    with torch.no_grad():
      greedy = self.q(obs).argmax(dim=-1)
    random_actions = torch.randint(0, 2, greedy.shape, device=greedy.device)
    explore = torch.rand(greedy.shape, device=greedy.device) < epsilon
    return torch.where(explore, random_actions, greedy)

  def sync_target(self) -> None:
    self.q_target.load_state_dict(self.q.state_dict())

  def update(
    self,
    obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_obs: torch.Tensor,
    done: torch.Tensor,
  ) -> torch.Tensor:
    """One Double-DQN gradient step; returns the scalar loss."""
    q_values = self.q(obs).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
      next_actions = self.q(next_obs).argmax(dim=-1)
      next_q = self.q_target(next_obs).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
      target = rewards + self.gamma * (1.0 - done.float()) * next_q
    loss = F.mse_loss(q_values, target)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.detach()
