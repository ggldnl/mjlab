"""The networks arch_1 needs that rsl_rl does not provide.

Two are genuinely new (rsl_rl has no adversarial-imitation or off-policy piece):
- `AIRLDiscriminator`: the referee of the copy-catch game, split into a reward net
  `g` and a potential net `h`.
- `SwitchQNetwork` / `DoubleDQN`: the yes/no "hand over now?" decider and its
  Double-DQN training wrapper.

The bridge's actor is a plain rsl_rl `MLPModel`; `build_bridge_actor` is the single
place it is constructed so the Arch1 meta policy (which holds it for inference) and
train.py (which trains it) cannot drift on how it was built.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.models import MLPModel
from tensordict import TensorDict

# The obs groups a bridge actor/critic read, and the actor's action distribution.
# Shared by build_bridge_actor and train.py's critic so they stay in lock-step.
OBS_GROUPS = {"actor": ["actor"], "critic": ["critic"]}
GAUSSIAN_DISTRIBUTION_CFG = {
  "class_name": "GaussianDistribution",
  "init_std": 1.0,
  "std_type": "scalar",
}


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
) -> MLPModel:
  """Build a bridge's actor the one way arch_1 ever builds it."""
  return MLPModel(
    obs_td,
    OBS_GROUPS,
    "actor",
    action_dim,
    hidden_dims=hidden_dims,
    activation="tanh",
    distribution_cfg=dict(GAUSSIAN_DISTRIBUTION_CFG),
  ).to(device)


@dataclass
class DiscBatch:
  """One half (policy or expert) of a discriminator update, or a reward query."""

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

  Inputs are standardized before either net sees them. Observations and actions come
  in whatever units the experiment uses (the diffdrive commands wheel velocities in
  the tens of rad/s), and feeding those raw into a freshly initialized MLP saturates
  it on the first batch. `fit_normalization` sets the statistics from the expert data
  once, and they are buffers so they travel with a checkpoint.
  """

  # Declared so the registered buffers below have a type; `register_buffer` alone
  # leaves them looking like `Tensor | Module` to a checker.
  obs_mean: torch.Tensor
  obs_std: torch.Tensor
  action_mean: torch.Tensor
  action_std: torch.Tensor

  def __init__(
    self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...], gamma: float
  ) -> None:
    super().__init__()
    self.gamma = gamma
    self.g = _mlp(obs_dim + action_dim, hidden_dims, 1)  # obs+action look right?
    self.h = _mlp(obs_dim, hidden_dims, 1)  # how good is this obs?
    self.register_buffer("obs_mean", torch.zeros(obs_dim))
    self.register_buffer("obs_std", torch.ones(obs_dim))
    self.register_buffer("action_mean", torch.zeros(action_dim))
    self.register_buffer("action_std", torch.ones(action_dim))

  @torch.no_grad()
  def fit_normalization(self, obs: torch.Tensor, actions: torch.Tensor) -> None:
    """Set the input statistics from a batch of expert transitions."""
    self.obs_mean.copy_(obs.mean(dim=0))
    self.obs_std.copy_(obs.std(dim=0).clamp_min(1e-3))
    self.action_mean.copy_(actions.mean(dim=0))
    self.action_std.copy_(actions.std(dim=0).clamp_min(1e-3))

  def f(
    self,
    obs: torch.Tensor,
    actions: torch.Tensor,
    done: torch.Tensor,
    next_obs: torch.Tensor,
  ) -> torch.Tensor:
    obs_n = (obs - self.obs_mean) / self.obs_std
    next_obs_n = (next_obs - self.obs_mean) / self.obs_std
    actions_n = (actions - self.action_mean) / self.action_std
    g = self.g(torch.cat([obs_n, actions_n], dim=-1)).squeeze(-1)
    h = self.h(obs_n).squeeze(-1)
    h_next = self.h(next_obs_n).squeeze(-1)
    return g + self.gamma * (1.0 - done.float()) * h_next - h

  def logits(self, batch: DiscBatch) -> torch.Tensor:
    return self.f(batch.obs, batch.actions, batch.done, batch.next_obs) - batch.log_prob

  def reward(self, batch: DiscBatch) -> torch.Tensor:
    """PPO reward for the bridge: `-logsigmoid(-logit)` (== `softplus(logit)`)."""
    return F.softplus(self.logits(batch))

  def loss(self, policy_batch: DiscBatch, expert_batch: DiscBatch) -> torch.Tensor:
    """Binary cross-entropy: bridge rollouts are fake, target-window rollouts real."""
    policy_loss = -F.logsigmoid(-self.logits(policy_batch)).mean()
    expert_loss = -F.logsigmoid(self.logits(expert_batch)).mean()
    return policy_loss + expert_loss


class SwitchQNetwork(nn.Module):
  """state -> hidden -> 2 Q-network for the hand-over decision.

  Action 0 is "stay" (keep bridging), action 1 is "switch" (hand control to the
  target skill now).
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
