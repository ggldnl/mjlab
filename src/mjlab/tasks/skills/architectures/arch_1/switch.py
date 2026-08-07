"""The hand-over decider, and the signal it learns from. arch_1 and arch_2 only.

Learning to move like the target skill needs no notion of success: that is a copying
game (see common/imitation.py). Only learning when to let go needs one, and only an
architecture with a decider needs that. arch_3 and arch_4_backup hand over on a schedule and
import none of this.

A `SuccessFn` is deliberately a plain function rather than a class: both architectures
that have one answer a single question and neither keeps state. arch_1 supplies a test
the experiment author wrote; arch_2 supplies `always_ok` and lets survival, which the
caller already requires, be the whole judgment.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.architectures.common.nets import mlp

# Given the env, a bool per env: is the robot in a state the target skill can work from?
# Privileged and external, never read off the skill itself (see skill.py). Called at each
# env's own evaluation moment, so it must be a pure function of the current state.
SuccessFn = Callable[[ManagerBasedRlEnv], torch.Tensor]


def always_ok(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Every hand-over passes. The deliberate absence of a judgment (arch_2).

  The caller already requires survival of every hand-over it counts, so this adds nothing
  on top. On a task whose failure mode is a termination (the diffdrive tips, a robot
  falls) it is exactly the right signal and costs no hand-written function. On a task
  that never terminates it makes every hand-over look equally good.
  """
  return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)


class SwitchQNetwork(nn.Module):
  """state -> hidden -> 2. Action 0 is "stay", action 1 is "hand over now".

  Reads the same standardized state the actor does.
  """

  def __init__(self, obs_dim: int, hidden_dims: tuple[int, ...]) -> None:
    super().__init__()
    self.q = mlp(obs_dim, hidden_dims, 2)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.q(obs)


class DoubleDQN:
  """Double-DQN training wrapper around a given `SwitchQNetwork`.

  Trains the passed-in `q` in place, so the caller keeps the same net for inference, and
  holds its own target copy for the Bellman update.
  """

  def __init__(
    self, q: SwitchQNetwork, gamma: float, learning_rate: float, device: str
  ) -> None:
    self.q = q.to(device)
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
