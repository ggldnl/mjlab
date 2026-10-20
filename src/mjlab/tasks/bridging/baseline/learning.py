"""Replay and optimization ported from rl/trainer.py and rl/proximity_predictor.py."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import math
from collections import deque

import torch

from mjlab.tasks.bridging.baseline.model import TransitionPolicy


class UpstreamAdam(torch.optim.Adam):
  """Match MpiAdam's epsilon placement without MPI on a single training host."""

  def __init__(self, parameters, lr: float):
    super().__init__(parameters, lr=lr, eps=1e-8)
    self.updates = 0

  def step(self, closure=None):
    self.updates += 1
    for group in self.param_groups:
      group["eps"] = 1e-8 / math.sqrt(1 - group["betas"][1] ** self.updates)
    return super().step(closure)


class Replay:
  """A bounded state buffer and permanent collected initiation examples."""

  def __init__(self, capacity: int = 1_000_000):
    if capacity < 1:
      raise ValueError("Replay capacity must be positive")
    self.capacity = capacity
    self.chunks: deque[torch.Tensor] = deque()
    self.size = 0
    self.collected: list[torch.Tensor] = []

  def add(self, observations: torch.Tensor, targets: torch.Tensor, collected=False):
    if not len(observations):
      return
    rows = torch.cat((observations.detach().cpu(), targets.detach().cpu()[:, None]), -1)
    if collected:
      self.collected.append(rows)
      return
    rows = rows[-self.capacity :]
    self.chunks.append(rows)
    self.size += len(rows)
    while self.size > self.capacity:
      excess = self.size - self.capacity
      first = self.chunks.popleft()
      if len(first) > excess:
        self.chunks.appendleft(first[excess:])
      self.size -= min(len(first), excess)

  def tensors(self):
    online = torch.cat(list(self.chunks)) if self.chunks else None
    collected = torch.cat(self.collected) if self.collected else None
    return online, collected

  @staticmethod
  def sample(pools, count: int, device: torch.device):
    online, collected = pools
    if online is None and collected is None:
      raise ValueError("Cannot sample empty replay")
    parts = []
    for pool in (online, collected):
      if pool is not None:
        size = count // 2 if online is not None and collected is not None else count
        parts.append(pool[torch.randint(len(pool), (size,))])
    return torch.cat(parts).to(device)


def joint_log_probability(model, observations, actions, terminal):
  action = model.distribution(observations).log_prob(actions).sum(-1)
  stop = model.termination(observations).log_prob(terminal.long())
  return (1 - terminal.float()) * action + stop


def update_ppo(
  model: TransitionPolicy, optimizer, batch: dict, cfg, multiplier=1.0
) -> float:
  observations, actions = batch["observations"], batch["actions"]
  terminal, returns = batch["terminal"], batch["returns"]
  advantage = batch["advantages"]
  advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(
    1e-6
  )
  # Upstream updates statistics before copying the current policy to oldpi
  model.ob_rms.update(observations)
  with torch.no_grad():
    old_action = model.distribution(observations).log_prob(actions).sum(-1)
    old_stop = model.termination(observations).log_prob(terminal.long()).clamp(-20, 20)
  for group in optimizer.param_groups:
    group["lr"] = cfg.learning_rate * multiplier
  loss = observations.new_zeros(())
  count, limit = 0, cfg.ppo_epochs * cfg.num_trans_batches
  for _ in range(cfg.ppo_epochs):
    for ids in torch.randperm(len(observations), device=observations.device).split(
      cfg.minibatch_size
    ):
      if count == limit:
        break
      distribution = model.distribution(observations[ids])
      action_delta = distribution.log_prob(actions[ids]).sum(-1) - old_action[ids]
      stop_delta = (
        model.termination(observations[ids]).log_prob(terminal[ids].long())
        - old_stop[ids]
      )
      ratio = ((1 - terminal[ids].float()) * action_delta + stop_delta).exp()
      clip = cfg.clip_param * multiplier
      surrogate = torch.minimum(
        ratio * advantage[ids], ratio.clamp(1 - clip, 1 + clip) * advantage[ids]
      )
      loss = (
        -surrogate.mean()
        + (model.value(observations[ids]) - returns[ids]).square().mean()
        - cfg.entropy_coef * distribution.entropy().sum(-1).mean()
      )
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.policy_parameters(), 10.0)
      optimizer.step()
      count += 1
  # The reference advances Adam with zero gradients for missing minibatches
  for _ in range(limit - count):
    for parameter in model.policy_parameters():
      parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
  return float(loss.detach())


def fit_proximity(
  model: TransitionPolicy,
  optimizer,
  success: Replay,
  failure: Replay,
  updates: int,
  batch_size: int,
) -> float:
  positive, negative = success.tensors(), failure.tensors()
  if all(pool is None for pool in positive) or all(pool is None for pool in negative):
    for _ in range(updates):
      for parameter in model.proximity.parameters():
        parameter.grad = torch.zeros_like(parameter)
      optimizer.step()
    return 0.0
  device = next(model.parameters()).device
  loss = torch.zeros((), device=device)
  for _ in range(updates):
    good = Replay.sample(positive, batch_size, device)
    bad = Replay.sample(negative, batch_size, device)
    loss = (model.proximity_raw(good[:, :-1]) - good[:, -1]).square().mean() + (
      model.proximity_raw(bad[:, :-1]) - bad[:, -1]
    ).square().mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.proximity.parameters(), 10.0)
    optimizer.step()
  return float(loss.detach())
