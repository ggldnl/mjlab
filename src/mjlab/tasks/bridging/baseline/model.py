"""PyTorch port of the authors' transition and proximity networks."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F

UPSTREAM_COMMIT = "79ab8e412b21df47779c97b4bef796d5be4a7660"


def training_distance(metadata: dict) -> float | None:
  """Return the fixed distance, or None for earlier multi-distance jump/walk runs."""
  config = metadata["config"]
  distance = config.get("distance")
  if distance is None:
    if config.get("scene", {}).get("pair") in ("walk2jump", "jump2walk"):
      distances = config.get("distances", ())
      if not distances or any(not math.isfinite(x) or x <= 0 for x in distances):
        raise ValueError("Checkpoint training distances must be finite and positive")
      return None
    raise ValueError(
      "Walk2kick requires fixed distance training and effective kick labels; retrain that transition policy"
    )
  distance = float(distance)
  if not math.isfinite(distance) or distance <= 0:
    raise ValueError("Checkpoint training distance must be finite and positive")
  return distance


def fingerprint(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


class RunningMeanStd(nn.Module):
  """Upstream running statistics with a 0.01 variance floor."""

  total: torch.Tensor
  squared: torch.Tensor
  count: torch.Tensor

  def __init__(self, size: int) -> None:
    super().__init__()
    self.register_buffer("total", torch.zeros(size, dtype=torch.float64))
    self.register_buffer("squared", torch.full((size,), 0.01, dtype=torch.float64))
    self.register_buffer("count", torch.tensor(0.01, dtype=torch.float64))

  @torch.no_grad()
  def update(self, observations: torch.Tensor) -> None:
    if observations.ndim != 2 or observations.shape[1] != self.total.numel():
      raise ValueError("Expected a matrix matching the observation dimension")
    if not bool(torch.isfinite(observations).all()):
      raise ValueError("Observations must be finite")
    values = observations.double()
    self.total.add_(values.sum(0))
    self.squared.add_(values.square().sum(0))
    self.count.add_(len(values))

  def forward(self, observations: torch.Tensor) -> torch.Tensor:
    mean = (self.total / self.count).float()
    variance = (self.squared / self.count).float() - mean.square()
    return ((observations - mean) / variance.clamp_min(0.01).sqrt()).clamp(-5, 5)


def normc(layer: nn.Linear, std: float = 1.0) -> nn.Linear:
  with torch.no_grad():
    layer.weight.normal_()
    layer.weight.mul_(std / layer.weight.square().sum(1, keepdim=True).sqrt())
    layer.bias.zero_()
  return layer


class TransitionPolicy(nn.Module):
  """One destination controller with a shared action and termination trunk."""

  def __init__(
    self,
    observation_dim: int,
    action_dim: int,
    num_primitives: int = 2,
    primitive_index: int = 1,
  ) -> None:
    super().__init__()
    if (
      observation_dim < 1 or action_dim < 1 or not 0 <= primitive_index < num_primitives
    ):
      raise ValueError("Invalid policy dimensions or primitive index")
    self.observation_dim, self.action_dim = observation_dim, action_dim
    self.num_primitives, self.primitive_index = num_primitives, primitive_index
    self.ob_rms = RunningMeanStd(observation_dim)
    self.proximity_rms = RunningMeanStd(observation_dim)
    inputs = observation_dim + num_primitives
    self.actor = nn.Sequential(
      normc(nn.Linear(inputs, 32)),
      nn.Tanh(),
      normc(nn.Linear(32, 32)),
      nn.Tanh(),
    )
    self.action_head = normc(nn.Linear(32, action_dim), 0.01)
    self.termination_head = normc(nn.Linear(32, 2), 0.01)
    self.critic = nn.Sequential(
      normc(nn.Linear(inputs, 32)),
      nn.Tanh(),
      normc(nn.Linear(32, 32)),
      nn.Tanh(),
      normc(nn.Linear(32, 1)),
    )
    self.log_std = nn.Parameter(torch.zeros(action_dim))
    self.proximity = nn.Sequential(
      nn.Linear(observation_dim, 96),
      nn.ReLU(),
      nn.Linear(96, 96),
      nn.ReLU(),
      nn.Linear(96, 1),
    )
    for layer in self.proximity:
      if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

  def inputs(self, observations: torch.Tensor, primitive: torch.Tensor | None = None):
    if primitive is None:
      primitive = torch.full(
        observations.shape[:-1],
        self.primitive_index,
        device=observations.device,
        dtype=torch.long,
      )
    identity = F.one_hot(primitive.long(), self.num_primitives).to(observations)
    return torch.cat((self.ob_rms(observations), identity), -1)

  def forward(self, observation: torch.Tensor) -> torch.Tensor:
    return self.action_head(self.actor(self.inputs(observation)))

  def distribution(self, observation: torch.Tensor) -> Normal:
    return Normal(self(observation), self.log_std.exp())

  def termination(self, observation: torch.Tensor) -> Categorical:
    return Categorical(
      logits=self.termination_head(self.actor(self.inputs(observation)))
    )

  def value(self, observation: torch.Tensor) -> torch.Tensor:
    return self.critic(self.inputs(observation)).squeeze(-1)

  def proximity_raw(self, observation: torch.Tensor) -> torch.Tensor:
    return self.proximity(self.proximity_rms(observation)).squeeze(-1)

  def potential(self, observation: torch.Tensor) -> torch.Tensor:
    return self.proximity_raw(observation).clamp(0, 1)

  @torch.no_grad()
  def act(self, observation: torch.Tensor, stochastic: bool = False):
    actions = self.distribution(observation)
    termination = self.termination(observation)
    return (
      actions.sample() if stochastic else actions.mean,
      self.value(observation),
      termination.sample()
      if stochastic
      else cast(torch.Tensor, termination.logits).argmax(-1),
    )

  def policy_parameters(self):
    return [
      *self.actor.parameters(),
      *self.action_head.parameters(),
      *self.termination_head.parameters(),
      *self.critic.parameters(),
      self.log_std,
    ]

  def save(self, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
      {
        "version": 2,
        "upstream_commit": UPSTREAM_COMMIT,
        "observation_dim": self.observation_dim,
        "action_dim": self.action_dim,
        "num_primitives": self.num_primitives,
        "primitive_index": self.primitive_index,
        "model": self.state_dict(),
        "metadata": metadata,
      },
      temporary,
    )
    temporary.replace(path)

  @classmethod
  def load(cls, path: Path, device: str = "cpu") -> tuple[TransitionPolicy, dict]:
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get("version") != 2:
      raise ValueError(
        "Old adapted baseline checkpoint; retrain with the faithful architecture"
      )
    policy = cls(
      saved["observation_dim"],
      saved["action_dim"],
      saved["num_primitives"],
      saved["primitive_index"],
    ).to(device)
    policy.load_state_dict(saved["model"])
    return policy.eval(), saved["metadata"]


def proximity_targets(success: torch.Tensor, steps: int, decay: float) -> torch.Tensor:
  """The last action and the termination record both receive label one upstream."""
  if steps < 1 or not 0 < decay <= 1:
    raise ValueError("steps must be positive and decay must be in (0, 1]")
  remaining = (torch.arange(steps - 1, -1, -1, device=success.device) - 1).clamp_min(0)
  return decay ** remaining[:, None] * success.float()[None]


def proximity_rewards(potentials: torch.Tensor, terminal: torch.Tensor) -> torch.Tensor:
  """Pre-action difference rewards and the upstream final proximity bonus."""
  if potentials.ndim != 1 or terminal.shape != potentials.shape or not len(potentials):
    raise ValueError("Expected one nonempty transition trace")
  previous = torch.cat((potentials.new_zeros(1), potentials[:-1]))
  rewards = torch.where(terminal, potentials, potentials - previous)
  if not bool(terminal[-1]):
    rewards[-1] += potentials[-1]
  return rewards


def advantages(
  rewards: torch.Tensor, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Transition GAE with gamma 0.99 and lambda 0.98."""
  result = torch.empty_like(rewards)
  last = rewards.new_zeros(())
  for index in range(len(rewards) - 1, -1, -1):
    next_value = values[index + 1] if index + 1 < len(values) else 0.0
    delta = rewards[index] + 0.99 * next_value - values[index]
    last = delta + 0.99 * 0.98 * last
    result[index] = last
  return result, result + values
