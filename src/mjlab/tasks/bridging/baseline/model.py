"""State only actor, critic and learned initiation proximity."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal


def fingerprint(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def mlp(inputs: int, outputs: int) -> nn.Sequential:
  return nn.Sequential(
    nn.Linear(inputs, 256),
    nn.ELU(),
    nn.Linear(256, 128),
    nn.ELU(),
    nn.Linear(128, outputs),
  )


class TransitionPolicy(nn.Module):
  """One frozen skill pair per checkpoint. Targets are never actor inputs."""

  mean: torch.Tensor
  scale: torch.Tensor

  def __init__(self, observation_dim: int, action_dim: int) -> None:
    super().__init__()
    self.actor = mlp(observation_dim, action_dim)
    self.critic = mlp(observation_dim, 1)
    self.proximity = mlp(observation_dim, 1)
    self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))
    self.register_buffer("mean", torch.zeros(observation_dim))
    self.register_buffer("scale", torch.ones(observation_dim))
    self.observation_dim = observation_dim
    self.action_dim = action_dim
    output = self.actor[-1]
    assert isinstance(output, nn.Linear)
    nn.init.zeros_(output.weight)
    nn.init.zeros_(output.bias)

  @torch.no_grad()
  def fit_normalizer(self, observations: torch.Tensor) -> None:
    self.mean.copy_(observations.mean(0))
    self.scale.copy_(observations.std(0, unbiased=False).clamp_min(0.1))

  def normalize(self, observation: torch.Tensor) -> torch.Tensor:
    return ((observation - self.mean) / self.scale).clamp(-10.0, 10.0)

  def forward(self, observation: torch.Tensor) -> torch.Tensor:
    return self.actor(self.normalize(observation))

  def distribution(self, observation: torch.Tensor) -> Normal:
    return Normal(self(observation), self.log_std.clamp(-5, 1).exp())

  def value(self, observation: torch.Tensor) -> torch.Tensor:
    return self.critic(self.normalize(observation)).squeeze(-1)

  def potential(self, observation: torch.Tensor) -> torch.Tensor:
    return self.proximity(self.normalize(observation)).squeeze(-1).sigmoid()

  def save(self, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
      {
        "version": 1,
        "observation_dim": self.observation_dim,
        "action_dim": self.action_dim,
        "model": self.state_dict(),
        "metadata": metadata,
      },
      temporary,
    )
    temporary.replace(path)

  @classmethod
  def load(cls, path: Path, device: str = "cpu") -> tuple[TransitionPolicy, dict]:
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved["version"] != 1:
      raise ValueError("Unsupported transition checkpoint version")
    policy = cls(saved["observation_dim"], saved["action_dim"]).to(device)
    policy.load_state_dict(saved["model"])
    return policy.eval(), saved["metadata"]


def proximity_targets(success: torch.Tensor, steps: int, decay: float) -> torch.Tensor:
  """Successful endpoints have proximity one, earlier states decay backward."""
  if steps < 1 or not 0 < decay <= 1:
    raise ValueError("steps must be positive and decay must be in (0, 1]")
  remaining = torch.arange(steps - 1, -1, -1, device=success.device)
  return decay ** remaining[:, None] * success.float()[None]


def returns_to_go(rewards: torch.Tensor) -> torch.Tensor:
  """Undiscounted finite episode returns preserve the proximity difference reward."""
  return rewards.flip(0).cumsum(0).flip(0)
