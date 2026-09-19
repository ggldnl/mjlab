"""Residual conditional VAE used by the endpoint bridge."""

from __future__ import annotations

import copy
from typing import cast

import torch
from rsl_rl.modules import MLP, EmpiricalNormalization
from tensordict import TensorDict
from torch import nn


class ResidualCvaeModel(nn.Module):
  """Decode actions from an endpoint condition and a short path latent."""

  is_recurrent = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    latent_dim: int = 16,
    activation: str = "elu",
    obs_normalization: bool = True,
    posterior_obs_set: str = "posterior",
  ) -> None:
    super().__init__()
    if latent_dim < 1:
      raise ValueError("latent_dim must be positive")
    self.obs_groups = obs_groups[obs_set]
    self.posterior_obs_groups = obs_groups[posterior_obs_set]
    self.obs_dim = sum(obs[name].shape[-1] for name in self.obs_groups)
    posterior_dim = sum(obs[name].shape[-1] for name in self.posterior_obs_groups)
    self.output_dim = output_dim
    self.latent_dim = latent_dim
    self.obs_normalization = obs_normalization
    self.obs_normalizer = (
      EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()
    )
    self.posterior_normalizer = (
      EmpiricalNormalization(posterior_dim) if obs_normalization else nn.Identity()
    )

    self.prior = MLP(self.obs_dim, 2 * latent_dim, hidden_dims, activation)
    self.posterior_residual = MLP(
      self.obs_dim + posterior_dim, 2 * latent_dim, hidden_dims, activation
    )
    self.decoder = MLP(self.obs_dim + latent_dim, output_dim, hidden_dims, activation)
    last = next(
      layer
      for layer in reversed(self.posterior_residual)
      if isinstance(layer, nn.Linear)
    )
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)

    self._epsilon: torch.Tensor | None = None
    self._last_latent_std = torch.ones(1)

  def _condition(self, obs: TensorDict) -> torch.Tensor:
    value = torch.cat(
      [cast(torch.Tensor, obs[name]) for name in self.obs_groups], dim=-1
    )
    return self.obs_normalizer(value)

  def _posterior_condition(self, obs: TensorDict) -> torch.Tensor:
    value = torch.cat(
      [cast(torch.Tensor, obs[name]) for name in self.posterior_obs_groups], dim=-1
    )
    return self.posterior_normalizer(value)

  @staticmethod
  def _split_parameters(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean, log_variance = value.chunk(2, dim=-1)
    return mean, log_variance.clamp(-10.0, 4.0)

  def prior_parameters(
    self, obs: TensorDict
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    condition = self._condition(obs)
    mean, log_variance = self._split_parameters(self.prior(condition))
    return condition, mean, log_variance

  def posterior_parameters(
    self, obs: TensorDict
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    condition, prior_mean, prior_log_variance = self.prior_parameters(obs)
    path = self._posterior_condition(obs)
    mean_delta, log_variance_delta = self._split_parameters(
      self.posterior_residual(torch.cat((condition, path), dim=-1))
    )
    return (
      condition,
      prior_mean + mean_delta,
      (prior_log_variance + log_variance_delta).clamp(-10.0, 4.0),
    )

  def _episode_epsilon(self, reference: torch.Tensor) -> torch.Tensor:
    shape = (*reference.shape[:-1], self.latent_dim)
    if self._epsilon is None or self._epsilon.shape != shape:
      self._epsilon = torch.randn(shape, device=reference.device, dtype=reference.dtype)
    return self._epsilon

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state=None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    del masks, hidden_state
    condition, mean, log_variance = self.prior_parameters(obs)
    self._last_latent_std = torch.exp(0.5 * log_variance).detach()
    latent = mean
    if stochastic_output:
      latent = mean + self._last_latent_std * self._episode_epsilon(mean)
    return self.decoder(torch.cat((condition, latent), dim=-1))

  def reconstruct(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode with the privileged path and return KL(q || p)."""
    condition, prior_mean, prior_log_variance = self.prior_parameters(obs)
    path = self._posterior_condition(obs)
    residual_mean, residual_log_variance = self._split_parameters(
      self.posterior_residual(torch.cat((condition, path), dim=-1))
    )
    posterior_mean = prior_mean + residual_mean
    posterior_log_variance = (prior_log_variance + residual_log_variance).clamp(
      -10.0, 4.0
    )
    epsilon = torch.randn_like(posterior_mean)
    latent = posterior_mean + torch.exp(0.5 * posterior_log_variance) * epsilon
    action = self.decoder(torch.cat((condition, latent), dim=-1))

    variance_ratio = torch.exp(posterior_log_variance - prior_log_variance)
    mean_error = (posterior_mean - prior_mean).square() / torch.exp(prior_log_variance)
    kl = 0.5 * (
      prior_log_variance - posterior_log_variance + variance_ratio + mean_error - 1.0
    ).sum(dim=-1)
    return action, kl

  def update_normalization(self, obs: TensorDict) -> None:
    if not self.obs_normalization:
      return
    condition = torch.cat(
      [cast(torch.Tensor, obs[name]) for name in self.obs_groups], dim=-1
    )
    path = torch.cat(
      [cast(torch.Tensor, obs[name]) for name in self.posterior_obs_groups], dim=-1
    )
    cast(EmpiricalNormalization, self.obs_normalizer).update(condition)
    cast(EmpiricalNormalization, self.posterior_normalizer).update(path)

  def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
    del hidden_state
    if self._epsilon is None:
      return
    if dones is None:
      self._epsilon = None
      return
    done = dones.bool().view(-1)
    if done.any():
      self._epsilon[done] = torch.randn_like(self._epsilon[done])

  def get_hidden_state(self):
    return None

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    del dones

  @property
  def output_std(self) -> torch.Tensor:
    return self._last_latent_std.mean()

  def as_jit(self) -> nn.Module:
    return _InferenceModel(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    del verbose
    return _InferenceModel(self)


class _InferenceModel(nn.Module):
  """Deterministic prior-mean policy used for deployment."""

  is_recurrent = False

  def __init__(self, model: ResidualCvaeModel) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.prior = copy.deepcopy(model.prior)
    self.decoder = copy.deepcopy(model.decoder)
    self.latent_dim = model.latent_dim
    self.input_size = model.obs_dim

  def forward(self, observation: torch.Tensor) -> torch.Tensor:
    condition = self.obs_normalizer(observation)
    mean = self.prior(condition)[..., : self.latent_dim]
    return self.decoder(torch.cat((condition, mean), dim=-1))

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
