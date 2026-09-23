"""Route-conditioned CVAE with one route and latent per bridge window."""

from __future__ import annotations

import copy
from typing import cast

import torch
from rsl_rl.modules import MLP, EmpiricalNormalization
from tensordict import TensorDict
from torch import nn


class GoalCvaeModel(nn.Module):
  """Choose a touchdown route, then decode actions with a fixed latent."""

  is_recurrent = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    latent_dim: int = 16,
    route_slots: int = 6,
    posterior_obs_set: str = "posterior",
    route_obs_set: str = "route",
    activation: str = "elu",
    obs_normalization: bool = True,
  ) -> None:
    super().__init__()
    if latent_dim < 1 or route_slots < 1:
      raise ValueError("latent_dim and route_slots must be positive")
    live_name, initial_name = obs_groups[obs_set]
    self.live_name = live_name
    self.initial_name = initial_name
    self.path_name = obs_groups[posterior_obs_set][0]
    self.route_name = obs_groups[route_obs_set][0]
    self.live_dim = obs[live_name].shape[-1]
    self.initial_dim = obs[initial_name].shape[-1]
    self.path_dim = obs[self.path_name].shape[-1]
    self.latent_dim = latent_dim
    self.route_slots = route_slots
    self.route_dim = route_slots + 1 + route_slots * 3
    self.input_size = self.live_dim
    self.output_dim = output_dim
    self.obs_normalization = obs_normalization
    self.live_normalizer = (
      EmpiricalNormalization(self.live_dim) if obs_normalization else nn.Identity()
    )
    self.initial_normalizer = (
      EmpiricalNormalization(self.initial_dim) if obs_normalization else nn.Identity()
    )
    self.path_normalizer = (
      EmpiricalNormalization(self.path_dim) if obs_normalization else nn.Identity()
    )

    self.route_prior = MLP(
      self.initial_dim, route_slots + 1 + route_slots * 2, hidden_dims, activation
    )
    self.prior = MLP(
      self.initial_dim + self.route_dim, 2 * latent_dim, hidden_dims, activation
    )
    self.posterior = MLP(
      self.initial_dim + self.route_dim + self.path_dim,
      2 * latent_dim,
      hidden_dims,
      activation,
    )
    self.decoder = MLP(
      self.live_dim + self.route_dim + latent_dim,
      hidden_dims[-1],
      hidden_dims[:-1],
      activation,
    )
    self.action_head = nn.Linear(hidden_dims[-1], output_dim)
    self.path_head = nn.Linear(hidden_dims[-1], self.path_dim)
    self._route: torch.Tensor | None = None
    self._latent: torch.Tensor | None = None
    self._last_latent_std = torch.ones(1)

  @staticmethod
  def _split_parameters(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean, log_variance = value.chunk(2, dim=-1)
    return mean, log_variance.clamp(-10.0, 4.0)

  def _live(self, obs: TensorDict) -> torch.Tensor:
    return self.live_normalizer(cast(torch.Tensor, obs[self.live_name]))

  def _initial(self, obs: TensorDict) -> torch.Tensor:
    return self.initial_normalizer(cast(torch.Tensor, obs[self.initial_name]))

  def _path(self, obs: TensorDict) -> torch.Tensor:
    return self.path_normalizer(cast(torch.Tensor, obs[self.path_name]))

  def _route_vector(self, count: torch.Tensor, feet: torch.Tensor) -> torch.Tensor:
    one_count = torch.nn.functional.one_hot(
      count.long().clamp(0, self.route_slots), self.route_slots + 1
    )
    one_feet = torch.nn.functional.one_hot(feet.long().clamp(0, 2), 3).flatten(1)
    return torch.cat((one_count, one_feet), dim=-1).float()

  def _labels(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    label = cast(torch.Tensor, obs[self.route_name]).long()
    return label[:, 0], label[:, 1:]

  def _route_logits(self, initial: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = self.route_prior(initial)
    return logits[:, : self.route_slots + 1], logits[:, self.route_slots + 1 :].reshape(
      -1, self.route_slots, 2
    )

  def _sample_route(self, initial: torch.Tensor) -> torch.Tensor:
    count_logits, foot_logits = self._route_logits(initial)
    count = torch.distributions.Categorical(logits=count_logits).sample()
    feet = torch.distributions.Categorical(logits=foot_logits).sample() + 1
    slots = torch.arange(self.route_slots, device=initial.device)
    feet = torch.where(slots[None] < count[:, None], feet, 0)
    return self._route_vector(count, feet)

  def _episode_sample(self, initial: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    route = self._sample_route(initial)
    mean, log_variance = self._split_parameters(
      self.prior(torch.cat((initial, route), dim=-1))
    )
    latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
    self._last_latent_std = torch.exp(0.5 * log_variance).mean().detach()
    return route, latent

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state=None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    del masks, hidden_state, stochastic_output
    live = self._live(obs)
    initial = self._initial(obs)
    shape = (live.shape[0], self.latent_dim)
    if self._latent is None or self._latent.shape != shape:
      self._route, self._latent = self._episode_sample(initial)
    else:
      pending = torch.isnan(self._latent[:, 0])
      if pending.any():
        route, latent = self._episode_sample(initial[pending])
        assert self._route is not None
        self._route[pending] = route
        self._latent[pending] = latent
    assert self._route is not None and self._latent is not None
    hidden = self.decoder(torch.cat((live, self._route, self._latent), dim=-1))
    return self.action_head(hidden)

  def reconstruct(
    self, obs: TensorDict
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode teacher-route actions and return KL, route, foot-path losses."""
    live = self._live(obs)
    initial = self._initial(obs)
    path = self._path(obs)
    count, feet = self._labels(obs)
    route = self._route_vector(count, feet)
    prior_mean, prior_logvar = self._split_parameters(
      self.prior(torch.cat((initial, route), dim=-1))
    )
    posterior_mean, posterior_logvar = self._split_parameters(
      self.posterior(torch.cat((initial, route, path), dim=-1))
    )
    latent = posterior_mean + torch.exp(0.5 * posterior_logvar) * torch.randn_like(
      posterior_mean
    )
    hidden = self.decoder(torch.cat((live, route, latent), dim=-1))
    action = self.action_head(hidden)
    predicted_path = self.path_head(hidden)
    path_loss = torch.nn.functional.smooth_l1_loss(predicted_path, path)
    kl = (
      0.5
      * (
        prior_logvar
        - posterior_logvar
        + torch.exp(posterior_logvar - prior_logvar)
        + (posterior_mean - prior_mean).square() / torch.exp(prior_logvar)
        - 1.0
      )
      .sum(dim=-1)
      .mean()
    )

    count_logits, foot_logits = self._route_logits(initial)
    count_loss = torch.nn.functional.cross_entropy(
      count_logits, count.clamp(0, self.route_slots)
    )
    slots = torch.arange(self.route_slots, device=feet.device)
    foot_mask = slots[None] < count[:, None]
    foot_error = torch.nn.functional.cross_entropy(
      foot_logits.flatten(0, 1),
      (feet.clamp(1, 2) - 1).flatten(),
      reduction="none",
    ).reshape_as(feet)
    foot_loss = (foot_error * foot_mask).sum() / foot_mask.sum().clamp(min=1)
    return action, kl, count_loss + foot_loss, path_loss

  def update_normalization(self, obs: TensorDict) -> None:
    if not self.obs_normalization:
      return
    cast(EmpiricalNormalization, self.live_normalizer).update(
      cast(torch.Tensor, obs[self.live_name])
    )
    cast(EmpiricalNormalization, self.initial_normalizer).update(
      cast(torch.Tensor, obs[self.initial_name])
    )
    cast(EmpiricalNormalization, self.path_normalizer).update(
      cast(torch.Tensor, obs[self.path_name])
    )

  def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
    del hidden_state
    if dones is None:
      self._route = None
      self._latent = None
    elif self._latent is not None:
      done = dones.bool().view(-1)
      self._latent[done] = float("nan")

  def get_hidden_state(self):
    return None

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    del dones

  @property
  def output_std(self) -> torch.Tensor:
    return self._last_latent_std

  def as_jit(self) -> nn.Module:
    return _InferenceModel(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    del verbose
    return _InferenceModel(self)


class _InferenceModel(nn.Module):
  """Explicit route and fixed noise inputs keep deployment sampling outside the graph."""

  is_recurrent = False

  def __init__(self, model: GoalCvaeModel) -> None:
    super().__init__()
    self.live_normalizer = copy.deepcopy(model.live_normalizer)
    self.initial_normalizer = copy.deepcopy(model.initial_normalizer)
    self.route_prior = copy.deepcopy(model.route_prior)
    self.prior = copy.deepcopy(model.prior)
    self.decoder = copy.deepcopy(model.decoder)
    self.action_head = copy.deepcopy(model.action_head)
    self.live_dim = model.live_dim
    self.initial_dim = model.initial_dim
    self.route_dim = model.route_dim
    self.latent_dim = model.latent_dim

  def forward(
    self,
    live: torch.Tensor,
    initial: torch.Tensor,
    route: torch.Tensor,
    epsilon: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    live = self.live_normalizer(live)
    initial = self.initial_normalizer(initial)
    mean, log_variance = self.prior(torch.cat((initial, route), dim=-1)).chunk(
      2, dim=-1
    )
    latent = mean + torch.exp(0.5 * log_variance.clamp(-10.0, 4.0)) * epsilon
    hidden = self.decoder(torch.cat((live, route, latent), dim=-1))
    action = self.action_head(hidden)
    return action, self.route_prior(initial)

  def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
    return (
      torch.zeros(1, self.live_dim),
      torch.zeros(1, self.initial_dim),
      torch.zeros(1, self.route_dim),
      torch.zeros(1, self.latent_dim),
    )

  @property
  def input_names(self) -> list[str]:
    return ["live", "initial", "route", "epsilon"]

  @property
  def output_names(self) -> list[str]:
    return ["actions", "route_logits"]
