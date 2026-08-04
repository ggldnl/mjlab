"""The actor an architecture trains, and control of its exploration.

The actor is a plain rsl_rl `MLPModel`; `build_actor` is the single place it is
constructed, so the meta policy that holds it for inference and the trainer that fits it
cannot drift on how it was built.

Everything arriving here is already in the run's own units: observations standardized
and actions normalized by spaces.py. Nothing in this file converts anything.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from tensordict import TensorDict

# The obs groups an actor and its critic read. Shared by `build_actor` and by every
# trainer's critic so they stay in lock-step.
OBS_GROUPS = {"actor": ["actor"], "critic": ["critic"]}


def obs_td(state: torch.Tensor) -> TensorDict:
  """Pack an already-standardized state into the TensorDict the nets read.

  Both entries hold the same vector. There is no privileged critic input on purpose: the
  actor is taught to match a distribution the referee judges on this exact vector, and a
  critic seeing more than the referee would be valuing something the reward cannot
  express.
  """
  return TensorDict({"actor": state, "critic": state}, batch_size=[int(state.shape[0])])


def mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
  """A plain ReLU stack, the shape every hand-rolled network here uses."""
  dims = (input_dim, *hidden_dims, output_dim)
  layers: list[nn.Module] = []
  for i in range(len(dims) - 1):
    layers.append(nn.Linear(dims[i], dims[i + 1]))
    if i < len(dims) - 2:
      layers.append(nn.ReLU())
  return nn.Sequential(*layers)


def build_actor(
  template: TensorDict,
  action_dim: int,
  hidden_dims: tuple[int, ...],
  device: str,
  init_std: float = 0.4,
) -> MLPModel:
  """Build the actor the one way this package ever builds it.

  `init_std` is in normalized action units (see spaces.py), where the range the pool's
  skills command spans roughly [-1, 1]. So it reads directly as "how much of the useful
  range does this policy explore": 0.4 is a fifth of it per step. This was 1.0 while the
  actor emitted raw wheel velocities spanning forty units, i.e. two percent exploration,
  which is why the bridge never found the braking behavior it was asked for.
  """
  return MLPModel(
    template,
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
  """The actor's learnable std, and whether it is held in log space."""
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
  """Set the actor's exploration, in normalized action units.

  `build_actor` picks a default before any training config has been seen; a trainer calls
  this so the number the experiment actually declared is the one that applies.
  """
  parameter, is_log = _std_parameter(actor)
  parameter.fill_(math.log(std) if is_log else std)


@torch.no_grad()
def clamp_action_std(actor: MLPModel, max_std: float) -> None:
  """Hold the actor's exploration below `max_std`, in normalized action units.

  rsl_rl keeps `log_std` free and nothing bounds it, so PPO's entropy bonus pushes it up
  without limit whenever the surrogate is weaker than the bonus. That is not hypothetical
  here: the referee cannot be moved much by a poor policy, so the surrogate is small from
  the start and `log_std` grows by about a percent an iteration forever. Over a few
  thousand iterations that overflows to inf, the next gradient is NaN, and PPO dies
  inside `Normal`.

  Clamping rather than lowering `entropy_coef` alone because the two do different jobs:
  the coefficient sets how hard the run is pushed to explore, this sets the point past
  which more exploration is not exploration. At 1.0 the actor already spans the pool's
  entire commanded range.
  """
  parameter, is_log = _std_parameter(actor)
  parameter.clamp_max_(math.log(max_std) if is_log else max_std)
