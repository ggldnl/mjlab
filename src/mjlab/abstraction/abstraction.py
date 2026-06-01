"""Base classes for abstractions.

An *abstraction* is a reduced-order ("template") model of a task. It encodes
domain knowledge - simple physics or kinematics - about what a behavior should
roughly look like, and turns that knowledge into:

1. A per-episode *reference* (e.g. a ballistic trajectory for a jump, or a
   velocity-integrated path for walking).
2. A set of scalar *signals* in ``[0, 1]`` that measure how well the full-body
   robot matches the reference. These are surfaced to the reward manager as
   ordinary reward terms (so weighting, logging and dt-scaling stay idiomatic),
   and the abstraction's reference can also be fed to the policy through the
   observation manager.

The point is to *guide* exploration with cheap analytical knowledge without
hand-authoring joint-level behavior: the RL algorithm still discovers how to
move, but it is nudged toward the family of solutions the abstraction
describes. Multiple abstractions can coexist, each contributing its own
signals.

This mirrors the established idea of reference-trajectory tracking / template
models (SLIP, centroidal dynamics, DeepMimic-style imitation), except the
reference is generated on the fly from physics rather than from motion capture.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class AbstractionCfg(abc.ABC):
  """Base configuration for an abstraction term."""

  debug_vis: bool = False
  """Whether to draw the abstraction's reference (target, trajectory) in the
  viewer."""

  @abc.abstractmethod
  def build(self, env: ManagerBasedRlEnv) -> Abstraction:
    """Build the abstraction term from this config."""
    raise NotImplementedError


class Abstraction(ManagerTermBase, abc.ABC):
  """Base class for abstraction terms.

  Subclasses define *what* the reference is (``_resample`` samples it per
  episode) and *how* to score it (``_update`` recomputes per-step signals and
  policy observations from the current simulation state).
  """

  def __init__(self, cfg: AbstractionCfg, env: ManagerBasedRlEnv):
    self.cfg = cfg
    super().__init__(env)
    # Cached per-step outputs, refreshed by ``_update``.
    self._signals: dict[str, torch.Tensor] = {}
    self._obs: dict[str, torch.Tensor] = {}
    self.metrics: dict[str, torch.Tensor] = {}
    self._debug_vis_enabled: bool = True

  # Accessors.

  @property
  def signals(self) -> dict[str, torch.Tensor]:
    """Named reward signals, each shape ``(num_envs,)``."""
    return self._signals

  def get_signal(self, name: str) -> torch.Tensor:
    return self._signals[name]

  def get_obs(self, name: str) -> torch.Tensor:
    return self._obs[name]

  # Lifecycle.

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    """Resample the reference for ``env_ids`` and return logging extras."""
    assert isinstance(env_ids, torch.Tensor)
    extras: dict[str, float] = {}
    for metric_name, metric_value in self.metrics.items():
      extras[metric_name] = torch.mean(metric_value[env_ids]).item()
      metric_value[env_ids] = 0.0
    self._resample(env_ids)
    return extras

  def compute(self, dt: float) -> None:
    """Refresh per-step signals and observations from the current state."""
    self._update(dt)

  # Visualization.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if self.cfg.debug_vis and self._debug_vis_enabled:
      self._debug_vis_impl(visualizer)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    pass

  # Subclass interface.

  @abc.abstractmethod
  def _resample(self, env_ids: torch.Tensor) -> None:
    """Sample a new per-episode reference for the given environments."""
    raise NotImplementedError

  @abc.abstractmethod
  def _update(self, dt: float) -> None:
    """Recompute ``self._signals`` and ``self._obs`` from the current state."""
    raise NotImplementedError
