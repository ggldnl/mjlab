"""Abstraction manager.

Owns a set of abstraction terms (template models). Each step it refreshes their
references and signals; each reset it resamples them. Signals are read back by
the reward manager through ``mdp`` helpers, and references by the observation
manager, so weighting and logging stay in the usual managers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import torch
from prettytable import PrettyTable

from mjlab.abstraction.abstraction import Abstraction, AbstractionCfg
from mjlab.managers.manager_base import ManagerBase

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class AbstractionManager(ManagerBase):
  """Manages abstraction terms for the environment."""

  _env: ManagerBasedRlEnv

  def __init__(self, cfg: dict[str, AbstractionCfg], env: ManagerBasedRlEnv):
    self._terms: dict[str, Abstraction] = dict()
    self.cfg = cfg
    super().__init__(env)

  def __str__(self) -> str:
    msg = f"<AbstractionManager> contains {len(self._terms)} active terms.\n"
    table = PrettyTable()
    table.title = "Active Abstraction Terms"
    table.field_names = ["Index", "Name", "Type", "Signals"]
    table.align["Name"] = "l"
    table.align["Signals"] = "l"
    for index, (name, term) in enumerate(self._terms.items()):
      signals = ", ".join(term.signals.keys())
      table.add_row([index, name, term.__class__.__name__, signals])
    msg += table.get_string()
    msg += "\n"
    return msg

  # Properties.

  @property
  def active_terms(self) -> list[str]:
    return list(self._terms.keys())

  # Methods.

  def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    extras = {}
    for name, term in self._terms.items():
      metrics = term.reset(env_ids=env_ids)
      for metric_name, metric_value in metrics.items():
        extras[f"Metrics/{name}/{metric_name}"] = metric_value
    return extras

  def compute(self, dt: float) -> None:
    for term in self._terms.values():
      term.compute(dt)

  def get_term(self, name: str) -> Abstraction:
    return self._terms[name]

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    for term in self._terms.values():
      term.debug_vis(visualizer)

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    terms = []
    for name, term in self._terms.items():
      for signal_name, signal in term.signals.items():
        terms.append((f"{name}/{signal_name}", [signal[env_idx].cpu().item()]))
    return terms

  def _prepare_terms(self) -> None:
    for term_name, term_cfg in self.cfg.items():
      if term_cfg is None:
        continue
      term = term_cfg.build(self._env)
      if not isinstance(term, Abstraction):
        raise TypeError(
          f"Returned object for the term {term_name} is not of type Abstraction."
        )
      self._terms[term_name] = term


class NullAbstractionManager:
  """Placeholder for an absent abstraction manager that no-ops all operations."""

  def __init__(self):
    self.active_terms: list[str] = []
    self._terms: dict[str, Any] = {}
    self.cfg = None

  def __str__(self) -> str:
    return "<NullAbstractionManager> (inactive)"

  def __repr__(self) -> str:
    return "NullAbstractionManager()"

  def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    return {}

  def compute(self, dt: float) -> None:
    pass

  def get_term(self, name: str) -> None:
    return None

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    pass

  def get_active_iterable_terms(
    self, env_idx: int
  ) -> Sequence[tuple[str, Sequence[float]]]:
    return []
