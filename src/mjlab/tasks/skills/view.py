"""What part of a skill's observation the bridging machinery is allowed to see.

A bridge is judged by how close it gets to the target skill's own behavior, and that
comparison is only as meaningful as the vector it is made on. The env's observation is
built for the skill: it carries everything that skill needs to do its job, including
terms that say where the robot is in the world or what goal it happens to have been
given. Those are exactly the terms a bridge must not be compared on.

The diffdrive is the clean example. Its observation ends with the command term, whose
second channel is the live error to a heading target fixed when the episode began. The
`drive` skill's own recordings all start from a reset, so that error reads ~0 in every
one of them. The bridge into `drive`, though, is dropped in wherever `turn` left the
robot -- a quarter turn away, so the error reads ~pi/2. A discriminator comparing the
two on the full observation separates them on that one channel alone and never has to
look at the motion, and the only way for the bridge to close the gap is to steer the
robot back to the heading the episode started at. So a bridge whose whole job was "shed
the speed and hand over" gets trained to navigate instead, and the hand-over it produces
is worse than the naive one. The same argument applies to any absolute position, any
world-frame heading, and any per-episode goal: `x`, `y` and `theta` are the textbook
case.

Nothing about that is diffdrive-specific, and nothing about it can be decided here: only
the experiment knows which of its observation terms describe *the robot* and which
describe *the task*. So a view is declared next to the skill pool, as a `StateViewCfg`
naming terms to keep or drop, and resolved against the env into a `StateView` that every
piece of the bridging machinery then shares -- the recorded windows, the discriminator,
the bridge actor and the switch-decider. Sharing one projection is the point: an actor
that saw a channel its discriminator did not would be rewarded for something it could
not perceive, and vice versa.

The skills themselves are untouched. They keep acting on the full observation; only the
bridge's view of the world is narrowed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv


class StateView(ABC):
  """A fixed projection of one observation group's vector.

  Resolved once against an env and then applied everywhere, so `dim` is a plain
  attribute: whatever builds a network can size it without a sample observation.
  """

  #: Short description of what survives the projection, for training logs.
  label: str

  #: Width of the projected vector.
  dim: int

  @abstractmethod
  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    """Project observations shaped (..., source_dim) to (..., dim)."""


class FullState(StateView):
  """The whole observation, unchanged. The right default when nothing is known."""

  def __init__(self, dim: int) -> None:
    self.dim = dim
    self.label = f"full observation ({dim} dims)"

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    return obs


class SlicedState(StateView):
  """A fixed subset of the observation's channels, in their original order."""

  def __init__(self, indices: Sequence[int], source_dim: int, label: str) -> None:
    if not indices:
      raise ValueError("A state view cannot be empty; it would leave nothing to match.")
    self._indices = torch.tensor(sorted(indices), dtype=torch.long)
    self.dim = int(self._indices.numel())
    self.source_dim = source_dim
    self.label = label

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    if obs.shape[-1] != self.source_dim:
      raise ValueError(
        f"This view was resolved against a {self.source_dim}-dim observation but got "
        f"{obs.shape[-1]}. It is tied to one observation group of one env cfg."
      )
    return obs.index_select(-1, self._indices.to(obs.device))


@dataclass(frozen=True)
class StateViewCfg:
  """Which observation terms the bridging machinery works on, named rather than sliced.

  Declared by an experiment beside its skill pool, so the choice is written where the
  observation layout is understood and survives a change to it: terms are resolved by
  name against the env, so inserting an observation term shifts no index here.

  Pass `drop` to name the terms to leave out (the usual case: a handful of task or
  world-frame terms out of an otherwise proprioceptive observation), or `keep` to name
  the ones to retain. Exactly one of the two, since giving both is a good way to write a
  view that means something other than what was intended.
  """

  drop: tuple[str, ...] = ()
  """Observation terms the bridge must not see, by name."""

  keep: tuple[str, ...] | None = None
  """Observation terms the bridge sees, by name. Everything else is dropped."""

  group: str = "actor"
  """The observation group the terms are looked up in."""

  def __post_init__(self) -> None:
    if self.keep is not None and self.drop:
      raise ValueError("Give a state view either `keep` or `drop`, not both.")

  def resolve(self, env: ManagerBasedRlEnv) -> StateView:
    """Turn the named terms into a projection of this env's observation."""
    manager = env.observation_manager
    names = manager.active_terms.get(self.group)
    if names is None:
      raise KeyError(
        f"No observation group '{self.group}'; this env has "
        f"{sorted(manager.active_terms)}."
      )
    dims = manager.group_obs_term_dim[self.group]

    spans: dict[str, tuple[int, int]] = {}
    offset = 0
    for name, shape in zip(names, dims, strict=True):
      width = 1
      for extent in shape:
        width *= int(extent)
      spans[name] = (offset, offset + width)
      offset += width

    requested = set(self.drop) | set(self.keep or ())
    unknown = requested - set(spans)
    if unknown:
      raise KeyError(
        f"No observation term(s) {sorted(unknown)} in group '{self.group}'; it has "
        f"{list(spans)}."
      )

    if self.keep is None and not self.drop:
      return FullState(offset)

    if self.keep is not None:
      kept = [n for n in names if n in self.keep]
    else:
      kept = [n for n in names if n not in self.drop]
    indices = [i for name in kept for i in range(*spans[name])]
    if not indices:
      raise ValueError(
        f"This view keeps no observation term at all (group '{self.group}' has "
        f"{list(spans)}). A bridge with nothing to see cannot be trained."
      )
    dropped = [n for n in names if n not in kept]
    return SlicedState(
      indices,
      offset,
      label=f"{len(indices)}/{offset} dims, dropping {dropped or ['nothing']}",
    )


def resolve_view(
  env: ManagerBasedRlEnv, cfg: StateViewCfg | None, group: str = "actor"
) -> StateView:
  """The view `cfg` describes, or the full observation when an experiment declares none."""
  if cfg is not None:
    return cfg.resolve(env)
  obs = env.observation_manager.compute()[group]
  assert isinstance(obs, torch.Tensor)
  return FullState(int(obs.shape[-1]))
