"""
Experts: the frozen, independently trained policies that a bridge composes.

An expert is trained on its own, knows nothing about the other experts, and is never
fine-tuned to make a transition work. Preserving that independence is the point of
bridging, so nothing here may modify an expert.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import IntEnum

import torch

from mjlab.envs import ManagerBasedRlEnv, VecEnvObs

# Expert id standing for no expert, used wherever an id may be absent
NO_EXPERT = -1


class ExpertStatus(IntEnum):
  """What an expert reports about its own progress.

  This is the signal a controller switches on and the label that marks a bridge
  attempt as having worked or not.
  """

  RUNNING = 0
  SUCCEEDED = 1
  FAILED = 2


class Expert(ABC):
  """One independently trained behaviour."""

  name: str

  @abstractmethod
  def act(self, obs: VecEnvObs, active: torch.Tensor) -> torch.Tensor:
    """Actions for every env, shaped (num_envs, action_dim).

    active marks the envs this expert is driving. Actions returned for the other envs
    are discarded, so a stateless policy may ignore the mask, while an expert that
    carries internal state must advance it only where active is set.
    """

  @abstractmethod
  def status(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Progress in every env as ExpertStatus values, shaped (num_envs,).

    This is a judgement about the task rather than an output of the policy, so it
    reads the env rather than the observations.
    """

  def reset(self, mask: torch.Tensor) -> None:  # noqa: B027
    """Clear whatever internal state this expert keeps, where mask is set.

    The default does nothing, which is right for a stateless policy.
    """


class ExpertPool:
  """The ordered set of experts a controller may choose from.

  Position in the pool fixes an expert's integer id. Those ids are what a controller
  emits and what a bridge is conditioned on, so the order is part of the experiment.
  """

  def __init__(self, experts: Sequence[Expert]) -> None:
    if not experts:
      raise ValueError("An expert pool needs at least one expert")
    self.experts = tuple(experts)
    self.ids = {expert.name: i for i, expert in enumerate(self.experts)}

  def __len__(self) -> int:
    return len(self.experts)

  def __getitem__(self, expert_id: int) -> Expert:
    return self.experts[expert_id]

  def act(self, obs: VecEnvObs, assignment: torch.Tensor) -> torch.Tensor:
    """Actions for every env, taken from the expert that env is assigned to.

    assignment holds one expert id per env; envs set to NO_EXPERT get zeros. Every
    expert is evaluated on the whole batch and its rows are then selected, which keeps
    the call batched at the cost of one forward pass per expert.
    """
    actions = [
      expert.act(obs, assignment == i) for i, expert in enumerate(self.experts)
    ]
    out = actions[0]
    for expert_id, action in enumerate(actions[1:], start=1):
      out = torch.where((assignment == expert_id).unsqueeze(-1), action, out)
    return torch.where((assignment >= 0).unsqueeze(-1), out, torch.zeros_like(out))

  def status(self, env: ManagerBasedRlEnv, assignment: torch.Tensor) -> torch.Tensor:
    """Status of the expert each env is assigned to, shaped (num_envs,).

    Envs set to NO_EXPERT report RUNNING, since no expert is making a claim there.
    """
    statuses = [expert.status(env) for expert in self.experts]
    out = statuses[0]
    for expert_id, status in enumerate(statuses[1:], start=1):
      out = torch.where(assignment == expert_id, status, out)
    return torch.where(assignment >= 0, out, ExpertStatus.RUNNING)
