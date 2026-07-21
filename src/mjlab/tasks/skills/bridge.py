"""Bridges: the short-lived policies engaged around a switch.

A bridge takes control at the moment a switch fires, from wherever the previous skill
left the robot, and drives it into a state the next skill can start from. It also
decides, itself, when that state has been reached. This decision is the bridge's
own judgment call (a learned scorer, a distribution-matching critic, a reused value
function, ...), not something a skill reports.

The interface says nothing about how many policies sit behind it. One policy for
every transition, one per target skill as in Lee et al. (2019), or one per ordered pair of
skills as in Byun and Perrault (2022), are all the same interface: every call is told,
per env, which skill control is coming from and which one it is going to.
"""

from abc import ABC, abstractmethod

import torch

from mjlab.envs import VecEnvObs
from mjlab.tasks.skills.skill import NO_SKILL  # noqa: F401  (re-exported for callers)


class Bridge(ABC):
  """A source of actions between two skills.

  A bridge shares the action space of the skills it connects, so the composed system
  can hand control back and forth without translating anything.
  """

  @abstractmethod
  def act(
    self,
    obs: VecEnvObs,
    source: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Actions and a handover signal for every env.

    source and target hold, per env, the skill control came from and the skill it is
    headed to; source is NO_SKILL when nothing was running. active marks the envs
    this bridge is driving, under the same rule as Skill.act.

    Returns actions shaped (num_envs, action_dim) and a boolean mask (num_envs,) that
    is set where the bridge declares the target skill may take over now. Choosing
    that moment belongs to the bridge because every method in the literature puts it
    there: a termination head inside the policy in Lee et al. (2019), a separate
    Q network in Byun and Perrault 2022, a switch output in Tidd et al. 2022 ...
    """

  def begin(  # noqa: B027
    self, mask: torch.Tensor, source: torch.Tensor, target: torch.Tensor
  ) -> None:
    """Start a transition in the envs where mask is set.

    Called once when a switch fires, before the first act of that transition. The
    default does nothing, which is right for a bridge that keeps no state across
    steps.
    """
