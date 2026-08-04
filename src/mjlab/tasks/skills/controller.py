"""Controllers: the high level decision of which skill should be running.

The controller is assumed given. This study is about executing a switch well, not about
choosing it well, so a controller may be anything that reads the situation and names a
skill: a state machine, a planner, a rule, a learned policy.

It emits one skill id per env. A switch fires wherever that id differs from the one
control is already committed to. It does not run the transition and does not know one
exists, and it cannot lean on a skill self-reporting progress: whatever it switches on
has to come from observing the env.
"""

from abc import ABC, abstractmethod

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.skill import SkillPool


class Controller(ABC):
  """Names the skill that should be running in each env."""

  def __init__(self, pool: SkillPool) -> None:
    self.pool = pool

  @abstractmethod
  def decide(self, env: ManagerBasedRlEnv, target: torch.Tensor) -> torch.Tensor:
    """The skill that should be running in every env, shaped (num_envs,) int64.

    `target` holds the skill control is currently committed to. Returning it unchanged
    means carry on; returning a different id is the switch signal, whether it interrupts
    a skill or retargets a transition already underway.
    """

  def reset(self, mask: torch.Tensor) -> None:  # noqa: B027
    """Clear this controller's state where `mask` is set. Stateless ones need nothing."""
