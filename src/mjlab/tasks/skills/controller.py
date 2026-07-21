"""
Controllers: the high level decision of which expert should be running.

The controller is assumed given. This study is about executing a switch well, not
about choosing it well, so a controller may be anything that reads the situation and
names an expert: a finite state machine, a planner, a rule read off the world, or a
learned policy. It is the only part of the decision that is experiment specific.

It sees the env and the pool of experts and emits one expert id per env. A switch
fires wherever that id differs from the one control is already committed to. The
controller does not run the bridge and does not know a bridge exists.
"""

from abc import ABC, abstractmethod

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.expert import ExpertPool


class Controller(ABC):
  """Names the expert that should be running in each env."""

  def __init__(self, pool: ExpertPool) -> None:
    self.pool = pool

  @abstractmethod
  def decide(self, env: ManagerBasedRlEnv, target: torch.Tensor) -> torch.Tensor:
    """The expert that should be running in every env, shaped (num_envs,) int64.

    target holds the expert control is currently committed to, which is the running
    expert in normal operation and the expert being bridged to during a transition.
    Returning it unchanged means carry on. Returning a different id is the switch
    signal, whether it interrupts an expert or retargets a transition already underway.
    """

  def reset(self, mask: torch.Tensor) -> None:  # noqa: B027
    """Clear whatever internal state this controller keeps, where mask is set.

    The default does nothing, which is right for a controller that decides from the
    env alone.
    """
