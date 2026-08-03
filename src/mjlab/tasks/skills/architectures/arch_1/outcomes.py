"""Where a switch-decider's "did that hand-over go well?" signal comes from.

This belongs to the bridge family, not to the shared skills package. Learning to *move*
like the target skill needs no such signal at all: that is a copying game between the
bridge and a discriminator. Only learning *when to let go* needs one, and only an
architecture that has a switch-decider needs that. An architecture that hands over on a
fixed schedule, or has no hand-over decision to make, imports none of this.

`OracleOutcome` is arch_1's own answer: a function the experiment author wrote, saying
whether the robot is in a state the target skill can take over from. It is the most
controllable option and the most fragile one, since the test has to encode what "safe"
means and there is no sign when it is wrong. arch_2 keeps the same contract
and replace the judgement with something the environment already reports;
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.skill import Skill

# Given the env, returns a bool per env saying whether the robot is in a state the
# target skill can safely be in. Privileged and external, never read off the skill
# itself (see Skill). Called every step and read at each env's own evaluation moment, so
# it must be a pure function of the current state with no memory of its own.
SuccessFn = Callable[[ManagerBasedRlEnv], torch.Tensor]


class OutcomeSource(ABC):
  """Judges each env's hand-over, once the target skill has had time to show."""

  #: Shown in the training header so a log says which signal produced it.
  label: str = "outcome"

  def prepare(  # noqa: B027
    self, env: ManagerBasedRlEnv, target_skill: Skill, eval_steps: int
  ) -> None:
    """Measure whatever this source needs before training starts.

    Free to roll the env around; training resets it at the top of every window.
    """

  @abstractmethod
  def begin(self, num_envs: int, device: str) -> None:
    """Start a fresh hand-over window."""

  def record(self, reward: torch.Tensor, counting: torch.Tensor) -> None:  # noqa: B027
    """Take in one step of the window.

    `counting` marks the envs whose hand-over has already happened, so a source that
    accumulates something only accumulates it while the target skill is driving.
    """

  @abstractmethod
  def verdict(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Whether each env's hand-over came out well, read at its evaluation moment."""

  def summary(self) -> str:
    """A short fragment for the training log, or empty."""
    return ""


class OracleOutcome(OutcomeSource):
  """A hand-written test the experiment author supplies.

  Note what it is not asked: the caller already requires the episode to have survived,
  because a terminated env auto-resets and anything read from it afterwards describes a
  fresh episode rather than the hand-over. The oracle only has to say whether the state
  it is shown is one the target skill can work from.
  """

  label = "oracle"

  def __init__(self, success_fn: SuccessFn) -> None:
    self.success_fn = success_fn

  def begin(self, num_envs: int, device: str) -> None:
    del num_envs, device

  def verdict(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    return self.success_fn(env)
