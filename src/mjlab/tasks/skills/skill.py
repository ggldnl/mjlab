"""Skills: the frozen, independently trained policies that a bridge composes.

A skill is trained on its own, knows nothing about the other skills, and is never
fine-tuned to make a transition work. Preserving that independence is the point of
bridging, so nothing here may modify a skill. A skill also makes no claim about its
own progress: it is not required to say whether it succeeded, failed, or is done,
since that would force every skill (analytic controllers included) to carry extra
machinery it may not have. Whatever decides to switch away from a skill (the
controller) or judges whether a switch landed safely (the bridge) has to work from
what it can observe of the world, not from a self-report.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from dataclasses import asdict
from pathlib import Path

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, VecEnvObs
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

# Skill id standing for no skill, used wherever an id may be absent.
NO_SKILL = -1


class Skill(ABC):
  """One independently trained behavior."""

  name: str

  @abstractmethod
  def act(self, obs: VecEnvObs, active: torch.Tensor) -> torch.Tensor:
    """Actions for every env, shaped (num_envs, action_dim).

    active marks the envs this skill is driving. Actions returned for the other envs
    are discarded, so a stateless policy may ignore the mask, while a skill that
    carries internal state must advance it only where active is set.
    """

  def reset(self, mask: torch.Tensor) -> None:  # noqa: B027
    """Clear whatever internal state this skill keeps, where mask is set.

    The default does nothing, which is right for a stateless policy.
    """


class PolicySkill(Skill):
  """A frozen policy loaded from a trained mjlab task's checkpoint."""

  def __init__(
    self,
    name: str,
    task_id: str,
    checkpoint_path: str | Path,
    env: ManagerBasedRlEnv,
    device: str,
  ) -> None:
    """Load `task_id`'s policy from `checkpoint_path` against `env`.

    `env` must already be built from that task's own env cfg (matching
    observation/action spaces), and is shared with whatever else is using it --
    wrapping it here resets it once, as `RslRlVecEnvWrapper` documents ("rsl_rl
    does not call reset"), which is harmless since arch_0's own collection and
    training loops always reset explicitly before doing anything with a skill.
    """
    self.name = name
    agent_cfg = load_rl_cfg(task_id)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = runner_cls(wrapped_env, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    self._policy = runner.get_inference_policy(device=device)

  def act(self, obs: VecEnvObs, active: torch.Tensor) -> torch.Tensor:
    del active
    state = obs["actor"]
    assert isinstance(state, torch.Tensor)
    obs_td = TensorDict(obs, batch_size=[state.shape[0]])
    return self._policy(obs_td)


class SkillPool:
  """The ordered set of skills a controller may choose from.

  Position in the pool fixes a skill's integer id. Those ids are what a controller
  emits and what a bridge is conditioned on, so the order is part of the experiment.
  """

  def __init__(self, skills: Sequence[Skill]) -> None:
    if not skills:
      raise ValueError("A skill pool needs at least one skill")
    self.skills = tuple(skills)
    self.ids = {skill.name: i for i, skill in enumerate(self.skills)}

  def __len__(self) -> int:
    return len(self.skills)

  def __getitem__(self, skill_id: int) -> Skill:
    return self.skills[skill_id]

  def act(self, obs: VecEnvObs, assignment: torch.Tensor) -> torch.Tensor:
    """Actions for every env, taken from the skill that env is assigned to.

    assignment holds one skill id per env; envs set to NO_SKILL get zeros. Every
    skill is evaluated on the whole batch and its rows are then selected, which keeps
    the call batched at the cost of one forward pass per skill.
    """
    actions = [skill.act(obs, assignment == i) for i, skill in enumerate(self.skills)]
    out = actions[0]
    for skill_id, action in enumerate(actions[1:], start=1):
      out = torch.where((assignment == skill_id).unsqueeze(-1), action, out)
    return torch.where((assignment >= 0).unsqueeze(-1), out, torch.zeros_like(out))

  def reset(self, mask: torch.Tensor) -> None:
    """Reset every skill's internal state where mask is set."""
    for skill in self.skills:
      skill.reset(mask)
