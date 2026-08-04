"""Skills: the frozen, independently trained policies that an architecture composes.

A skill is trained on its own, knows nothing about the other skills, and is never
fine-tuned to make a transition work. Preserving that independence is the point of
bridging, so nothing here may modify a skill.

A skill makes no claim about its own progress either: it never reports success, failure
or completion, since that would force every skill (analytic controllers included) to
carry machinery it may not have. Whatever decides to switch away (the controller) or
judges whether a switch landed (the architecture) works from what it can observe.

A skill may or may not have memory. diffdrive's `TurnSkill` for example integrates
a yaw rate to know how far it has turned.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, VecEnvObs
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_runner_cls

# Skill id standing for no skill, used wherever an id may be absent.
NO_SKILL = -1


class Skill(ABC):
  """One independently trained behavior."""

  name: str

  @abstractmethod
  def act(self, obs: VecEnvObs, active: torch.Tensor) -> torch.Tensor:
    """Actions for every env, shaped (num_envs, action_dim).

    `active` marks the envs this skill is driving. Actions returned for the others are
    discarded, so a stateless policy may ignore the mask; a skill with memory must
    advance it only where `active` is set, and may read the mask's rising edge to learn
    that control has arrived.
    """

  def reset(self, mask: torch.Tensor) -> None:  # noqa: B027
    """Clear this skill's memory where `mask` is set. Stateless skills need nothing."""


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
    observation/action spaces). Wrapping it here resets it once, as `RslRlVecEnvWrapper`
    documents; harmless, since every caller resets before using a skill.
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

  @torch.no_grad()
  def act(self, obs: VecEnvObs, active: torch.Tensor) -> torch.Tensor:
    del active  # Stateless
    state = obs["actor"]
    assert isinstance(state, torch.Tensor)
    obs_td = TensorDict(obs, batch_size=[state.shape[0]])
    return self._policy(obs_td)


class SkillPool:
  """The ordered set of skills a controller may choose from.

  Position in the pool fixes a skill's integer id. Those ids are what a controller emits
  and what an architecture is conditioned on, so the order is part of the experiment.
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

  ##
  # The one tick, and the two pure helpers around it.
  ##

  def act_each(self, obs: VecEnvObs, involved: torch.Tensor) -> torch.Tensor:
    """Tick every skill once. Returns (num_skills, num_envs, action_dim).

    `involved` is (num_skills, num_envs): the envs each skill is driving this step. It
    must already name every skill whose action will be used, whatever it is used for,
    because this is the only pass over the pool in a step (see the module docstring).
    Rows for envs a skill is not driving are produced anyway and discarded by the caller.
    """
    return torch.stack(
      [skill.act(obs, involved[i]) for i, skill in enumerate(self.skills)]
    )

  def involvement(self, assignment: torch.Tensor) -> torch.Tensor:
    """The plain mask for an assignment: each skill drives the envs assigned to it."""
    return torch.stack([assignment == i for i in range(len(self.skills))])

  def driven_by(self, skill_id: int, num_envs: int, device: str) -> torch.Tensor:
    """The mask in which `skill_id` drives every env and the rest drive none."""
    mask = torch.zeros(len(self.skills), num_envs, dtype=torch.bool, device=device)
    if skill_id != NO_SKILL:
      mask[skill_id] = True
    return mask

  @staticmethod
  def select(skill_actions: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
    """Each env's row of `act_each`'s output; envs set to NO_SKILL get zeros."""
    width = skill_actions.shape[-1]
    index = assignment.clamp_min(0).view(1, -1, 1).expand(1, -1, width)
    chosen = skill_actions.gather(0, index).squeeze(0)
    return torch.where(
      (assignment >= 0).unsqueeze(-1), chosen, torch.zeros_like(chosen)
    )

  def act(self, obs: VecEnvObs, assignment: torch.Tensor) -> torch.Tensor:
    """Tick the pool and take each env's action from the skill it is assigned to.

    The convenience form of the three calls above, for a caller with nothing else to do
    with the actions. `assignment` holds one skill id per env; NO_SKILL gets zeros.
    """
    return self.select(self.act_each(obs, self.involvement(assignment)), assignment)

  def reset(self, mask: torch.Tensor) -> None:
    """Reset every skill's memory where `mask` is set."""
    for skill in self.skills:
      skill.reset(mask)


if __name__ == "__main__":
  """Watch one skill on its own, with no controller and no architecture.

  The env is built straight from a registered task id and left running the same skill
  forever, so this answers "does this skill work at all" before anything is composed.

      uv run python -m mjlab.tasks.skills.skill --task-id <id> --checkpoint <path>
      uv run python -m mjlab.tasks.skills.skill --task-id <id> --factory module:attr
  """

  from dataclasses import dataclass

  import torch
  import tyro

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.tasks.skills.skill import PolicySkill, Skill

  @dataclass(frozen=True)
  class ViewSkillConfig:
    task_id: str
    """Registered mjlab task id; its (play-mode) env cfg is what the skill acts in."""

    checkpoint: str | None = None
    """Path to a trained checkpoint, loaded as a `PolicySkill`."""

    factory: str | None = None
    """'module:attribute' resolving to a zero-arg callable returning a `Skill`, for
    watching a hand-written analytical skill instead of a checkpoint."""

    name: str = "skill"
    num_envs: int = 1
    device: str | None = None

  def view_skill(cfg: ViewSkillConfig) -> None:
    if (cfg.checkpoint is None) == (cfg.factory is None):
      raise ValueError("Pass exactly one of --checkpoint or --factory.")

    import mjlab.tasks  # noqa: F401  (populates the task registry)
    from mjlab.utils.lab_api.string import string_to_callable
    from mjlab.viewer import ViserPlayViewer

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(cfg.task_id, play=True)
    env_cfg.scene.num_envs = cfg.num_envs
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    skill: Skill
    if cfg.checkpoint is not None:
      skill = PolicySkill(cfg.name, cfg.task_id, cfg.checkpoint, env, device)
    else:
      assert cfg.factory is not None
      built = string_to_callable(cfg.factory)()
      if not isinstance(built, Skill):
        raise TypeError(
          f"'{cfg.factory}' must return a Skill, got {type(built).__name__}"
        )
      skill = built

    viewer_env = RslRlVecEnvWrapper(
      env, clip_actions=load_rl_cfg(cfg.task_id).clip_actions
    )
    active = torch.ones(cfg.num_envs, dtype=torch.bool, device=device)

    def policy(obs) -> torch.Tensor:
      return skill.act(obs, active)

    ViserPlayViewer(viewer_env, policy).run()
    viewer_env.close()

  view_skill(tyro.cli(ViewSkillConfig))
