"""The three frozen policies, wired to act inside the shared arena.

`PolicySkill` assumes a skill reads the observation group its own task called `actor`.
That holds when every skill in a pool was trained in the same environment, which is
true of the cart-pole and the diffdrive and false here: walk and run read a commanded
velocity, jump reads a reference clip, and the arena carries both under separate names
(see arena.py). So each skill is told which group it reads, and the rest is
`PolicySkill` unchanged.

Nothing about the individual tasks is touched by this. Their own environments still
call the group `actor`, they still train and export normally, and checkpoints trained
before the arena existed still load: the group name is configuration, the observation
layout underneath it is what the network was fitted to, and that is identical.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, VecEnvObs
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.tasks.skills.skill import Skill

if TYPE_CHECKING:
  from mjlab.tasks.skills.experiments.parkour.jump.mdp.commands import JumpCommand


class ArenaSkill(Skill):
  """A frozen policy that reads a named observation group of the shared arena."""

  def __init__(
    self,
    name: str,
    task_id: str,
    checkpoint_path: str | Path,
    env: ManagerBasedRlEnv,
    device: str,
    obs_group: str,
    critic_group: str | None = None,
  ) -> None:
    self.name = name
    self.obs_group = obs_group

    agent_cfg = load_rl_cfg(task_id)
    # The one change: point both roles at this arena's groups instead of the task's
    # own. `load_rl_cfg` hands back a deep copy, so the registry is not disturbed.
    agent_cfg.obs_groups = {
      "actor": (obs_group,),
      "critic": (critic_group or obs_group,),
    }

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = runner_cls(wrapped_env, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    self._policy = runner.get_inference_policy(device=device)

  @torch.no_grad()
  def act(self, obs: VecEnvObs, active: torch.Tensor) -> torch.Tensor:
    del active
    state = obs[self.obs_group]
    assert isinstance(state, torch.Tensor)
    return self._policy(TensorDict(obs, batch_size=[state.shape[0]]))


class JumpSkill(ArenaSkill):
  """The jump, with its reference pinned to the robot when it takes over.

  The jump is a motion-tracking policy: it follows a clip anchored somewhere in the
  world. In its own environment that anchor is the origin and the reset teleports the
  robot onto the clip's first frame. In the corridor there is no reset to hide behind
  -- the robot is wherever walking left it -- so control passing to this skill has to
  move the clip to the robot instead.

  `Skill.reset` is exactly the hook for that: the composition calls it on the
  environments where this skill is starting, and the reference is re-anchored and
  rewound. It is also why the jump is one of the skills a bridge matters for. The
  reference always starts from its own first frame, which is a stand; a robot arriving
  at speed is being handed a clip that assumes it is not.
  """

  def __init__(self, *args, command_name: str = "motion", **kwargs) -> None:
    self._env: ManagerBasedRlEnv = kwargs.get("env") or args[3]
    self._command_name = command_name
    self.entry_frame = 0
    """Which frame of the clip control arrives at. Zero is the clip's own opening, a
    stand. An architecture that can deliver the robot somewhere better sets this before
    handing over; arch_0 through arch_3 cannot, and leave it alone."""
    super().__init__(*args, **kwargs)

  @property
  def command(self) -> JumpCommand:
    from mjlab.tasks.skills.experiments.parkour.jump.mdp.commands import JumpCommand

    term = self._env.command_manager.get_term(self._command_name)
    assert isinstance(term, JumpCommand)
    return term

  def reset(self, mask: torch.Tensor) -> None:
    env_ids = mask.nonzero(as_tuple=False).squeeze(-1)
    if env_ids.numel() > 0:
      self.command.anchor_to_robot(env_ids, start_frame=self.entry_frame)
