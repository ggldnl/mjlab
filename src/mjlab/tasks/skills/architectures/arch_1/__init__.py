"""Architecture 1: one distribution-matching bridge per target skill.

One bridge per target skill (source-agnostic), trained the way Byun and Perrault
(2022) train a transition policy. For each target skill, training (see train.py) is:

- Step 1: collect the target skill's initiation window. Let the skill we are bridging
    toward run on its own and record its first few moments (states and actions). This
    is the "example to imitate".
- Step 2: collect places the bridge might be dropped into. Run every other skill in
    the pool for a random amount of time and record where the robot ends up: a pool
    of realistic interrupted states to start bridge-training episodes from.
- Step 3: train the bridge to move like the target skill. Starting from an
    interrupted state, the bridge steers the robot to look like the step-1 recording.
    It never copies the recording directly; it plays a copy-catch game against a
    referee (the AIRL discriminator) that keeps getting better at telling the bridge
    apart from the real thing, while the bridge keeps adjusting to fool it (PPO on the
    discriminator's confidence as reward).
- Step 4: freeze the bridge, then train a separate switch-decider. A small Q-network
    that only ever answers "hand over now, or keep bridging?", trained (Double-DQN) on
    whether the target skill actually succeeds once handed control.

The two phases are sequential: first teach the bridge to move well, then, only after
it is frozen, teach the switch-decider on top of it.

AIRL (Adversarial Inverse Reinforcement Learning) is the step-3 technique. The
discriminator is split into `g` (an action question) and `h` (a state question) so
the reward keeps "is this even a state the skill would be in?" apart from "given the
state, was this the right move?". AIRL only learns a policy that *moves* like the
target; it never decides *when* to stop. That is the switch-decider's job (step 4).

This module holds the inference-time meta policy. Fresh from the constructor its
networks are untrained; train.py fills them in place.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, VecEnvObs
from mjlab.tasks.skills.architectures.arch_1.networks import (
  SwitchQNetwork,
  build_bridge_actor,
)
from mjlab.tasks.skills.meta import MetaPolicy
from mjlab.tasks.skills.skill import SkillPool


class Arch1(MetaPolicy):
  """Meta policy holding one bridge actor and one switch-decider per target skill.

  For each skill in the pool it keeps a bridge actor (an rsl_rl `MLPModel`, trained by
  AIRL+PPO to move like that skill) and a switch-decider (`SwitchQNetwork`, trained to
  answer "hand over now?"). `bridge_step` routes the mid-bridge envs to their target
  skill's actor and switch. Use `train.py` to train the networks.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    pool: SkillPool,
    *,
    actor_hidden_dims: tuple[int, ...] = (64, 64),
    switch_hidden_dims: tuple[int, ...] = (128, 128),
    obs_group: str = "actor",
  ) -> None:

    self.obs_group = obs_group
    self.actor_hidden_dims = actor_hidden_dims
    self.switch_hidden_dims = switch_hidden_dims

    # An obs template to shape the actor's input, and the dims the switch net needs
    obs, _ = env.reset()
    obs_td = TensorDict(obs, batch_size=[env.num_envs])
    state = obs_td[obs_group]
    assert isinstance(state, torch.Tensor)
    obs_dim = state.shape[-1]
    action_dim = env.action_manager.total_action_dim

    # One (actor, switch) per skill, keyed by skill id. Untrained until train.py runs
    self.actors = {
      skill_id: build_bridge_actor(obs_td, action_dim, actor_hidden_dims, env.device)
      for skill_id in range(len(pool))
    }
    self.switches = {
      skill_id: SwitchQNetwork(obs_dim, switch_hidden_dims).to(env.device)
      for skill_id in range(len(pool))
    }

    super().__init__(env, pool)

  @torch.no_grad()
  def bridge_step(
    self,
    obs: VecEnvObs,
    source: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:

    del source  # Source-agnostic: one bridge per target, whatever it came from

    # Homogeneous-batch assumption: the mid-bridge envs all head to the same target
    # (the composition switches every env on the same schedule), so one target's
    # actor/switch serves the batch.
    # TODO Revisit if envs can bridge to different targets at once
    target_id = int(target[active][0])
    actor = self.actors[target_id]
    switch = self.switches[target_id]

    state = obs[self.obs_group]
    assert isinstance(state, torch.Tensor)
    obs_td = TensorDict({self.obs_group: state}, batch_size=[state.shape[0]])
    actions = actor(obs_td)
    handover = active & (switch(state).argmax(dim=-1) == 1)
    return actions, handover

  # What arch_1 needs to persist is one (bridge actor, switch-decider) per skill;
  # they all go in a single file, keyed by skill id.
  _CHECKPOINT = "arch_1.pt"

  def save(self, path: Path) -> None:
    torch.save(
      {
        "actors": {i: actor.state_dict() for i, actor in self.actors.items()},
        "switches": {i: switch.state_dict() for i, switch in self.switches.items()},
      },
      path / self._CHECKPOINT,
    )

  def load(self, path: Path) -> None:
    checkpoint = torch.load(path / self._CHECKPOINT, map_location=self.env.device)
    for i, state_dict in checkpoint["actors"].items():
      self.actors[i].load_state_dict(state_dict)
    for i, state_dict in checkpoint["switches"].items():
      self.switches[i].load_state_dict(state_dict)
