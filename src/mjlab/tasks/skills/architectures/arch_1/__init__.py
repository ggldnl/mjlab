"""Architecture 1: one distribution-matching bridge per target skill.

One bridge per target skill (source-agnostic), trained the way Byun and Perrault
(2022) train a transition policy. For each target skill, training (see train.py) is:

- Step 0: measure the units. Roll every skill in the pool and record the range of
    actions they command and the spread of the states they visit, so everything below
    works on numbers of roughly unit size (see spaces.py).
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
networks are untrained and its unit conversions are the identity; train.py fills both
in place, and both are written to the checkpoint, because an actor restored without the
conversion it was trained under emits numbers that mean something else entirely.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv, VecEnvObs
from mjlab.tasks.skills.architectures.arch_1.networks import (
  SwitchQNetwork,
  bridge_obs_td,
  build_bridge_actor,
)
from mjlab.tasks.skills.architectures.arch_1.spaces import ActionSpace, StateSpace
from mjlab.tasks.skills.meta import MetaPolicy
from mjlab.tasks.skills.skill import SkillPool
from mjlab.tasks.skills.view import StateView, resolve_view


class Arch1(MetaPolicy):
  """Meta policy holding one bridge actor and one switch-decider per target skill.

  For each skill in the pool it keeps a bridge actor (an rsl_rl `MLPModel`, trained by
  AIRL+PPO to move like that skill) and a switch-decider (`SwitchQNetwork`, trained to
  answer "hand over now?"). `bridge_step` routes the mid-bridge envs to their target
  skill's actor and switch. Use `train.py` to train the networks.

  Alongside those it keeps the two unit conversions the whole run shares (see
  spaces.py). They are not per skill: one `ActionSpace` and one `StateSpace` measured
  over the whole pool serve every bridge, which is what makes two bridges comparable.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    pool: SkillPool,
    view: StateView | None = None,
    *,
    actor_hidden_dims: tuple[int, ...] = (64, 64),
    switch_hidden_dims: tuple[int, ...] = (128, 128),
    obs_group: str = "actor",
  ) -> None:
    self.obs_group = obs_group
    self.actor_hidden_dims = actor_hidden_dims
    self.switch_hidden_dims = switch_hidden_dims
    # Both networks live on the experiment's state view, not on the raw observation:
    # they must see exactly what the discriminator that trained them saw (see view.py).
    projection = resolve_view(env, None, obs_group) if view is None else view
    obs_dim = projection.dim
    action_dim = env.action_manager.total_action_dim

    # Identity until training measures them, so a freshly constructed architecture is
    # callable (it just does not do anything useful yet).
    self.action_space = ActionSpace(action_dim).to(env.device)
    self.state_space = StateSpace(obs_dim).to(env.device)

    # An obs template of the right width to shape the actor's input.
    obs, _ = env.reset()
    full_state = obs[obs_group]
    assert isinstance(full_state, torch.Tensor)
    obs_td = bridge_obs_td(self.state_space.standardize(projection(full_state)))

    # One (actor, switch) per skill, keyed by skill id. Untrained until train.py runs
    self.actors = {
      skill_id: build_bridge_actor(obs_td, action_dim, actor_hidden_dims, env.device)
      for skill_id in range(len(pool))
    }
    self.switches = {
      skill_id: SwitchQNetwork(obs_dim, switch_hidden_dims).to(env.device)
      for skill_id in range(len(pool))
    }

    super().__init__(env, pool, projection)

  def adopt_spaces(self, action_space: ActionSpace, state_space: StateSpace) -> None:
    """Take on the unit conversions a training run measured. Called by train.py."""
    self.action_space = action_space.to(self.env.device)
    self.state_space = state_space.to(self.env.device)

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

    full_state = obs[self.obs_group]
    assert isinstance(full_state, torch.Tensor)
    # Raw observation -> the experiment's view of it -> the units the nets were trained
    # in. Every network below reads the last of those three and nothing else.
    state = self.state_space.standardize(self.view(full_state))

    # The actor emits a normalized action; the env wants a raw one.
    normalized_action = actor(bridge_obs_td(state))
    actions = self.action_space.denormalize(normalized_action)

    handover = active & (switch(state).argmax(dim=-1) == 1)
    return actions, handover

  # What arch_1 needs to persist is one (bridge actor, switch-decider) per skill plus
  # the two unit conversions they were trained under; they all go in a single file.
  _CHECKPOINT = "arch_1.pt"

  def save(self, path: Path) -> None:
    torch.save(
      {
        "actors": {i: actor.state_dict() for i, actor in self.actors.items()},
        "switches": {i: switch.state_dict() for i, switch in self.switches.items()},
        "action_space": self.action_space.state_dict(),
        "state_space": self.state_space.state_dict(),
      },
      path / self._CHECKPOINT,
    )

  def load(self, path: Path) -> None:
    checkpoint = torch.load(path / self._CHECKPOINT, map_location=self.env.device)
    for i, state_dict in checkpoint["actors"].items():
      self.actors[i].load_state_dict(state_dict)
    for i, state_dict in checkpoint["switches"].items():
      self.switches[i].load_state_dict(state_dict)
    # A checkpoint written before the unit conversions existed has neither, and an actor
    # from one is not interpretable without them, so this refuses rather than silently
    # running the identity conversion.
    missing = {"action_space", "state_space"} - set(checkpoint)
    if missing:
      raise KeyError(
        f"{path / self._CHECKPOINT} has no {sorted(missing)}. It predates the unit "
        f"conversions in spaces.py, and its actor's output cannot be interpreted "
        f"without them. Retrain."
      )
    self.action_space.load_state_dict(checkpoint["action_space"])
    self.state_space.load_state_dict(checkpoint["state_space"])
