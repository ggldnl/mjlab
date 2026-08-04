"""Architecture 1: one distribution-matching bridge per target skill.

One bridge per target skill, source-agnostic, trained the way Byun and Perrault (2022)
train a transition policy. Two sequential phases (see train.py):

- Phase 1, move like the target. Record the target skill's opening window, harvest states
  from the other skills to be dropped into, then play a copying game: a referee learns to
  tell the bridge's rollouts from the recording, and PPO trains the bridge on the
  referee's verdict.
- Phase 2, decide when to let go. Freeze the bridge and train a small Q-network that only
  ever answers "hand over now, or keep bridging?", by Double-DQN, on whether the target
  skill actually succeeded once handed control.

The split is the architecture's whole shape. The copying game teaches a policy that
*moves* like the target and has no opinion about when to stop; that is the decider's job,
and it is the part that needs a notion of success (see switch.py).

Fresh from the constructor the networks are untrained and the unit conversions are the
identity. train.py fills both in place, and both go in the checkpoint: an actor restored
without the conversion it was trained under emits numbers that mean something else.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv, VecEnvObs
from mjlab.tasks.skills.architectures.arch_1.switch import SwitchQNetwork
from mjlab.tasks.skills.architectures.common.nets import build_actor, obs_td
from mjlab.tasks.skills.meta import MetaPolicy
from mjlab.tasks.skills.skill import SkillPool
from mjlab.tasks.skills.spaces import ActionSpace, StateSpace, Units
from mjlab.tasks.skills.view import StateView, resolve_view


class Arch1(MetaPolicy):
  """Meta policy holding one bridge actor and one hand-over decider per target skill.

  `bridge_step` routes the mid-transition envs to their target skill's pair. Alongside
  them it keeps the two unit conversions the run shares (spaces.py). Those are not per
  skill: one of each, measured over the whole pool, is what makes two bridges comparable.
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
    # Both networks live on the experiment's state view, not the raw observation: they
    # must see exactly what the referee that trained them saw (see view.py).
    projection = resolve_view(env, None, obs_group) if view is None else view
    obs_dim = projection.dim
    action_dim = env.action_manager.total_action_dim

    # Identity until training measures them, so a freshly constructed architecture is
    # callable (it just does not do anything useful yet).
    self.action_space = ActionSpace(action_dim).to(env.device)
    self.state_space = StateSpace(obs_dim).to(env.device)

    # Shapes only: an input template of the right width to size the actor.
    template = obs_td(torch.zeros(1, obs_dim, device=env.device))
    self.actors = {
      skill_id: build_actor(template, action_dim, actor_hidden_dims, env.device)
      for skill_id in range(len(pool))
    }
    self.switches = {
      skill_id: SwitchQNetwork(obs_dim, switch_hidden_dims).to(env.device)
      for skill_id in range(len(pool))
    }

    super().__init__(env, pool, projection)

  def adopt_units(self, units: Units) -> None:
    """Take on the unit conversions a training run measured. Called by train.py."""
    self.action_space = units.action.to(self.env.device)
    self.state_space = units.state.to(self.env.device)

  @torch.no_grad()
  def bridge_step(
    self,
    obs: VecEnvObs,
    skill_actions: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del source  # Source-agnostic: one bridge per target, whatever it came from
    del skill_actions  # The bridge drives on its own; neither skill acts while it does

    # Homogeneous-batch assumption: the mid-transition envs all head to the same target
    # (the composition switches every env on the same schedule), so one target's pair
    # serves the batch.
    # TODO Revisit if envs can bridge to different targets at once
    target_id = int(target[active][0])
    actor = self.actors[target_id]
    switch = self.switches[target_id]

    full_state = obs[self.obs_group]
    assert isinstance(full_state, torch.Tensor)
    # Raw observation -> the experiment's view -> the units the nets were trained in.
    # Every network below reads the last of those three and nothing else.
    state = self.state_space.standardize(self.view(full_state))

    actions = self.action_space.denormalize(actor(obs_td(state)))
    handover = active & (switch(state).argmax(dim=-1) == 1)
    return actions, handover

  # One (actor, decider) per skill plus the two conversions they were trained under.
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
    missing = {"action_space", "state_space"} - set(checkpoint)
    if missing:
      raise KeyError(
        f"{path / self._CHECKPOINT} has no {sorted(missing)}. It predates the unit "
        f"conversions in spaces.py, and its actor's output cannot be interpreted "
        f"without them. Retrain."
      )
    self.action_space.load_state_dict(checkpoint["action_space"])
    self.state_space.load_state_dict(checkpoint["state_space"])
