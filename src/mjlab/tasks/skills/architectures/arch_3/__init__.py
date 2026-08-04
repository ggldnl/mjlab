"""Architecture 3: a transition is a fade from one skill to the next, plus a residual.

No bridge policy and no hand-over decision. A transition is a fixed number of steps over
which control dissolves from the skill being left into the skill being entered, with a
learned correction riding on top that is scaled by the same dial and expires with it:

    a = (1 - alpha) * target(s) + alpha * source(s) + alpha * residual(s, alpha, blend)

alpha is 1 the step the switch fires and 0 exactly `transition_steps` later. Read the two
ends. At alpha = 1 the command is precisely the source skill's, so the transition begins
with no discontinuity at all. At alpha = 0 both the source and the correction are gone
and the command is precisely the target's, so the moment control formally passes is a
no-op. Continuity at both ends is a property of the form, not something training has to
discover.

That is why this exists. In arch_1 and arch_2 something has to answer "is now a good
moment to let go?", which needs a second phase, a Q-network and a notion of what a good
hand-over looks like. Here the question is never asked: the schedule answers it in
advance, and the residual's problem becomes "make the moment you were given work".

Why anchor the fade on the source skill rather than a bridge trained from scratch.
Regroup:

    a = (1 - alpha) * target(s) + alpha * (source(s) + residual(...))

which is a bridge policy whose authority decays, with that policy defined as "the skill
we are leaving, corrected". It spans the same family a from-scratch bridge would (the
residual is not restricted to small values), so the difference is entirely where learning
starts. With the residual at zero, which is where it starts, the architecture already
performs a smooth dissolve. That is a competent transition to improve on rather than a
random network with full authority, and it gives an honest baseline: run it with the
residual disabled and the comparison says what the learning bought.

The cost is that the anchor is a bias whose weight is largest at the very start, which is
the moment one most wants to stop doing what the source was doing. If that shows up,
dropping the anchor is a one-line change, because the two forms are the same equation.

Two consequences for the rest of the package, both load-bearing. Both skills act while a
transition runs, so both must appear in the step's involvement mask (see `involved`). And
the target skill is engaged the step the switch fires, not at the hand-over, because it
contributes from the first step of the blend.
"""

from __future__ import annotations

from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv, VecEnvObs
from mjlab.tasks.skills.architectures.common.nets import build_actor, obs_td
from mjlab.tasks.skills.meta import MetaPolicy
from mjlab.tasks.skills.skill import NO_SKILL, SkillPool
from mjlab.tasks.skills.spaces import ActionSpace, StateSpace, Units
from mjlab.tasks.skills.view import StateView, resolve_view


def alpha_at(elapsed: torch.Tensor, steps: torch.Tensor | int) -> torch.Tensor:
  """How much authority the source side still has, `elapsed` steps into a transition.

  1 on the step the switch fires and 0 from step `steps` onward, linearly in between.
  `steps` may be a per-env tensor, which is what training uses to cover a range of
  transition lengths with one residual.
  """
  if not isinstance(steps, torch.Tensor):
    steps = torch.as_tensor(float(steps), device=elapsed.device)
  return (1.0 - elapsed.float() / steps.clamp_min(1).float()).clamp(0.0, 1.0)


def residual_input(
  state: torch.Tensor, alpha: torch.Tensor, normalized_blend: torch.Tensor
) -> torch.Tensor:
  """What the residual reads: the state, the dial, and the action it is correcting.

  alpha has to be in here: without it the residual sees the same state early in a
  transition (full authority, robot still doing the old thing) and late in one (almost
  none), and those demand different corrections. The blended action is in here because it
  is the residual's actual question -- "this is what the fade is about to command, in this
  state" is far more direct to correct than a state alone.
  """
  return torch.cat([state, alpha.unsqueeze(-1), normalized_blend], dim=-1)


def compose(
  action_space: ActionSpace,
  source_action: torch.Tensor,
  target_action: torch.Tensor,
  alpha: torch.Tensor,
  residual: torch.Tensor,
  residual_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """The blend and the raw action to step, given both skills' actions and a residual.

  Returns (normalized blend, raw action). The residual is bounded and applied in
  normalized action units, so `residual_scale` reads as "how much of the range the pool
  commands may the correction move the action by". Bounding it is the ordinary precaution
  against a network with full authority in its first iterations; a clamp rather than a
  squash, so what PPO optimizes and what the env receives differ only where it bites.
  """
  weight = alpha.unsqueeze(-1)
  blend = (1.0 - weight) * target_action + weight * source_action
  normalized_blend = action_space.normalize(blend)
  correction = residual.clamp(-residual_scale, residual_scale)
  return normalized_blend, action_space.denormalize(
    normalized_blend + weight * correction
  )


class Arch3(MetaPolicy):
  """Meta policy holding one residual network per target skill.

  The fade itself is arithmetic and needs no parameters. What is learned is one residual
  per skill being entered, source-agnostic like arch_1's bridge: which skill is being left
  enters through its own action in the blend rather than through the network's weights,
  which is a further nicety of anchoring the fade on the source.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    pool: SkillPool,
    view: StateView | None = None,
    *,
    residual_hidden_dims: tuple[int, ...] = (64, 64),
    transition_steps: int = 32,
    residual_scale: float = 1.0,
    obs_group: str = "actor",
  ) -> None:
    self.obs_group = obs_group
    self.residual_hidden_dims = residual_hidden_dims
    self.transition_steps = transition_steps
    self.residual_scale = residual_scale

    projection = resolve_view(env, None, obs_group) if view is None else view
    obs_dim = projection.dim
    action_dim = env.action_manager.total_action_dim

    # Identity until training measures them, so a freshly constructed architecture is
    # callable (it just does not do anything useful yet).
    self.action_space = ActionSpace(action_dim).to(env.device)
    self.state_space = StateSpace(obs_dim).to(env.device)

    # Shapes only: an input template of the right width to size the residuals.
    template = obs_td(
      residual_input(
        torch.zeros(1, obs_dim, device=env.device),
        torch.ones(1, device=env.device),
        torch.zeros(1, action_dim, device=env.device),
      )
    )
    self.residuals = {
      skill_id: build_actor(template, action_dim, residual_hidden_dims, env.device)
      for skill_id in range(len(pool))
    }

    super().__init__(env, pool, projection)

  def adopt_units(self, units: Units) -> None:
    """Take on the unit conversions a training run measured. Called by train.py."""
    self.action_space = units.action.to(self.env.device)
    self.state_space = units.state.to(self.env.device)

  ##
  # Transition bookkeeping: how far into its fade each env is.
  ##

  def reset(self) -> None:
    super().reset()
    self._elapsed = torch.zeros(
      self.env.num_envs, dtype=torch.long, device=self.env.device
    )

  def notify_reset(self, done: torch.Tensor) -> None:
    super().notify_reset(done)
    self._elapsed = torch.where(done, torch.zeros_like(self._elapsed), self._elapsed)

  def begin_switch(
    self, switching: torch.Tensor, source: torch.Tensor, target: torch.Tensor
  ) -> None:
    del source, target
    self._elapsed = torch.where(
      switching, torch.zeros_like(self._elapsed), self._elapsed
    )
    # The target contributes from the first step of the fade, so it takes control now
    # rather than at the hand-over. A skill with memory has no later moment to be told.
    self.engage(switching)

  def involved(self, assignment: torch.Tensor) -> torch.Tensor:
    """Both ends of a running transition are driving, on top of the plain assignment."""
    involved = super().involved(assignment)
    for ids in (self._source, self._target):
      driving = self._bridging & (ids >= 0)
      if bool(driving.any()):
        involved = involved | self.pool.involvement(
          torch.where(driving, ids, torch.full_like(ids, NO_SKILL))
        )
    return involved

  @torch.no_grad()
  def bridge_step(
    self,
    obs: VecEnvObs,
    skill_actions: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    # Homogeneous-batch assumption: the mid-transition envs all head to the same target
    # (the composition switches every env on the same schedule), so one target's residual
    # serves the batch.
    # TODO Revisit if envs can bridge to different targets at once
    target_id = int(target[active][0])

    full_state = obs[self.obs_group]
    assert isinstance(full_state, torch.Tensor)
    state = self.state_space.standardize(self.view(full_state))

    alpha = alpha_at(self._elapsed, self.transition_steps)
    idle = torch.full_like(target, NO_SKILL)
    source_action = SkillPool.select(skill_actions, torch.where(active, source, idle))
    target_action = SkillPool.select(skill_actions, torch.where(active, target, idle))

    # Two passes over the same arithmetic, because the residual reads the blend it is
    # correcting: compose once with no correction to get it, then again for real.
    normalized_blend, _ = compose(
      self.action_space,
      source_action,
      target_action,
      alpha,
      torch.zeros_like(source_action),
      self.residual_scale,
    )
    correction = self.residuals[target_id](
      obs_td(residual_input(state, alpha, normalized_blend))
    )
    _, actions = compose(
      self.action_space,
      source_action,
      target_action,
      alpha,
      correction,
      self.residual_scale,
    )

    self._elapsed = torch.where(active, self._elapsed + 1, self._elapsed)
    # The fade is over exactly where alpha has reached zero, and on that step the action
    # above is already the target skill's own, so handing over changes nothing.
    handover = active & (alpha <= 0.0)
    return actions, handover

  # One residual per skill plus the units and the schedule they were trained under.
  _CHECKPOINT = "arch_3.pt"

  def save(self, path: Path) -> None:
    torch.save(
      {
        "residuals": {i: net.state_dict() for i, net in self.residuals.items()},
        "action_space": self.action_space.state_dict(),
        "state_space": self.state_space.state_dict(),
        "transition_steps": self.transition_steps,
        "residual_scale": self.residual_scale,
      },
      path / self._CHECKPOINT,
    )

  def load(self, path: Path) -> None:
    checkpoint = torch.load(path / self._CHECKPOINT, map_location=self.env.device)
    for i, state_dict in checkpoint["residuals"].items():
      self.residuals[i].load_state_dict(state_dict)
    self.action_space.load_state_dict(checkpoint["action_space"])
    self.state_space.load_state_dict(checkpoint["state_space"])
    # The schedule is as much a part of a trained residual as its weights: the network
    # reads alpha, so replaying it on a different fade queries it off distribution.
    self.transition_steps = int(checkpoint["transition_steps"])
    self.residual_scale = float(checkpoint["residual_scale"])
