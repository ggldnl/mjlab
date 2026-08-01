"""The meta policy: the skill pool plus the machinery that carries out a switch.

A meta policy owns the frozen skill pool and whatever an architecture needs to move
between skills (a learned bridge, a switch-decider, or nothing at all). It does not
own the controller. Which skill should be running is decided elsewhere and handed
in per step; the meta policy only owns how a commanded switch is executed:
immediately (arch_0's direct hand-off) or through a bridge that first drives
the robot into a state the next skill can safely start from (arch_1).

Each concrete architecture is a `MetaPolicy` subclass. The base class implements the
one shared piece: the per-env bookkeeping of which skill control is committed to,
which one it came from, and which envs are mid-bridge. It leaves the
architecture-specific behavior to two hooks: `begin_switch` (called once when a
switch fires) and `bridge_step` (the actions and hand-over decision while bridging).

`ComposedPolicy` pairs a controller with a meta policy into a single `policy(obs)`
that a viewer or `run_episode` can drive. Keeping the controller out of the meta
policy is deliberate: it is the one experiment-specific part, swapped freely without
touching the switching machinery.

`bridge_step` is handed the full per-env `source`/`target`/`active` tensors, so an
architecture that keeps a different bridge per (source, target) pair or per target is
free to route internally.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from mjlab.envs import ManagerBasedRlEnv, VecEnvObs
from mjlab.tasks.skills.controller import Controller
from mjlab.tasks.skills.skill import NO_SKILL, SkillPool
from mjlab.tasks.skills.view import StateView, resolve_view


class MetaPolicy(ABC):
  """Skill pool plus the architecture-specific machinery to switch between skills.

  Given the controller's desired skill per env, produces actions, engaging this
  architecture's switching machinery whenever the desired skill differs from the one
  control is already committed to.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    pool: SkillPool,
    view: StateView | None = None,
  ) -> None:
    """`view` is the slice of the observation this architecture's bridging machinery
    works on (see view.py); None means the whole observation. It belongs to the
    experiment rather than to the architecture, so every architecture takes it here and
    one that has no machinery to point it at (arch_0) simply ignores it."""
    self.env = env
    self.pool = pool
    self.view = resolve_view(env, None) if view is None else view
    self.reset()

  @property
  def target(self) -> torch.Tensor:
    """The skill control is committed to in each env, shaped (num_envs,) int64.

    The running skill in normal operation, or the skill a bridge is driving toward
    during a transition. NO_SKILL in an env that has not been commanded yet (right
    after a reset), which is exactly the controller's cue to decide fresh.
    """
    return self._target

  def reset(self) -> None:
    """Restart the composition, as if a fresh episode were beginning in every env.

    Takes no arguments so a viewer's reset button can call it directly
    (`BaseViewer.reset_environment` resets the env, then calls this).
    """
    num_envs = self.env.num_envs
    device = self.env.device
    self.pool.reset(torch.ones(num_envs, dtype=torch.bool, device=device))
    self._target = torch.full((num_envs,), NO_SKILL, dtype=torch.long, device=device)
    self._source = torch.full_like(self._target, NO_SKILL)
    self._bridging = torch.zeros(num_envs, dtype=torch.bool, device=device)

  def notify_reset(self, done: torch.Tensor) -> None:
    """Clear per-episode state in the envs whose episode just ended.

    The pool and our own commitment bookkeeping must not carry across an episode
    boundary. The committed target drops back to NO_SKILL so the next command is
    adopted fresh: a new episode starts its skill directly and is never bridged into.
    """
    if not done.any():
      return
    self.pool.reset(done)
    self._target = torch.where(
      done, torch.full_like(self._target, NO_SKILL), self._target
    )
    self._source = torch.where(
      done, torch.full_like(self._source, NO_SKILL), self._source
    )
    self._bridging = self._bridging & ~done

  def act(self, obs: VecEnvObs, command: torch.Tensor) -> torch.Tensor:
    """Actions for every env given the controller's desired skill per env.

    `command` is what the controller wants running right now. Where it differs from
    the skill we are already committed to, that is the switch signal (per env, so
    different envs can switch on different steps).
    """
    # Envs with no commitment yet (a fresh episode) adopt the command with no switch:
    # a new episode starts its skill directly, it is never bridged into.
    fresh = self._target == NO_SKILL
    if fresh.any():
      self._target = torch.where(fresh, command, self._target)

    # A switch is a command that differs from the skill control is committed to.
    switching = (command != self._target) & ~fresh
    if switching.any():
      # The skill being left becomes the source the bridge starts from, advanced
      # before it is used so the bridge sees where control actually came from.
      self._source = torch.where(switching, self._target, self._source)
      self._target = torch.where(switching, command, self._target)
      self._bridging = self._bridging | switching
      self.begin_switch(switching, self._source, self._target)

    # Skill actions for every non-bridging env; bridging envs are set to NO_SKILL
    # (SkillPool.act returns zeros for them) and overwritten with the bridge's below.
    assignment = torch.where(
      self._bridging, torch.full_like(self._target, NO_SKILL), self._target
    )
    actions = self.pool.act(obs, assignment)

    if self._bridging.any():
      bridge_actions, handover = self.bridge_step(
        obs, self._source, self._target, self._bridging
      )
      actions = torch.where(self._bridging.unsqueeze(-1), bridge_actions, actions)
      # Wherever the bridge signals hand-over, the target skill takes over next step.
      self._bridging = self._bridging & ~handover

    return actions

  def active_label(self, env_idx: int = 0) -> str:
    """What env env_idx is doing right now, as a short human-readable string.

    The running skill's name in normal operation, or "bridge: A -> B" while this
    architecture is driving a transition. Used by the viewer's info box; harmless to
    call at any time.
    """
    if bool(self._bridging[env_idx]):
      source = int(self._source[env_idx])
      target = int(self._target[env_idx])
      source_name = self.pool[source].name if source >= 0 else "?"
      target_name = self.pool[target].name if target >= 0 else "?"
      return f"bridge: {source_name} -> {target_name}"
    target = int(self._target[env_idx])
    return "-" if target < 0 else self.pool[target].name

  def begin_switch(  # noqa: B027
    self, switching: torch.Tensor, source: torch.Tensor, target: torch.Tensor
  ) -> None:
    """Start a transition in the envs where `switching` is set.

    Called once when a switch fires, before the first `bridge_step` of that
    transition. The default does nothing, which is right for a stateless bridge.
    """

  @abstractmethod
  def bridge_step(
    self,
    obs: VecEnvObs,
    source: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Actions and a hand-over signal for the envs this bridge is driving.

    `source`/`target` hold, per env, the skill control came from and the one it is
    headed to; `active` marks the mid-bridge envs. Returns actions shaped
    (num_envs, action_dim) and a boolean mask (num_envs,) set where the bridge
    declares the target skill may take over now (choosing that moment belongs to the
    architecture).
    """

  def save(self, path: Path) -> None:  # noqa: B027
    """Persist this architecture's trained state into directory `path`.

    `path` is an existing directory owned by one training run. The default writes
    nothing, which is exactly right for an architecture that trains nothing (arch_0).
    Architectures holding networks override this to write whatever they need; the
    matching `load` reads it back. What each writes is its own business -- only the
    save/load pair is shared.
    """

  def load(self, path: Path) -> None:  # noqa: B027
    """Restore trained state from a directory an earlier `save` wrote to.

    The default restores nothing (arch_0). Overrides mirror `save`.
    """


class ComposedPolicy:
  """Pairs a controller with a meta policy into one `policy(obs)`.

  Each step: ask the controller which skill should run (given what the meta policy is
  currently committed to), then let the meta policy carry out any switch. Also absorbs
  resets, clearing the controller and the meta policy in whichever envs just started a
  fresh episode, since a caller owning `env.step` never hands the done flags back.

  A fresh episode is read off `episode_length_buf`, not off `reset_buf`. The two agree
  on an auto-reset, but only the counter also catches a reset the caller asked for:
  `env.reset()` (the viewer's reset button) and `env.reset(env_ids=...)` (its per-env
  one) both rewind the counter and neither touches `reset_buf`, which still holds
  whatever the last `step` computed. Reading `reset_buf` there leaves the composition
  committed to the skill it was running before the reset, so a restarted episode carries
  on mid-sequence instead of beginning at the controller's first skill.
  """

  def __init__(
    self, env: ManagerBasedRlEnv, controller: Controller, meta: MetaPolicy
  ) -> None:
    self.env = env
    self.controller = controller
    self.meta = meta
    self.reset()

  def reset(self) -> None:
    everything = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
    self.controller.reset(everything)
    self.meta.reset()
    self._just_reset = True

  def __call__(self, obs: Any) -> torch.Tensor:
    """Actions for every env. `obs` is a `VecEnvObs`.

    Typed loosely so this object satisfies the viewer's `PolicyProtocol` (which declares
    a plain tensor) and can be handed to a viewer as-is. That matters beyond typing: a
    viewer resets the policy through `getattr(policy, "reset")`, so wrapping this in a
    lambda to appease the annotation would hide `reset` and leave the composition
    running its old skill after a reset.
    """
    if self._just_reset:
      self._just_reset = False
    else:
      fresh = self.env.episode_length_buf == 0
      if fresh.any():
        self.controller.reset(fresh)
        self.meta.notify_reset(fresh)
    command = self.controller.decide(self.env, self.meta.target)
    return self.meta.act(obs, command)

  def active_label(self, env_idx: int = 0) -> str:
    """What env `env_idx` is doing right now (delegates to the meta policy)."""
    return self.meta.active_label(env_idx)


def run_episode(
  env: ManagerBasedRlEnv,
  controller: Controller,
  meta: MetaPolicy,
  num_steps: int,
) -> None:
  obs, _ = env.reset()
  policy = ComposedPolicy(env, controller, meta)
  for _ in range(num_steps):
    obs, _, _, _, _ = env.step(policy(obs))
