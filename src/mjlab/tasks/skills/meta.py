"""The meta policy: the skill pool plus the machinery that carries out a switch.

A meta policy owns the frozen pool and whatever an architecture needs to move between
skills. It does not own the controller: which skill should run is decided elsewhere
and handed in per step, and all a meta policy owns is how a commanded switch is carried
out.

Each architecture is a `MetaPolicy` subclass. The base class implements the one shared
piece, the per-env bookkeeping of which skill control is committed to, which one it came
from, and which envs are mid-transition. Architectures fill in three hooks:

    begin_switch    once, when a switch fires
    involved        each step, which skills have their hands on the wheel
    bridge_step     each step, the actions and the hand-over decision

One step, in order:

    fresh envs adopt the command directly (a new episode is never bridged into)
    command != committed target  ->  a switch fires  ->  begin_switch
    engage anything about to act for the first time  ->  skill.reset()
    involved()  ->  one pool tick  ->  select rows
    bridge_step() overwrites the mid-transition envs, and may hand over

`involved` is the load-bearing hook and the easiest to get wrong. The pool is ticked
once per step (see skill.py), so the mask must name every skill whose action will be
used that step, not only the ones formally in control. A bridge that drives on its own:
names neither end (arch_1, arch_4_backup); a bridge built out of the skills' own actions: names
both (arch_3); a hand-off that is immediate: names the target (arch_0).

`ComposedPolicy` pairs a controller with a meta policy into one `policy(obs)` a viewer
can drive. Keeping the controller out of the meta policy is deliberate: it is the one
experiment-specific part, swapped without touching the switching machinery.
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
  """Skill pool plus the architecture-specific machinery to switch between skills."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    pool: SkillPool,
    view: StateView | None = None,
  ) -> None:
    """`view` is the slice of the observation this architecture works on (see view.py).

    It belongs to the experiment rather than to the architecture, so every architecture
    takes it and one with no machinery to point it at (arch_0) ignores it. None means
    the whole observation.
    """
    self.env = env
    self.pool = pool
    self.view = resolve_view(env, None) if view is None else view
    self.reset()

  @property
  def target(self) -> torch.Tensor:
    """The skill control is committed to in each env, shaped (num_envs,) int64.

    The running skill in normal operation, or the skill a transition is driving toward.
    NO_SKILL in an env that has not been commanded yet, which is the controller's cue to
    decide fresh.
    """
    return self._target

  ##
  # Bookkeeping.
  ##

  def reset(self) -> None:
    """Restart the composition, as if a fresh episode were beginning in every env.

    Takes no arguments so a viewer's reset button can call it directly.
    """
    num_envs = self.env.num_envs
    device = self.env.device
    self.pool.reset(torch.ones(num_envs, dtype=torch.bool, device=device))
    self._target = torch.full((num_envs,), NO_SKILL, dtype=torch.long, device=device)
    self._source = torch.full_like(self._target, NO_SKILL)
    self._bridging = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self._engaged = torch.zeros(num_envs, dtype=torch.bool, device=device)

  def notify_reset(self, done: torch.Tensor) -> None:
    """Clear per-episode state in the envs whose episode just ended.

    The committed target drops back to NO_SKILL so the next command is adopted fresh: a
    new episode starts its skill directly and is never bridged into.
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
    self._engaged = self._engaged & ~done

  ##
  # The step.
  ##

  def act(self, obs: VecEnvObs, command: torch.Tensor) -> torch.Tensor:
    """Actions for every env, given the controller's desired skill per env."""
    # A fresh episode adopts the command with no switch.
    fresh = self._target == NO_SKILL
    if fresh.any():
      self._target = torch.where(fresh, command, self._target)
      self._engaged = self._engaged & ~fresh

    # A command that differs from what control is committed to *is* the switch signal.
    switching = (command != self._target) & ~fresh
    if switching.any():
      self._source = torch.where(switching, self._target, self._source)
      self._target = torch.where(switching, command, self._target)
      self._bridging = self._bridging | switching
      self._engaged = self._engaged & ~switching
      self.begin_switch(switching, self._source, self._target)

    # Whatever is about to act for the first time is engaged first: a skill with memory
    # has no other moment to learn that control has arrived.
    self.engage(~self._bridging & ~self._engaged)

    # One pool tick, whatever the actions are then used for. An architecture that builds
    # its transition out of the skills' own actions says so in `involved` and reads them
    # off `skill_actions` rather than ticking the pool again.
    assignment = torch.where(
      self._bridging, torch.full_like(self._target, NO_SKILL), self._target
    )
    skill_actions = self.pool.act_each(obs, self.involved(assignment))
    actions = SkillPool.select(skill_actions, assignment)

    if self._bridging.any():
      bridge_actions, handover = self.bridge_step(
        obs, skill_actions, self._source, self._target, self._bridging
      )
      actions = torch.where(self._bridging.unsqueeze(-1), bridge_actions, actions)
      # Where the architecture signals hand-over, the target takes over next step.
      self._bridging = self._bridging & ~handover

    return actions

  def engage(self, mask: torch.Tensor) -> None:
    """Hand control to the target skill where `mask` is set, by calling its `reset`.

    This is how a skill with memory is told it is starting now. The jump is why it
    exists: its reset pins the reference clip to wherever the robot currently is.

    Called by the base class the step before a skill's first action, and from
    `begin_switch` by an architecture whose target skill acts on the switch step itself
    (arch_0, arch_3). Narrowed to envs not already engaged, since engaging twice rewinds
    a skill that has already started.
    """
    mask = mask & ~self._engaged
    if not mask.any():
      return
    for skill_id in self._target[mask].unique().tolist():
      if skill_id == NO_SKILL:
        continue
      self.pool[skill_id].reset(mask & (self._target == skill_id))
    self._engaged = self._engaged | mask

  ##
  # The hooks.
  ##

  def begin_switch(  # noqa: B027
    self, switching: torch.Tensor, source: torch.Tensor, target: torch.Tensor
  ) -> None:
    """Start a transition where `switching` is set, before its first `bridge_step`.

    The default does nothing, which is right for an architecture that drives the robot
    itself for a while: the target is engaged by the base class once it hands over. One
    that hands over at once has to engage the target here instead.
    """

  def involved(self, assignment: torch.Tensor) -> torch.Tensor:
    """Which envs each skill drives this step, shaped (num_skills, num_envs).

    The default is the assignment itself, right for a bridge that drives on its own: the
    skills at either end contribute nothing while it runs. Override to add any skill
    whose action the transition uses (see the module docstring).
    """
    return self.pool.involvement(assignment)

  @abstractmethod
  def bridge_step(
    self,
    obs: VecEnvObs,
    skill_actions: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Actions and a hand-over signal for the envs this transition is driving.

    `skill_actions` is this step's pool tick, shaped (num_skills, num_envs, action_dim),
    covering whatever `involved` asked for; `SkillPool.select` picks rows out of it.
    `source`/`target` hold, per env, the skill control came from and the one it is headed
    to; `active` marks the mid-transition envs. Returns actions (num_envs, action_dim)
    and a bool mask (num_envs,) set where the target skill may take over now.
    """

  ##
  # Persistence and display.
  ##

  def save(self, path: Path) -> None:  # noqa: B027
    """Persist trained state into the existing directory `path`.

    The default writes nothing, right for an architecture that trains nothing (arch_0).
    What each writes is its own business; only the save/load pair is shared.
    """

  def load(self, path: Path) -> None:  # noqa: B027
    """Restore trained state from a directory an earlier `save` wrote to."""

  def active_label(self, env_idx: int = 0) -> str:
    """What env `env_idx` is doing right now, for the viewer's info box."""
    if bool(self._bridging[env_idx]):
      source = int(self._source[env_idx])
      target = int(self._target[env_idx])
      source_name = self.pool[source].name if source >= 0 else "?"
      target_name = self.pool[target].name if target >= 0 else "?"
      return f"bridge: {source_name} -> {target_name}"
    target = int(self._target[env_idx])
    return "-" if target < 0 else self.pool[target].name


class ComposedPolicy:
  """Pairs a controller with a meta policy into one `policy(obs)`.

  Each step: ask the controller which skill should run, then let the meta policy carry
  out any switch. Also absorbs resets, since a caller owning `env.step` never hands the
  done flags back.

  A fresh episode is read off `episode_length_buf`, not `reset_buf`. The two agree on an
  auto-reset, but only the counter also catches a reset the caller asked for: the
  viewer's reset button calls `env.reset()`, which rewinds the counter and never touches
  `reset_buf`, so reading `reset_buf` would leave the composition mid-sequence.
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

    Typed loosely so this object satisfies the viewer's `PolicyProtocol` and can be
    handed to a viewer as-is. That matters beyond typing: a viewer resets the policy
    through `getattr(policy, "reset")`, so wrapping this in a lambda would hide `reset`
    and leave the composition running its old skill after a reset.
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
