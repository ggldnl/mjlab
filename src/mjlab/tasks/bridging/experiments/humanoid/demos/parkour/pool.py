"""The skills, each wrapping one frozen policy and everything it needs telling.

    pool = SkillPool.load(ROSTER, env, device)
    pool["climb"].enter(env, pos, quat, frame)   put its reference where it belongs
    pool["walk"].tell(forward=0.7, heading=0.1)  what it is being asked for
    pool["walk"].condition(env)                  write that where the policy reads it
    action = pool["walk"](obs)                   drive

A `Skill` is the seam between the controller and a checkpoint. The controller knows the
world and decides; a skill knows how its own policy has to be spoken to and nothing about
courses or obstacles. That split is why the controller can be rewritten without touching a
skill and why a new skill costs a declaration rather than a special case.

What a skill can be asked to do is whatever its `Actor` declares, and that lives in
`tests.actors` beside the skill rather than here. This adds the policy, the knob values and
the entry table to it, and no behaviour of its own.

Two loading steps, on purpose:

    resolve   names to checkpoint paths. No simulation, so a missing file is reported
              before a minute of arena build rather than after it
    load      those paths into policies. Needs the env, because a policy binds to the
              observation group it will read

Run

Nothing here is runnable on its own. See run.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.bridge import BRIDGE_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.selector import (
  Entry,
  EntryTable,
  Reach,
  nearest,
)
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import (
  BRIDGE_GROUP,
  Actor,
  Policy,
  defaults,
  find_checkpoint,
)
from mjlab.tasks.registry import load_rl_cfg

BRIDGE = Actor(BRIDGE_GROUP, BRIDGE_TASK_ID)
"""The bridge as an actor, so it loads through the same path as a skill.

It declares nothing else: no controls, because the controller aims it rather than telling it
anything; no `enter`, because it has no reference to place; no `place`, because it puts
nothing on the floor."""


class Skill:
  """One frozen policy, what it is currently being told, and where it can be joined."""

  def __init__(
    self, actor: Actor, policy: Policy, entries: tuple[Entry, ...], table: EntryTable
  ) -> None:
    self.actor = actor
    self.policy = policy
    self.entries = entries
    self._table = table
    self.values: dict[str, float] = defaults(actor)
    """What this skill is being asked for, by knob name. `condition` and `enter` read it.

    Held per skill rather than in one dictionary, because two skills can share a command
    term, which the walk and the run do, and a term written by both at once holds whichever
    wrote last. Only the driving skill's values are ever written to the world."""

  @property
  def name(self) -> str:
    return self.actor.name

  ##
  # Being told things.
  ##

  def tell(self, **values: float) -> None:
    """Set what this skill is being asked for. Raises on a knob it does not have.

    Raising rather than ignoring, because a control written under the wrong name is a
    skill quietly running on its defaults, which looks like a bad policy.
    """
    for name, value in values.items():
      if name not in self.values:
        raise KeyError(
          f"'{self.name}' has no control called '{name}'. It takes "
          f"{sorted(self.values) or 'nothing'}."
        )
      self.values[name] = float(value)

  def condition(self, env: ManagerBasedRlEnv) -> None:
    """Write those values into whatever the policy reads. Every step it drives.

    Idempotent by contract, so calling it on a step where nothing changed is free. Called
    only while this skill owns the world: see `values`.
    """
    if self.actor.condition is not None:
      self.actor.condition(env, self.values)

  def enter(
    self,
    env: ManagerBasedRlEnv,
    pos: torch.Tensor,
    quat: torch.Tensor,
    frame: int = 0,
  ) -> None:
    """Put this skill's reference where it belongs, once, before it takes over.

    `pos` and `quat` are where the robot is *going* to be, not where it is. A clip pinned to
    where the robot stands now would slide onto the arrival and erase the error the
    hand-over is being measured on, and for the climb it would also drag the reference's own
    obstacle off the real one.
    """
    if self.actor.enter is not None:
      self.actor.enter(env, pos, quat, frame, self.values)

  ##
  # Being joined.
  ##

  def reach(self, state: np.ndarray, seconds: float) -> Reach:
    """The entry of this skill easiest to reach from `state`, and what it would demand."""
    return nearest(self._table, self.name, state, seconds)[0]

  ##
  # Driving.
  ##

  @torch.no_grad()
  def __call__(self, obs) -> torch.Tensor:
    return self.policy(obs)


@dataclass
class SkillPool:
  """Every skill the demo drives, by name."""

  skills: dict[str, Skill]
  checkpoints: dict[str, Path]
  table: EntryTable
  missing: tuple[str, ...] = field(default=())
  """Skills with a checkpoint but no rows in the entry table. Kept rather than raised: a
  skill can be driven without entries, it just cannot be handed over to."""

  ##
  # Building.
  ##

  @staticmethod
  def resolve(
    actors: tuple[Actor, ...], checkpoints: dict[str, Path] | None = None
  ) -> dict[str, Path]:
    """Name to checkpoint path, without building anything."""
    explicit = checkpoints or {}
    return {
      actor.name: find_checkpoint(
        load_rl_cfg(actor.task).experiment_name, explicit.get(actor.name)
      )
      for actor in actors
    }

  @staticmethod
  def load(
    actors: tuple[Actor, ...],
    env: ManagerBasedRlEnv,
    device: str,
    checkpoints: dict[str, Path] | None = None,
    table: EntryTable | None = None,
  ) -> SkillPool:
    """Load every actor's policy against this arena.

    `actors` does not have to include the bridge, which is appended here: forgetting it
    produces a demo that runs every skill and can never switch between them.
    """
    everyone = tuple(actors)
    if all(actor.name != BRIDGE_GROUP for actor in everyone):
      everyone += (BRIDGE,)

    found = SkillPool.resolve(everyone, checkpoints)
    entries = table if table is not None else EntryTable.load()
    known = set(entries.skills)
    skills = {
      actor.name: Skill(
        actor=actor,
        policy=Policy(actor.task, found[actor.name], env, actor.name, device),
        entries=entries.of(actor.name) if actor.name in known else (),
        table=entries,
      )
      for actor in everyone
    }
    return SkillPool(
      skills=skills,
      checkpoints=found,
      table=entries,
      missing=tuple(
        actor.name
        for actor in everyone
        if actor.name != BRIDGE_GROUP and actor.name not in known
      ),
    )

  ##
  # Reading.
  ##

  def __getitem__(self, name: str) -> Skill:
    skill = self.skills.get(name)
    if skill is None:
      raise KeyError(f"No skill called '{name}'. This pool holds {self.names}.")
    return skill

  def __contains__(self, name: str) -> bool:
    return name in self.skills

  @property
  def names(self) -> tuple[str, ...]:
    """Everything loaded, the bridge included."""
    return tuple(self.skills)

  @property
  def traversals(self) -> tuple[str, ...]:
    """Everything except the bridge. What the controller may choose between."""
    return tuple(name for name in self.skills if name != BRIDGE_GROUP)

  def lines(self) -> list[str]:
    """What was loaded and from where. Print this before a run.

    The paths, not a count. Picking the newest checkpoint under a log directory is how a
    stale run silently outranks the one that was meant, and a path on screen makes that a
    visible mistake rather than a confusing result.
    """
    width = max((len(name) for name in self.names), default=0)
    out = [f"{name:<{width}}  {self.checkpoints[name]}" for name in self.names]
    if self.missing:
      out.append(
        f"no entry table rows for {', '.join(self.missing)}: these can drive but cannot "
        f"be handed over to. Build them with selector.build"
      )
    return out
