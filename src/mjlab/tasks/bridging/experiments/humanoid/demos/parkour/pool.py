"""Every policy the demo drives, loaded once from its own checkpoint.

    pool = SkillPool.load(ACTORS, env, device)
    pool["run"](obs)          one frozen skill, reading its own observation group
    pool.bridge(obs)          the bridge, loaded the same way as everything else
    pool.entries("vault")     where that skill can be joined, off the selector table

One place that turns a name into a policy, because a demo has as many checkpoints as it has
skills and every one of them is a separate way to load the wrong file. `find_checkpoint`
picks the newest under an experiment's log directory, which is convenient and has gone wrong
before, so `lines` prints what was resolved and the caller is expected to show it.

The bridge is in the pool rather than beside it. It is a policy loaded from a checkpoint
reading an observation group, which is what every other member is, and the one thing that
would justify special-casing it is that the controller calls it between skills rather than
instead of them. That is the controller's business.

Two steps, on purpose:

    resolve   names to checkpoint paths. No simulation, so a missing file is reported
              before a minute of arena build rather than after it
    load      those paths into policies. Needs the env, because a policy binds to the
              observation group it will read

Run

Nothing here is runnable on its own. See controller.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.bridge import BRIDGE_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.selector import Entry, EntryTable
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import (
  BRIDGE_GROUP,
  Actor,
  Policy,
  find_checkpoint,
)
from mjlab.tasks.registry import load_rl_cfg

BRIDGE = Actor(BRIDGE_GROUP, BRIDGE_TASK_ID)
"""The bridge as an actor, so it loads through the same path as a skill.

It declares nothing else: no controls, because the controller aims it rather than telling it
anything; no `enter`, because it has no reference to place; no `place`, because it puts
nothing on the floor."""


@dataclass
class SkillPool:
  """The loaded policies, their declarations, and where each skill can be joined."""

  actors: dict[str, Actor]
  policies: dict[str, Policy]
  checkpoints: dict[str, Path]
  table: EntryTable
  missing: tuple[str, ...] = field(default=())
  """Skills with a checkpoint but no rows in the entry table.

  Kept rather than raised. A skill can be driven without entries, it just cannot be handed
  over to, and which of those the demo needs is the controller's call: the locomotion skill
  the robot starts in is entered from a reset, never from a bridge."""

  ##
  # Building.
  ##

  @staticmethod
  def resolve(
    actors: tuple[Actor, ...], checkpoints: dict[str, Path] | None = None
  ) -> dict[str, Path]:
    """Name to checkpoint path, without building anything.

    Called first and on its own, so a demo that is missing one of eight checkpoints says so
    in a second rather than after the arena is up. An explicit path in `checkpoints` wins
    over the newest file under the experiment's log directory.
    """
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

    `actors` is whatever the demo drives and does not have to include the bridge, which is
    appended here: forgetting it produces a demo that runs every skill and can never switch
    between them, and that is not a configuration anybody wants.
    """
    everyone = tuple(actors)
    if all(actor.name != BRIDGE_GROUP for actor in everyone):
      everyone += (BRIDGE,)

    found = SkillPool.resolve(everyone, checkpoints)
    entries = table if table is not None else EntryTable.load()
    known = set(entries.skills)
    return SkillPool(
      actors={actor.name: actor for actor in everyone},
      policies={
        actor.name: Policy(actor.task, found[actor.name], env, actor.name, device)
        for actor in everyone
      },
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

  def __getitem__(self, name: str) -> Policy:
    policy = self.policies.get(name)
    if policy is None:
      raise KeyError(f"No policy called '{name}'. This pool holds {self.names}.")
    return policy

  def __contains__(self, name: str) -> bool:
    return name in self.policies

  @property
  def names(self) -> tuple[str, ...]:
    """Everything loaded, the bridge included."""
    return tuple(self.policies)

  @property
  def skills(self) -> tuple[str, ...]:
    """Everything loaded except the bridge. What the controller may choose between."""
    return tuple(name for name in self.policies if name != BRIDGE_GROUP)

  @property
  def bridge(self) -> Policy:
    return self.policies[BRIDGE_GROUP]

  def actor(self, name: str) -> Actor:
    """One skill's declaration: what it can be told, and what it needs placing."""
    actor = self.actors.get(name)
    if actor is None:
      raise KeyError(f"No actor called '{name}'. This pool holds {self.names}.")
    return actor

  def entries(self, name: str) -> tuple[Entry, ...]:
    """Where that skill can be joined, best first. Raises if it was never profiled."""
    return self.table.of(name)

  def lines(self) -> list[str]:
    """What was loaded and from where. Print this before a run.

    The paths, not a count. Picking the newest checkpoint under a log directory is how a
    stale run left in logs/ silently outranks the one that was meant, and a path on screen
    makes that a visible mistake rather than a confusing result.
    """
    width = max((len(name) for name in self.names), default=0)
    out = [f"{name:<{width}}  {self.checkpoints[name]}" for name in self.names]
    if self.missing:
      out.append(
        f"no entry table rows for {', '.join(self.missing)}: these can drive but "
        f"cannot be handed over to. Build them with selector.build"
      )
    return out
