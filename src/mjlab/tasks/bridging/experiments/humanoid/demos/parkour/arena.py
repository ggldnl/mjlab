"""The world the demo runs in: the bridge's environment, every skill's machinery merged into
it, and a course of obstacles on the floor.

`tests.stage.arena` does this for two skills. This does it for as many as the demo drives,
which is the only structural difference, plus the boxes. The merge rules are that module's
and are repeated rather than imported, because the loop they live in is written for a couple
and the couple is the thing being generalised away.

What each skill contributes, and nothing else:

    observations   copied verbatim from its own task, under a group named after it. A
                   checkpoint is bound to its term list in order, so the order comes from
                   the one place that cannot disagree with the checkpoint
    entities       whatever it puts in the scene, the ball and the crate included
    commands       frozen, because here the controller owns when a skill's goal changes
    sensors        the ones its observations read
    robot          patched, if it reads sites the bridge's own robot does not carry

Obstacles are static boxes, one entity per obstacle, sized from the course at build time.
Size is baked into the spec, so a course with different heights is a different model:
generate the course first, then build the arena for it. Poses are written at reset, which is
what lets every environment hold the same course at a different offset.

Run

Nothing here is runnable on its own. See controller.py.
"""

from __future__ import annotations

import copy
from dataclasses import fields, replace

import torch

from mjlab.asset_zoo.objects.box import get_box_cfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.bridging.experiments.humanoid.bridge.env_cfg import bridge_env_cfg
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp import BridgeCommandCfg
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.course import (
  Course,
  Obstacle,
)
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import (
  BRIDGE_GROUP,
  ROBOT,
  Actor,
  AimedCfg,
)
from mjlab.tasks.registry import load_env_cfg

OBSTACLE_PREFIX = "obstacle"
"""What an obstacle entity is called, before its index. Read back by `obstacle_names`, so
the controller can find the boxes in the scene without being handed the course."""

OBSTACLE_COLOR = (0.55, 0.58, 0.62, 1.0)
"""Concrete grey. Deliberately not the push crate's orange: one of these is scenery to be
cleared and the other is an object to be driven, and a demo running both should not make
somebody watching work out which is which."""

CONTACTS_PER_OBSTACLE = 60
"""How much to raise the contact budget per box.

The bridge's budgets were sized for a bare plane and one robot, and `tests.stage` measured
469 needed by the couple that asks the most. A constraint dropped on overflow is a contact
that silently did not happen, which looks like a robot sinking through a box rather than an
error, so this is deliberately generous."""


def obstacle_names(course: Course) -> tuple[str, ...]:
  """What each obstacle's entity is called, in course order."""
  return tuple(f"{OBSTACLE_PREFIX}_{i}" for i in range(len(course)))


def obstacle_cfg(obstacle: Obstacle):
  """One static box, sized and placed for one obstacle.

  Mass left unset is what makes it static: a box with no mass gets no freejoint, so mjlab
  wraps it as a mocap body. It collides like a wall and physics never moves it, which is
  what an obstacle is. Give it a mass and it becomes the push crate instead.
  """
  return get_box_cfg(
    half_size=obstacle.half_size,
    init_x=obstacle.at,
    mass=None,
    color=OBSTACLE_COLOR,
  )


def lay_out_course(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  names: tuple[str, ...],
  jitter: float = 0.0,
) -> None:
  """Put every obstacle where the course says, once per reset.

  Not optional decoration. A mocap body that nothing positions stacks at the world origin,
  every box on top of every other and all of them on top of the robot, however carefully the
  course was drawn.

  `jitter` moves each box along the lane by up to that many metres, drawn per environment
  and per obstacle. Off by default. Turned on it means the controller cannot be reading a
  course it was handed at build time and getting away with it, because no two environments
  agree on where anything is.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

  for name in names:
    box = env.scene[name]
    default = box.data.default_root_state
    assert default is not None
    pose = default[env_ids, 0:7].clone()
    pose[:, 0:3] += env.scene.env_origins[env_ids]
    if jitter > 0.0:
      pose[:, 0] += (torch.rand(len(env_ids), device=env.device) * 2.0 - 1.0) * jitter
    box.write_mocap_pose_to_sim(pose, env_ids=env_ids)


def course_env_cfg(
  actors: tuple[Actor, ...],
  course: Course,
  scored: str | None = None,
  jitter: float = 0.0,
) -> ManagerBasedRlEnvCfg:
  """The bridge's play environment, with every skill's machinery and the course in it.

  `scored` names one skill whose reward terms are carried, so a hand-over into it can be
  judged the way the selector judged an entry. One skill and not all of them, because a
  reward manager is built once and a course has many entering skills. Left None nothing is
  scored here, and `tests.handoff` is where a hand-over gets a number.
  """
  cfg = bridge_env_cfg(play=True)

  # The bridge command with its target supplied from outside instead of drawn from a corpus.
  # Same fields, so the policy reads what it trained on
  trained = cfg.commands["bridge"]
  assert isinstance(trained, BridgeCommandCfg)
  aimed = {f.name: getattr(trained, f.name) for f in fields(trained)}
  aimed["debug_vis"] = True
  # No corpus. Nothing here draws a window, and it lets the demo be staged before the
  # bridge's training set has been built
  aimed["dataset_path"] = None
  cfg.commands["bridge"] = AimedCfg(**aimed)
  cfg.observations = {BRIDGE_GROUP: cfg.observations["actor"]}

  # A fall is the result of the demo, not an error to recover from, and a reset mid-course
  # would move the robot out from under the phase machine
  cfg.terminations = {}
  cfg.rewards = (
    copy.deepcopy(load_env_cfg(scored, play=True).rewards or {}) if scored else {}
  )

  for actor in actors:
    if actor.name == BRIDGE_GROUP:
      continue
    task = load_env_cfg(actor.task, play=True)
    cfg.observations[actor.name] = replace(
      task.observations["actor"], enable_corruption=False
    )
    for name, entity in (task.scene.entities or {}).items():
      cfg.scene.entities.setdefault(name, entity)
    # setdefault never reaches the robot, since every skill has one, so a skill that needs
    # it modified has to say so. Patches compose in the order the actors were given
    if actor.robot is not None:
      cfg.scene.entities[ROBOT] = actor.robot(cfg.scene.entities[ROBOT])
    for name, command in task.commands.items():
      # Frozen. Here the controller owns when a skill's goal changes and `Actor.enter` is
      # how it says so. gui and debug_vis go off with it: the sliders fight the panel for
      # the same fields, and the ghosts draw a second robot that is not what is being watched
      cfg.commands.setdefault(
        name,
        replace(
          command, resampling_time_range=(1.0e9, 1.0e9), gui=False, debug_vis=False
        ),
      )
    have = {sensor.name for sensor in (cfg.scene.sensors or ())}
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + tuple(
      sensor for sensor in (task.scene.sensors or ()) if sensor.name not in have
    )
    # Deep copied because building an environment resolves names into indices inside the
    # config it is handed, and a term shared between two environments carries the first
    # one's indices into the second
    if actor.place is not None:
      cfg.events[f"place_{actor.name}"] = copy.deepcopy(actor.place)

  names = obstacle_names(course)
  for name, obstacle in zip(names, course, strict=True):
    cfg.scene.entities[name] = obstacle_cfg(obstacle)
  if names:
    cfg.events["lay_out_course"] = EventTermCfg(
      func=lay_out_course, mode="reset", params={"names": names, "jitter": jitter}
    )

  budget = 500 + CONTACTS_PER_OBSTACLE * len(names)
  cfg.sim.nconmax = budget
  cfg.sim.njmax = max(800, budget * 2)
  cfg.sim.contact_sensor_maxmatch = budget
  return cfg
