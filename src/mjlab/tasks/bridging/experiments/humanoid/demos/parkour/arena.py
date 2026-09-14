"""The world the demo runs in, and a way to look at it with nothing loaded.

Two things come out of one description of the course, which is the point of the file:

    course_env_cfg   the simulation. The bridge's environment, every skill's machinery
                     merged into it, and the obstacles on the floor
    course_model     the same obstacles as a bare MuJoCo model, with a floor and no robot

Both build their geometry from `obstacle_xml`, and `obstacle_xml` invents nothing: every
position, angle and colour was sampled once in `course.generate`. So what the viewer shows
and what the robot runs into cannot drift apart.

Every obstacle collides. A box is climbed onto and crossed, a hurdle is jumped, and hitting
either is how the demo ends. Nothing here is decoration.

Which obstacle the climb is looking at
--------------------------------------

The climb reads its obstacle. Its observation carries `box_pose`, which is the box in the
robot's heading frame plus its half extents, and the term looks the box up in the scene by
name. On a course there are several, so the name has to mean whichever one the robot is
dealing with, and `Focus` is what carries that.

`focused_box_pose_b` does not reimplement the term. It swaps the scene's entry for the
obstacle in play, calls the skill's own function, and puts the entry back. A second copy of
that maths here would be a second opinion that could drift from the checkpoint, which is the
one thing the observation of a frozen policy must never do.

Run

Look at the course. No robot, no policies, no simulation.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.demos.parkour.arena
    uv run python -m ...demos.parkour.arena --viewer native
    uv run python -m ...demos.parkour.arena --seed 7 --count 8 --config my_course.yml
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Literal

import mujoco
import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges import BridgeSpec
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp import (
  BridgeCommandCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.course import (
  Color,
  Course,
  Obstacle,
  Settings,
)
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.course import (
  generate as generate_course,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.climb import mdp as climb_mdp
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import (
  BRIDGE_GROUP,
  ROBOT,
  Actor,
  aimed_cfg,
)
from mjlab.tasks.registry import load_env_cfg

OBSTACLE_PREFIX = "obstacle"
"""What an obstacle entity is called, before its index."""

CLIMB_SKILL = "climb"
"""Which skill's observation reads an obstacle out of the scene. See `Focus`."""

BOX_ENTITY = climb_mdp.BOX
"""The name the climb's own term looks its obstacle up under. Taken from the skill so the
two cannot disagree about it."""

BOX_OBSERVATION = "box_pose"
"""The term in the climb's observation group that reads it."""

PRIVATE_COMMANDS: tuple[str, ...] = ("motion",)
"""Command names that belong to one skill rather than being shared, so they get namespaced.

The jump and the climb both call their reference `motion`, and merging two skills into one
environment with `setdefault` keeps whichever was added first. Left alone the climb reads the
jump's clip: no error, no warning, a policy tracking someone else's motion. So a private
command is registered as `<skill>_<name>` and that skill's observation terms are pointed at
the new name.

The twist is deliberately not in here. The walk and the run share one velocity term on
purpose, and namespacing it would give them a set point each and break the one thing
`Skill.condition` is careful about."""


def command_name(skill: str, name: str) -> str:
  """What one skill's command is called once merged. See `PRIVATE_COMMANDS`."""
  return f"{skill}_{name}" if name in PRIVATE_COMMANDS else name


SUPPLIED: tuple[str, ...] = (BOX_ENTITY,)
"""Entity names the course provides, so a skill's own copy is dropped when its machinery is
merged in.

The climb trains against one box, which is an entity in its task. Here the obstacles come
from the course, so merging the skill's would put a stray box on the floor beside them,
placed by the skill's own reset event and belonging to no obstacle."""

CONTACTS_PER_OBSTACLE = 60
"""How much to raise the contact budget per obstacle.

The bridge's budgets were sized for a bare plane and one robot, and `tests.stage` measured
469 needed by the couple that asks the most. A climb puts two feet, two hands and sometimes
a knee on a box at once. A constraint dropped on overflow is a contact that silently did not
happen, which looks like a robot sinking through a box rather than an error."""


##
# Which obstacle a skill is looking at.
##


@dataclass
class Focus:
  """Which obstacle the skills are being told about, by entity name.

  Mutable and shared: the arena hands it to the observation term at build time and the
  controller writes to it while the demo runs. That is the whole mechanism by which the
  controller feeds a skill its obstacle, and it is one integer so it can be printed.
  """

  names: tuple[str, ...]
  index: int = 0

  @property
  def name(self) -> str:
    """The obstacle in play. Clamped, so a course that has been finished still names one:
    the observation is computed every step and there is no such thing as no answer."""
    if not self.names:
      raise SystemExit("A focus over no obstacles has nothing to point at.")
    return self.names[min(max(self.index, 0), len(self.names) - 1)]


def focused_box_pose_b(
  env: ManagerBasedRlEnv, half_size: tuple[float, float, float], focus: Focus
):
  """The climb's own obstacle term, pointed at the obstacle in play.

  The swap is the whole function. `box_pose_b` reads one name out of the scene, so the
  entry is set to whichever obstacle the controller is working on, the skill's function is
  called, and the entry is put back. Borrowed rather than copied: the observation of a
  frozen policy is bound to its checkpoint, and a second implementation here would be a
  second opinion that could drift from it.
  """
  entities = env.scene.entities
  previous = entities.get(BOX_ENTITY)
  entities[BOX_ENTITY] = env.scene[focus.name]
  try:
    return climb_mdp.box_pose_b(env, half_size)
  finally:
    if previous is None:
      entities.pop(BOX_ENTITY, None)
    else:
      entities[BOX_ENTITY] = previous


##
# The geometry. One description, two consumers.
##


def _rgba(color: Color) -> str:
  return " ".join(f"{channel:.4f}" for channel in color)


def obstacle_xml(
  obstacle: Obstacle, name: str = "obstacle", at_origin: bool = True
) -> str:
  """One obstacle as a MuJoCo body: a solid block, resting on the floor.

  The geom is centred on the body and the body carries the whole pose, which is the same
  convention the asset zoo's box uses and for the same reason. A static box has no freejoint,
  so mjlab wraps it in a mocap body and attaches this one underneath: any lift written in
  here is added to the wrapper's own pose, and the box ends up hovering half its height off
  the floor. `init_state` supplies the lift instead, once.

  `at_origin` is which of the two consumers is asking. An entity leaves the pose at zero
  because mjlab overwrites it from `init_state` and the wrapper adds it per environment. The
  bare model has no wrapper and no reset, so there the body carries the position and the turn
  itself.

  The turn is on the body either way, never on the geom. The climb reads its obstacle's
  orientation off `root_link_quat_w`, which is the body's, so a yaw baked into the geom would
  be a box the policy sees as square while the robot walks into a corner of it.
  """
  sx, sy, sz = obstacle.half_size
  geom = (
    f'<geom name="{name}_collision" type="box" size="{sx:.4f} {sy:.4f} {sz:.4f}" '
    f'condim="3" friction="1.0 0.005 0.0001" rgba="{_rgba(obstacle.color)}"/>'
  )
  if at_origin:
    return f'<body name="{name}">{geom}</body>'
  half_yaw = obstacle.yaw / 2.0
  quat = f"{math.cos(half_yaw):.6f} 0 0 {math.sin(half_yaw):.6f}"
  return (
    f'<body name="{name}" pos="{obstacle.at[0]:.4f} {obstacle.at[1]:.4f} {sz:.4f}" '
    f'quat="{quat}">{geom}</body>'
  )


def obstacle_spec(obstacle: Obstacle) -> mujoco.MjSpec:
  """One obstacle as its own spec, for use as an entity.

  No freejoint and no inertial, which is what makes it static: mjlab wraps a massless
  fixed-base body as a mocap body, so it collides like a wall and physics never moves it.
  """
  return mujoco.MjSpec.from_string(
    f"<mujoco><worldbody>{obstacle_xml(obstacle)}</worldbody></mujoco>"
  )


def obstacle_cfg(obstacle: Obstacle) -> EntityCfg:
  """One obstacle as an mjlab entity, carrying where it belongs and how far it is turned."""
  half_yaw = obstacle.yaw / 2.0
  return EntityCfg(
    spec_fn=partial(obstacle_spec, obstacle),
    init_state=EntityCfg.InitialStateCfg(
      pos=(obstacle.at[0], obstacle.at[1], obstacle.half_size[2]),
      rot=(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
      joint_pos={},
    ),
  )


def obstacle_names(course: Course) -> tuple[str, ...]:
  """What each obstacle's entity is called, in course order."""
  return tuple(f"{OBSTACLE_PREFIX}_{i}" for i in range(len(course)))


def course_model(course: Course, margin: float = 4.0) -> mujoco.MjModel:
  """The course as a bare MuJoCo model: a floor and the obstacles, and nothing else.

  No robot, no policies, no environment. What `show` serves, so the course can be looked at
  without loading anything.
  """
  bodies = "".join(
    obstacle_xml(obstacle, name=name, at_origin=False)
    for name, obstacle in zip(obstacle_names(course), course, strict=True)
  )
  half = course.length / 2.0
  xml = f"""<mujoco model="parkour">
  <visual>
    <headlight diffuse="0.65 0.65 0.65" ambient="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    <rgba haze="0.85 0.87 0.90 1"/>
    <!-- The offscreen buffer defaults to 640x480 and a renderer asking for more than it
         holds raises rather than scaling. Sized here so the course can be rendered at a
         usable resolution without every caller having to know that -->
    <global offwidth="1920" offheight="1080"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="300" height="300"
             rgb1="0.24 0.26 0.30" rgb2="0.18 0.20 0.24"/>
    <material name="grid" texture="grid" texrepeat="{max(1, int(course.length))} 8"
              texuniform="true" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="{half} 0 8" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" pos="{half} 0 0" size="{half + 2.0} {margin} 0.1"
          material="grid"/>
    {bodies}
  </worldbody>
</mujoco>
"""
  return mujoco.MjSpec.from_string(xml).compile()


##
# The simulation.
##


def lay_out_course(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  names: tuple[str, ...],
) -> None:
  """Put every obstacle where the course says, once per reset.

  Not optional decoration. A mocap body that nothing positions stacks at the world origin,
  every obstacle on top of every other and all of them on top of the robot, however
  carefully the course was drawn.

  Pose and not position: the turn is what the controller has to solve an approach for and
  what the climb reads off the body, so it travels with the placement.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

  for name in names:
    box = env.scene[name]
    default = box.data.default_root_state
    assert default is not None
    pose = default[env_ids, 0:7].clone()
    pose[:, 0:3] += env.scene.env_origins[env_ids]
    box.write_mocap_pose_to_sim(pose, env_ids=env_ids)


def course_env_cfg(
  actors: tuple[Actor, ...],
  course: Course,
  focus: Focus,
  bridge: BridgeSpec,
  scored: str | None = None,
  supplied: tuple[str, ...] = SUPPLIED,
) -> ManagerBasedRlEnvCfg:
  """The chosen bridge's play environment, with every skill's machinery and the course in it.

  `focus` is handed to the climb's obstacle observation, so the skill reads whichever
  obstacle the controller points it at. Passed in rather than made here because the
  controller writes to it and the two have to be the same object.

  `scored` names one skill whose reward terms are carried, so a hand-over into it can be
  judged the way the selector judged an entry. One skill and not all of them, because a
  reward manager is built once and a course has many entering skills.

  `bridge` says which architecture the course is built on. Its play config decides the robot,
  the terrain and the observation the bridge policy reads, so the skills are untouched by it.
  """
  assert bridge.env_cfg is not None  # resolve refuses a stub before it gets here
  cfg = bridge.env_cfg(play=True)

  # The bridge command with its target supplied from outside instead of drawn from a corpus.
  # Derived from whatever window the chosen architecture declares, so an architecture that
  # subclasses it keeps its own fields and its own command here
  trained = cfg.commands["bridge"]
  assert isinstance(trained, BridgeCommandCfg)
  cfg.commands["bridge"] = aimed_cfg(trained)
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
    group = replace(task.observations["actor"], enable_corruption=False)

    # The climb reads an obstacle by name and the course has several, so its term is pointed
    # at whichever one is in play. Everything else about the group is untouched: a
    # checkpoint is bound to its term list in order, and this replaces one term with one of
    # the same width in the same slot
    if actor.name == CLIMB_SKILL and BOX_OBSERVATION in group.terms:
      term = group.terms[BOX_OBSERVATION]
      group.terms[BOX_OBSERVATION] = replace(
        term,
        func=focused_box_pose_b,
        params={**term.params, "focus": focus},
      )

    for name, entity in (task.scene.entities or {}).items():
      if name in supplied:
        continue
      cfg.scene.entities.setdefault(name, entity)
    if actor.robot is not None:
      cfg.scene.entities[ROBOT] = actor.robot(cfg.scene.entities[ROBOT])

    renamed: dict[str, str] = {}
    for name, command in task.commands.items():
      key = command_name(actor.name, name)
      renamed[name] = key
      # Frozen. Here the controller owns when a skill's goal changes and `Actor.enter` is
      # how it says so
      cfg.commands.setdefault(
        key,
        replace(
          command, resampling_time_range=(1.0e9, 1.0e9), gui=False, debug_vis=False
        ),
      )
    # And the terms that name one. A namespaced command with an observation still asking for
    # the old name is a term reading whichever skill got there first, which is the failure
    # the namespacing exists to stop
    for term_name, term in list(group.terms.items()):
      params = getattr(term, "params", None) or {}
      wanted = params.get("command_name")
      if wanted in renamed and renamed[wanted] != wanted:
        group.terms[term_name] = replace(
          term, params={**params, "command_name": renamed[wanted]}
        )
    cfg.observations[actor.name] = group
    have = {sensor.name for sensor in (cfg.scene.sensors or ())}
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + tuple(
      sensor for sensor in (task.scene.sensors or ()) if sensor.name not in have
    )
    if actor.place is not None and actor.name not in supplied:
      cfg.events[f"place_{actor.name}"] = copy.deepcopy(actor.place)

  names = obstacle_names(course)
  for name, obstacle in zip(names, course, strict=True):
    cfg.scene.entities[name] = obstacle_cfg(obstacle)
  if names:
    cfg.events["lay_out_course"] = EventTermCfg(
      func=lay_out_course, mode="reset", params={"names": names}
    )

  budget = 500 + CONTACTS_PER_OBSTACLE * len(names)
  cfg.sim.nconmax = budget
  cfg.sim.njmax = max(800, budget * 2)
  cfg.sim.contact_sensor_maxmatch = budget
  return cfg


##
# Looking at it.
##


def show(course: Course, viewer: Literal["viser", "native"], port: int = 8080) -> None:
  """Serve the course with no robot and no policies.

  No physics. Nothing here moves, so the model is compiled, forward kinematics is run once
  and the scene is handed over.
  """
  model = course_model(course)
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)

  if viewer == "native":
    # Bound under another name rather than as `import mujoco.viewer`, which would make
    # `mujoco` a local of this function and shadow the module imported at the top
    from mujoco import viewer as mj_viewer

    print("[parkour] opening the native viewer, close the window to stop")
    with mj_viewer.launch_passive(
      model, data, show_left_ui=False, show_right_ui=False
    ) as handle:
      handle.cam.lookat[:] = (course.length / 3.0, 0.0, 0.0)
      handle.cam.distance = course.length / 2.0
      handle.cam.azimuth, handle.cam.elevation = 120.0, -25.0
      while handle.is_running():
        handle.sync()
    return

  import time

  import viser
  from mjviser import ViserMujocoScene

  server = viser.ViserServer(port=port)
  scene = ViserMujocoScene(server, model, num_envs=1)
  scene.camera_tracking_enabled = False
  scene.update_from_mjdata(data)
  # On the scene rather than on the camera. The model is z up and the viewer is not by
  # default, so without this an orbit tips the course on its side
  server.scene.set_up_direction("+z")

  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    """Open on the first two obstacles rather than on the whole course.

    Reset View frames the scene, and the scene is a forty metre floor: it fits the plane and
    leaves the obstacles as specks on the horizon.
    """
    client.camera.position = (-2.0, -9.0, 4.0)
    client.camera.look_at = (9.0, 0.0, 0.0)

  server.gui.add_markdown("\n".join(course.lines()))
  print(f"[parkour] serving on http://localhost:{port}")
  while True:
    time.sleep(0.1)


@dataclass(frozen=True)
class ViewCfg:
  config: Path | None = None
  seed: int | None = None
  count: int | None = None
  viewer: Literal["viser", "native"] = "viser"
  port: int = 8080


def main(cfg: ViewCfg) -> None:
  course = generate_course(Settings.load(cfg.config), seed=cfg.seed, count=cfg.count)
  for line in course.lines():
    print(line)
  show(course, cfg.viewer, cfg.port)


if __name__ == "__main__":
  import tyro

  import mjlab

  main(tyro.cli(ViewCfg, config=mjlab.TYRO_FLAGS))
