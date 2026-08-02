"""The one environment all three parkour skills act in.

Every skill in a pool is rolled in a single shared arena, so the arena has to satisfy
all of them at once. Walk and run are velocity-tracking policies and read a commanded
twist; jump is a motion-tracking policy and reads a reference clip. Neither
observation is a superset of the other, and a policy handed the wrong one produces
nonsense rather than an error.

The arena therefore carries every skill's observation side by side, under its own
name, plus the command terms that feed them:

    jump_actor / jump_critic          what the jump policy was trained on
    velocity_actor / velocity_critic  what walk and run were trained on (identical
                                      environments, so one pair serves both)
    actor / critic                    proprioception only -- the bridge's view

Each frozen policy is told which group to read (see skills.py), so nothing about the
individual tasks has to change and their checkpoints stay valid. The plain `actor`
group is what the bridging machinery works on, and it is deliberately the narrow one:
a bridge compared on `jump_actor` would be compared partly on the reference clip, and
a bridge compared on `velocity_actor` partly on a commanded velocity, neither of which
describes the robot. See view.py for why that matters.

The corridor is a flat plane with a lane painted down it and a few boxes across the
lane. With a plane terrain every environment origin is the same point, so one row of
obstacles serves all of them.

Both commands are also constrained here, which is the difference between an arena and
the task environments the skills were trained in. A training environment randomizes
its command on purpose: the velocity task resamples a fresh twist every few seconds,
turns a tenth of its environments into standing ones and points a third of them at a
random heading, and the jump task picks a random clip and a random stretch at every
reset. That is what makes a goal-conditioned skill general, and it is exactly wrong
here -- a corridor is one direction at one of two speeds, and a jump over a known box
is one distance. Left alone, the composition inherits the randomization and the robot
shuffles in place and turns for no reason, which is a randomized command being tracked
faithfully rather than a broken policy.

So the twist is pinned to +x with a heading hold and never resampled (the controller
writes the speed, see controller.py), and the jump's goal is set from the obstacle at
the moment control reaches it. Neither command is exposed to the viewer: a slider that
retargets the jump belongs to the skill's own play script, not to an experiment whose
whole subject is what the controller asked for.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.skills.experiments.parkour.jump.jump_env_cfg import g1_jump_env_cfg
from mjlab.tasks.skills.experiments.parkour.jump.mdp import JumpCommandCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

# Observation group names. The skills are wired to these by `build_pool`.
JUMP_OBS_GROUP = "jump_actor"
JUMP_CRITIC_GROUP = "jump_critic"
VELOCITY_OBS_GROUP = "velocity_actor"
VELOCITY_CRITIC_GROUP = "velocity_critic"

# Command term names, as the arena registers them. The controller writes to both.
TWIST_COMMAND = "twist"
JUMP_COMMAND = "motion"

# How fast the corridor is covered [m/s]. Both are what the current checkpoints
# actually hold, not what their tasks are aimed at: swept over a bare plane, either
# policy tracks a command up to 2.0 m/s without a single termination, starts losing
# episodes at 2.5 and collapses at 3.0. The run task's curriculum climbs to 4.5 over
# 15k iterations (see run/run_env_cfg.py) and its checkpoint is a fraction of the way
# in, so RUN_SPEED is worth raising as that run continues -- and worth re-measuring
# rather than assuming, since a run skill that falls on the straight never reaches
# the hand-off this experiment is about.
WALK_SPEED = 1.0
RUN_SPEED = 2.0

# How wide the corridor is [m]. Narrow enough to read as a lane rather than as a
# field with things standing in it, and wide enough that a gait's lateral drift over a
# stride does not leave it. The obstacles span exactly this, so the painted edge is
# also the edge of what has to be jumped.
CORRIDOR_WIDTH = 1.6

# How far the painted corridor runs behind the start and past the last obstacle [m].
# Purely cosmetic, but the near end is not arbitrary: every episode begins with the
# robot on the clip's first frame at the origin, so the lane has to start behind that
# or the robot spawns off its own corridor.
CORRIDOR_MARGIN = 4.0

# Two shades of blue: the lane recedes, the obstacles come forward. The point of
# splitting them is that the thing to be jumped should be the thing the eye lands on.
PATH_COLOR = (0.13, 0.22, 0.42, 1.0)
EDGE_COLOR = (0.45, 0.68, 0.95, 1.0)
OBSTACLE_COLOR = (0.24, 0.51, 0.85, 1.0)

# What the bridge sees: the robot, and nothing about anyone's task.
PROPRIO_TERMS = (
  "base_lin_vel",
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
)


@dataclass(frozen=True)
class Obstacle:
  """A box across the corridor, low enough to jump and long enough to matter."""

  x: float
  """Centre of the box along the corridor [m]."""

  height: float = 0.15
  """How tall it stands [m]. The converted clips lift a foot 0.2 to 0.4 m at the apex,
  so this leaves clearance for a jump and none at all for a walk."""

  depth: float = 0.35
  """How deep it is along the corridor [m]. Every clip covers this several times over;
  the obstacle is a thing to clear, not a gap to span."""

  width: float = CORRIDOR_WIDTH
  """How far it reaches across the corridor [m]. The corridor's full width, so going
  around is not a strategy the controller could stumble into and the box reads as
  part of the lane rather than as an object placed on it."""


# The corridor. Spacing is what makes the three skills all necessary: the first gap is
# short enough to walk, the last is long enough that running is the only way to cover
# it in reasonable time, and every obstacle needs a jump.
CORRIDOR: tuple[Obstacle, ...] = (
  Obstacle(x=5.0),
  Obstacle(x=11.0),
  Obstacle(x=20.0),
  Obstacle(x=30.0),
)


# How thick the painted lane and its edge lines are [m]. Thin enough to be paint: the
# robot walks on the terrain plane at z=0 and never touches these, and the obstacles
# still sit on the plane rather than on top of the paint, so nothing about the physics
# or the clearance a jump needs changes when the corridor is drawn.
_PAINT_THICKNESS = 0.002

# How wide the two lines marking the lane's edge are [m].
_EDGE_WIDTH = 0.08


def _add_paint(
  spec: mujoco.MjSpec,
  name: str,
  size: tuple[float, float],
  pos: tuple[float, float],
  color: tuple[float, float, float, float],
  layer: int = 0,
) -> None:
  """Lay a flat, collisionless rectangle on the ground.

  `contype` and `conaffinity` are both zero, which is what makes this decoration
  rather than terrain: MuJoCo will not generate contacts for it, so it costs nothing
  from the contact budget and cannot be stood on. `layer` stacks coincident paint by
  a fraction of a millimetre so the edge lines draw over the lane instead of
  z-fighting with it.
  """
  half_thickness = _PAINT_THICKNESS / 2.0
  spec.worldbody.add_geom(
    name=name,
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(size[0] / 2.0, size[1] / 2.0, half_thickness),
    pos=(pos[0], pos[1], half_thickness + layer * _PAINT_THICKNESS),
    rgba=color,
    contype=0,
    conaffinity=0,
  )


def _add_corridor(obstacles: tuple[Obstacle, ...]):
  """
  Build a `spec_fn` that paints the corridor and puts the obstacle boxes in it.
  """

  def spec_fn(spec: mujoco.MjSpec) -> None:
    start = min(0.0, min(o.x for o in obstacles)) - CORRIDOR_MARGIN
    end = max(o.x for o in obstacles) + CORRIDOR_MARGIN
    length = end - start
    centre = (start + end) / 2.0

    _add_paint(
      spec,
      "corridor_path",
      size=(length, CORRIDOR_WIDTH),
      pos=(centre, 0.0),
      color=PATH_COLOR,
    )
    # The edges, drawn just inside the lane so the bright line reads as its border.
    for side, offset in (("left", 1.0), ("right", -1.0)):

      """
      _add_paint(
        spec,
        f"corridor_edge_{side}",
        size=(length, _EDGE_WIDTH),
        pos=(centre, offset * (CORRIDOR_WIDTH - _EDGE_WIDTH) / 2.0),
        color=EDGE_COLOR,
        layer=1,
      )
      """

    for i, obstacle in enumerate(obstacles):
      spec.worldbody.add_geom(
        name=f"obstacle_{i}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(obstacle.depth / 2.0, obstacle.width / 2.0, obstacle.height / 2.0),
        pos=(obstacle.x, 0.0, obstacle.height / 2.0),
        rgba=OBSTACLE_COLOR,
      )

  return spec_fn


def _constrain_twist(cmd: UniformVelocityCommandCfg) -> None:
  """Pin the velocity command to the corridor: forward, at one commanded speed.

  Everything the velocity task randomizes is turned off. The resampling clock is
  pushed out of reach so the command the controller writes is the command that
  stays; standing, world-frame and forward-only environments are all removed, since
  with one command for every environment there is nothing left for those fractions to
  select between.

  Heading control is the one thing kept on, and turned up rather than off: the target
  heading is +x for every environment, so the yaw rate becomes a corrective term that
  steers the robot back down the corridor instead of a commanded turn. Without it
  nothing removes the drift a gait accumulates over thirty metres, and a robot that
  leaves the corridor sideways never meets an obstacle at all.
  """
  cmd.resampling_time_range = (1.0e9, 1.0e9)
  cmd.rel_standing_envs = 0.0
  cmd.rel_world_envs = 0.0
  cmd.rel_forward_envs = 0.0
  cmd.init_velocity_prob = 0.0
  cmd.heading_command = True
  cmd.rel_heading_envs = 1.0
  cmd.heading_control_stiffness = 1.0
  cmd.ranges = UniformVelocityCommandCfg.Ranges(
    # The controller overwrites the forward speed every step; this is what a reset
    # leaves behind until it does.
    lin_vel_x=(WALK_SPEED, WALK_SPEED),
    lin_vel_y=(0.0, 0.0),
    # Not a command but a bound on the heading correction above.
    ang_vel_z=(-0.5, 0.5),
    heading=(0.0, 0.0),
  )
  # No joystick: the corridor is driven by the controller, and a slider writing the
  # same field would silently take it over.
  cmd.gui = False


def _rename_group(
  cfg: ManagerBasedRlEnvCfg, source: str, destination: str
) -> ObservationGroupCfg:
  group = cfg.observations.pop(source)
  cfg.observations[destination] = group
  return group


def parkour_arena_env_cfg(
  obstacles: tuple[Obstacle, ...] | None = CORRIDOR,
) -> ManagerBasedRlEnvCfg:
  """The shared arena: G1, a flat corridor, and every skill's observation.

  Args:
    obstacles: Boxes to put across the corridor. None leaves the plane bare, which
      is what bridge training wants: it practises hand-offs from harvested states
      and never navigates the corridor.
  """
  # Built on the jump environment because that is the one carrying machinery the
  # arena cannot reconstruct: the motion command, its library of clips, and the
  # reference-state initialization that puts the robot somewhere sensible on reset.
  cfg = g1_jump_env_cfg(play=True)
  velocity = unitree_g1_flat_env_cfg(play=True)

  # The jump policy's observation, under its own name.
  _rename_group(cfg, "actor", JUMP_OBS_GROUP)
  _rename_group(cfg, "critic", JUMP_CRITIC_GROUP)

  # Walk and run share an environment, so they share an observation.
  cfg.observations[VELOCITY_OBS_GROUP] = velocity.observations["actor"]
  cfg.observations[VELOCITY_CRITIC_GROUP] = velocity.observations["critic"]

  twist = velocity.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  _constrain_twist(twist)
  cfg.commands[TWIST_COMMAND] = twist

  # The jump's own goal machinery, likewise taken out of the user's hands: the
  # corridor decides how far each jump has to go (see controller.py). The ghost goes
  # with it. It is the right thing to watch when a tracking policy is being judged
  # against its clip, and noise here, where the clip spends most of the corridor idle
  # at the robot's feet and nothing is being tracked.
  motion = cfg.commands[JUMP_COMMAND]
  assert isinstance(motion, JumpCommandCfg)
  motion.debug_vis = False
  motion.gui = False

  # Sensors the velocity observation reads and the jump environment does not have.
  # Its critic scans the ground under each foot; the two environments' contact and
  # self-collision sensors are already the same, so only the new ones are added.
  have = {sensor.name for sensor in (cfg.scene.sensors or ())}
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + tuple(
    sensor for sensor in (velocity.scene.sensors or ()) if sensor.name not in have
  )

  # And the bridge's view: proprioception, nothing else. Taken from the jump group
  # because it is the one already resolved against this scene's sensors.
  jump_terms = cfg.observations[JUMP_OBS_GROUP].terms
  proprio: dict[str, ObservationTermCfg] = {
    name: jump_terms[name] for name in PROPRIO_TERMS
  }
  cfg.observations["actor"] = ObservationGroupCfg(
    terms=dict(proprio), concatenate_terms=True, enable_corruption=False
  )
  cfg.observations["critic"] = ObservationGroupCfg(
    terms=dict(proprio), concatenate_terms=True, enable_corruption=False
  )

  # Nothing here is being trained, so the reward set only has to exist. The bridging
  # architectures bring their own.
  cfg.rewards = {"alive": RewardTermCfg(func=envs_mdp.is_alive, weight=0.0)}

  # The jump task's terminations all measure the robot against the reference clip,
  # which in the corridor is idle most of the time and would end episodes for no
  # reason. What matters here is falling over.
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    ),
  }

  # The jump task's curriculum tightens the reward weights and termination thresholds
  # that were just replaced, so it has nothing left to point at. Nothing is being
  # trained here anyway.
  cfg.curriculum = {}

  cfg.episode_length_s = 60.0

  if obstacles:
    cfg.scene.spec_fn = _add_corridor(obstacles)
    # Boxes are extra contact pairs; the jump task's budget assumed a bare plane.
    cfg.sim.nconmax = 80
    cfg.sim.njmax = 400

  return cfg
