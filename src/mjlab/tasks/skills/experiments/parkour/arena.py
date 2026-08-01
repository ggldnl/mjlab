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

The corridor is a flat plane with a few boxes across it. With a plane terrain every
environment origin is the same point, so one row of obstacles serves all of them.
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
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

# Observation group names. The skills are wired to these by `build_pool`.
JUMP_OBS_GROUP = "jump_actor"
JUMP_CRITIC_GROUP = "jump_critic"
VELOCITY_OBS_GROUP = "velocity_actor"
VELOCITY_CRITIC_GROUP = "velocity_critic"

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

  width: float = 4.0
  """How far it reaches across the corridor [m]. Wide enough that going around is not
  a strategy the controller could stumble into."""


# The corridor. Spacing is what makes the three skills all necessary: the first gap is
# short enough to walk, the last is long enough that running is the only way to cover
# it in reasonable time, and every obstacle needs a jump.
CORRIDOR: tuple[Obstacle, ...] = (
  Obstacle(x=5.0),
  Obstacle(x=11.0),
  Obstacle(x=20.0),
  Obstacle(x=30.0),
)


def _add_corridor(obstacles: tuple[Obstacle, ...]):
  """Build a `spec_fn` that puts the obstacle boxes into the world."""

  def spec_fn(spec: mujoco.MjSpec) -> None:
    for i, obstacle in enumerate(obstacles):
      spec.worldbody.add_geom(
        name=f"obstacle_{i}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(obstacle.depth / 2.0, obstacle.width / 2.0, obstacle.height / 2.0),
        pos=(obstacle.x, 0.0, obstacle.height / 2.0),
        rgba=(0.55, 0.27, 0.24, 1.0),
      )

  return spec_fn


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
  cfg.commands["twist"] = velocity.commands["twist"]

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
