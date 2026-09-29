"""Build one environment containing walk, jump, climb and the course."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from functools import partial

import mujoco
import torch

from mjlab.asset_zoo.objects.box import get_box_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.bridging.bridges import BRIDGES
from mjlab.tasks.bridging.config.g1.demos.parkour.course import Course, Obstacle
from mjlab.tasks.bridging.config.g1.skills.climb import CLIMB_TASK_ID
from mjlab.tasks.bridging.config.g1.skills.climb import mdp as climb_mdp
from mjlab.tasks.bridging.config.g1.skills.jump import JUMP_TASK_ID
from mjlab.tasks.bridging.config.g1.skills.walk import WALK_TASK_ID
from mjlab.tasks.bridging.tests.stage import BRIDGE, arena
from mjlab.tasks.registry import load_env_cfg

ROBOT = "robot"
JUMP_MOTION = "jump_motion"
CLIMB_MOTION = "climb_motion"
OBSTACLE_PREFIX = "obstacle"
SIDE_OBSTACLE_PREFIX = "side_obstacle"
GOAL = "goal"


def obstacle_names(course: Course) -> tuple[str, ...]:
  return tuple(f"{OBSTACLE_PREFIX}_{index}" for index in range(len(course)))


def side_obstacle_names(course: Course) -> tuple[str, ...]:
  return tuple(
    f"{SIDE_OBSTACLE_PREFIX}_{index}" for index in range(len(course.side_obstacles))
  )


@dataclass
class Focus:
  """Obstacle currently presented to the climb policy."""

  course: Course
  names: tuple[str, ...]
  index: int = 0

  @property
  def name(self) -> str:
    return self.names[min(max(self.index, 0), len(self.names) - 1)]

  @property
  def obstacle(self) -> Obstacle:
    return self.course[min(max(self.index, 0), len(self.course) - 1)]


def active_box_pose_b(env: ManagerBasedRlEnv, focus: Focus) -> torch.Tensor:
  obstacle = focus.obstacle
  return climb_mdp.box_pose_b(env, obstacle.half_size, focus.name)


def obstacle_cfg(obstacle: Obstacle) -> EntityCfg:
  cfg = get_box_cfg(half_size=obstacle.half_size, color=obstacle.color)
  half_yaw = obstacle.yaw / 2.0
  return replace(
    cfg,
    init_state=EntityCfg.InitialStateCfg(
      pos=(obstacle.position[0], obstacle.position[1], obstacle.half_size[2]),
      rot=(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
      joint_pos={},
    ),
  )


def _side_box_spec(obstacle: Obstacle) -> mujoco.MjSpec:
  size = " ".join(str(value) for value in obstacle.half_size)
  rgba = " ".join(str(value) for value in obstacle.color)
  return mujoco.MjSpec.from_string(
    f"""<mujoco><worldbody><body name="side_box">
      <site name="side_box_visual" type="box" size="{size}" rgba="{rgba}"/>
    </body></worldbody></mujoco>"""
  )


def side_obstacle_cfg(obstacle: Obstacle) -> EntityCfg:
  """A visible, non-colliding box ignored by every policy and sensor."""
  half_yaw = obstacle.yaw / 2.0
  return EntityCfg(
    spec_fn=partial(_side_box_spec, obstacle),
    init_state=EntityCfg.InitialStateCfg(
      pos=(obstacle.position[0], obstacle.position[1], obstacle.half_size[2]),
      rot=(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
      joint_pos={},
    ),
  )


def _goal_spec(
  radius: float, color: tuple[float, float, float, float]
) -> mujoco.MjSpec:
  rgba = " ".join(str(value) for value in color)
  return mujoco.MjSpec.from_string(
    f"""<mujoco><worldbody><body name="goal">
      <site name="goal_marker" type="cylinder" size="{radius} 0.004"
            rgba="{rgba}"/>
    </body></worldbody></mujoco>"""
  )


def goal_cfg(course: Course) -> EntityCfg:
  return EntityCfg(
    spec_fn=partial(_goal_spec, course.goal.radius, course.goal.color),
    init_state=EntityCfg.InitialStateCfg(
      pos=(course.goal.position[0], course.goal.position[1], 0.004),
      joint_pos={},
    ),
  )


def place_static_entities(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  names: tuple[str, ...],
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  for name in names:
    entity = env.scene[name]
    default = entity.data.default_root_state
    assert default is not None
    pose = default[env_ids, :7].clone()
    pose[:, :3] += env.scene.env_origins[env_ids]
    entity.write_mocap_pose_to_sim(pose, env_ids=env_ids)


def _rename_command(cfg: ManagerBasedRlEnvCfg, group: str, old: str, new: str) -> None:
  cfg.commands[new] = cfg.commands.pop(old)
  for name, term in list(cfg.observations[group].terms.items()):
    if term.params.get("command_name") == old:
      cfg.observations[group].terms[name] = replace(
        term, params={**term.params, "command_name": new}
      )


def course_env_cfg(course: Course, focus: Focus) -> ManagerBasedRlEnvCfg:
  """Extend the walk to jump stage with climb and the generated course."""
  cfg = arena(BRIDGES["diffusion"].task, WALK_TASK_ID, JUMP_TASK_ID)
  cfg.observations["walk"] = cfg.observations.pop("leaving")
  cfg.observations["jump"] = cfg.observations.pop("entering")
  _rename_command(cfg, "jump", "motion", JUMP_MOTION)

  climb = load_env_cfg(CLIMB_TASK_ID, play=True)
  rate = cfg.sim.mujoco.timestep * cfg.decimation
  climb_rate = climb.sim.mujoco.timestep * climb.decimation
  if abs(climb_rate - rate) > 1.0e-9:
    raise ValueError("Climb has a different control rate")

  group = replace(copy.deepcopy(climb.observations["actor"]), enable_corruption=False)
  box_term = group.terms["box_pose"]
  group.terms["box_pose"] = replace(
    box_term, func=active_box_pose_b, params={"focus": focus}
  )
  for name, term in list(group.terms.items()):
    if term.params.get("command_name") == "motion":
      group.terms[name] = replace(
        term, params={**term.params, "command_name": CLIMB_MOTION}
      )
  cfg.observations["climb"] = group

  for name, entity in (climb.scene.entities or {}).items():
    if name not in {ROBOT, "box"}:
      cfg.scene.entities.setdefault(name, copy.deepcopy(entity))
  motion = copy.deepcopy(climb.commands["motion"])
  changes = {
    "resampling_time_range": (1.0e9, 1.0e9),
    "gui": False,
    "debug_vis": False,
  }
  if hasattr(motion, "reset_robot_to_clip"):
    changes["reset_robot_to_clip"] = False
  cfg.commands[CLIMB_MOTION] = replace(motion, **changes)

  present = {sensor.name for sensor in (cfg.scene.sensors or ())}
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + tuple(
    copy.deepcopy(sensor)
    for sensor in (climb.scene.sensors or ())
    if sensor.name not in present
  )

  names = obstacle_names(course)
  for name, obstacle in zip(names, course, strict=True):
    cfg.scene.entities[name] = obstacle_cfg(obstacle)
  side_names = side_obstacle_names(course)
  for name, obstacle in zip(side_names, course.side_obstacles, strict=True):
    cfg.scene.entities[name] = side_obstacle_cfg(obstacle)
  cfg.scene.entities[GOAL] = goal_cfg(course)
  static = names + side_names + (GOAL,)
  cfg.events["place_course"] = EventTermCfg(
    func=place_static_entities, mode="reset", params={"names": static}
  )

  contact_budget = 500 + 60 * len(names)
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, contact_budget)
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 2 * contact_budget)
  cfg.sim.contact_sensor_maxmatch = max(cfg.sim.contact_sensor_maxmatch, contact_budget)
  cfg.commands[BRIDGE].debug_vis = True
  return cfg
