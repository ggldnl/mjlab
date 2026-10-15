"""One physical G1 scene with independent locomotion, jump and kick observations."""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from functools import partial

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.objects.ball import BALL_RADIUS
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import (
  EventTermCfg,
  RecomputeLevel,
  requires_model_fields,
)
from mjlab.tasks.bridging.config.g1.demos.soccer.config import Scene, Settings
from mjlab.tasks.bridging.config.g1.skills.jump import JUMP_TASK_ID
from mjlab.tasks.bridging.config.g1.skills.kick import KICK_TASK_ID
from mjlab.tasks.bridging.config.g1.skills.kick import mdp as kick_mdp
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz
from mjlab.viewer.viewer_config import ViewerConfig

JUMP = "jump_motion"
KICK = "kick_motion"


def rotation(degrees: tuple[float, float, float]) -> tuple[float, float, float, float]:
  angles = torch.tensor(degrees).deg2rad()
  return tuple(quat_from_euler_xyz(*angles.unbind()).tolist())


def fallen_spec(joints: dict[str, float]) -> mujoco.MjSpec:
  """Bake a G1 joint pose into a fixed visual model without collisions."""
  robot = get_g1_robot_cfg()
  robot.init_state = EntityCfg.InitialStateCfg(joint_pos=joints)
  spec = robot.build().spec
  model = spec.compile()
  data = mujoco.MjData(model)
  data.qpos[:] = model.key("init_state").qpos
  mujoco.mj_forward(model, data)
  for body in spec.bodies[1:]:
    index = model.body(body.name).id
    parent = model.body_parentid[index]
    inverse = data.xquat[parent].copy()
    inverse[1:] *= -1
    position, quaternion = np.zeros(3), np.zeros(4)
    mujoco.mju_rotVecQuat(position, data.xpos[index] - data.xpos[parent], inverse)
    mujoco.mju_mulQuat(quaternion, inverse, data.xquat[index])
    body.pos, body.quat = position, quaternion
  for elements in (
    tuple(spec.actuators),
    tuple(spec.sensors),
    tuple(spec.keys),
    tuple(spec.joints),
    tuple(spec.pairs),
  ):
    for element in elements:
      spec.delete(element)
  for geom in spec.geoms:
    geom.contype = geom.conaffinity = 0
  return spec


def field_spec(scene: Scene) -> mujoco.MjSpec:
  """Paint a pitch ending at the goal, with a physical frame and visual net."""
  spec = mujoco.MjSpec()
  field = spec.worldbody.add_body(name="field")
  pitch = field.add_body(
    name="pitch",
    pos=(*scene.goal_position, 0.0),
    quat=rotation((0.0, 0.0, scene.goal_heading_degrees)),
  )
  length, width = scene.pitch_length, scene.pitch_width
  half, height = scene.goal_width / 2, scene.goal_height
  white = (0.94, 0.96, 0.9, 1.0)

  def box(name, position, size, color):
    pitch.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=position,
      size=size,
      rgba=color,
      contype=0,
      conaffinity=0,
    )

  def line(name, start, finish, radius=0.025, physical=False):
    pitch.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      size=(radius, 0, 0),
      fromto=(*start, *finish),
      rgba=white,
      contype=int(physical),
      conaffinity=int(physical),
    )

  for index in range(12):
    box(
      f"turf_{index}",
      (-length + (index + 0.5) * length / 12, 0, -0.006),
      (length / 24, width / 2, 0.01),
      (0.12, 0.34 + 0.035 * (index % 2), 0.16, 1),
    )
  z = 0.008
  for name, x in (("near_end", -length), ("goal_line", 0), ("halfway", -length / 2)):
    box(name, (x, 0, z), (0.025, width / 2, 0.002), white)
  for side in (-1, 1):
    box(
      f"touchline_{side}",
      (-length / 2, side * width / 2, z),
      (length / 2, 0.025, 0.002),
      white,
    )
  radius = min(width * 0.15, length * 0.1)
  for index in range(64):
    angles = (2 * math.pi * index / 64, 2 * math.pi * (index + 1) / 64)
    points = [
      (-length / 2 + radius * math.cos(a), radius * math.sin(a), z) for a in angles
    ]
    line(f"center_circle_{index}", *points, radius=0.018)
  pitch.add_geom(
    name="center_spot",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    pos=(-length / 2, 0, z),
    size=(0.065, 0.002, 0),
    rgba=white,
    contype=0,
    conaffinity=0,
  )
  for end in (0, 1):
    baseline, direction = (0, -1) if end == 0 else (-length, 1)
    for name, depth, span in (
      ("penalty", min(length * 0.18, 2.0), min(width * 0.35, half + 1.0)),
      ("goal_area", min(length * 0.07, 0.8), min(width * 0.25, half + 0.4)),
    ):
      box(
        f"{name}_{end}_front",
        (baseline + direction * depth, 0, z),
        (0.025, span, 0.002),
        white,
      )
      for side in (-1, 1):
        box(
          f"{name}_{end}_{side}",
          (baseline + direction * depth / 2, side * span, z),
          (depth / 2, 0.025, 0.002),
          white,
        )

  depth = 0.8
  for side in (-half, half):
    line(
      f"post_{side}", (0, side, 0.035), (0, side, height), radius=0.035, physical=True
    )
    line(f"rear_post_{side}", (depth, side, 0.035), (depth, side, height), radius=0.025)
    for h in (0.035, height):
      line(f"support_{side}_{h}", (0, side, h), (depth, side, h), radius=0.025)
  line("crossbar", (0, -half, height), (0, half, height), radius=0.035, physical=True)
  line("rear_crossbar", (depth, -half, height), (depth, half, height), radius=0.025)
  columns, rows = (
    max(1, math.ceil(scene.goal_width / 0.12)),
    max(1, math.ceil(height / 0.12)),
  )
  for index in range(columns + 1):
    y = -half + scene.goal_width * index / columns
    line(
      f"net_back_vertical_{index}", (depth, y, 0.035), (depth, y, height), radius=0.003
    )
    line(f"net_roof_{index}", (0, y, height), (depth, y, height), radius=0.003)
  for index in range(rows + 1):
    h = 0.035 + (height - 0.035) * index / rows
    line(
      f"net_back_horizontal_{index}", (depth, -half, h), (depth, half, h), radius=0.003
    )
    for side in (-half, half):
      line(f"net_side_{side}_{index}", (0, side, h), (depth, side, h), radius=0.003)
  for index in range(1, 7):
    x = depth * index / 7
    line(f"net_roof_cross_{index}", (x, -half, height), (x, half, height), radius=0.003)
    for side in (-half, half):
      line(
        f"net_side_vertical_{side}_{index}",
        (x, side, 0.035),
        (x, side, height),
        radius=0.003,
      )
  return spec


def reset_robot(env: ManagerBasedRlEnv, env_ids: torch.Tensor, scene: Scene) -> None:
  robot = env.scene["robot"]
  root = robot.data.default_root_state[env_ids].clone()
  root[:, :2] = torch.tensor(scene.robot_position, device=env.device)
  root[:, :3] += env.scene.env_origins[env_ids]
  root[:, 3:7] = torch.tensor(
    rotation((0.0, 0.0, scene.robot_heading_degrees)), device=env.device
  )
  root[:, 7:] = 0
  robot.write_root_state_to_sim(root, env_ids)
  robot.write_joint_state_to_sim(
    robot.data.default_joint_pos[env_ids],
    robot.data.default_joint_vel[env_ids],
    env_ids=env_ids,
  )


def randomize_fallen(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor, scene: Scene
) -> None:
  fallen = env.scene["fallen_robot"]
  pose = fallen.data.default_root_state[env_ids, :7].clone()
  bounds = pose.new_tensor(scene.fallen_position_jitter)
  pose[:, :2] += (torch.rand(len(env_ids), 2, device=env.device) * 2 - 1) * bounds
  yaw = (
    torch.rand(len(env_ids), device=env.device) * 2 - 1
  ) * scene.fallen_yaw_jitter_degrees
  rotations = (
    pose.new_tensor(scene.fallen_rotation_degrees).expand(len(env_ids), -1).clone()
  )
  rotations[:, 2] += yaw
  pose[:, 3:7] = quat_from_euler_xyz(*rotations.deg2rad().unbind(-1))
  pose[:, :3] += env.scene.env_origins[env_ids]
  fallen.write_mocap_pose_to_sim(pose, env_ids=env_ids)
  for env_id in env_ids.tolist():
    joints = dict(scene.fallen_joints)
    for name, bound in scene.fallen_joint_jitter.items():
      joints[name] = joints.get(name, 0.0) + float(
        torch.empty((), device=env.device).uniform_(-bound, bound)
      )
    spec = fallen_spec(joints)
    bodies = list(spec.bodies)[1:]
    ids = [env.sim.mj_model.body(f"fallen_robot/{body.name}").id for body in bodies]
    env.sim.model.body_pos[env_id, ids] = pose.new_tensor(
      np.array([body.pos for body in bodies])
    )
    env.sim.model.body_quat[env_id, ids] = pose.new_tensor(
      np.array([body.quat for body in bodies])
    )


@requires_model_fields("body_pos", "body_quat", recompute=RecomputeLevel.set_const_0)
def reset_match(env: ManagerBasedRlEnv, env_ids: torch.Tensor, scene: Scene) -> None:
  reset_robot(env, env_ids, scene)
  ball = env.scene["ball"]
  root = ball.data.default_root_state[env_ids].clone()
  root[:, :2] = torch.tensor(scene.ball_position, device=env.device)
  root[:, 2] = BALL_RADIUS
  root[:, :3] += env.scene.env_origins[env_ids]
  root[:, 3:7] = root.new_tensor([1.0, 0.0, 0.0, 0.0])
  root[:, 7:] = 0
  ball.write_root_state_to_sim(root, env_ids)
  for name in ("fallen_robot", "field"):
    entity = env.scene[name]
    pose = entity.data.default_root_state[env_ids, :7].clone()
    pose[:, :3] += env.scene.env_origins[env_ids]
    entity.write_mocap_pose_to_sim(pose, env_ids=env_ids)
  randomize_fallen(env, env_ids, scene)


def external_twist(cfg: UniformVelocityCommandCfg) -> UniformVelocityCommandCfg:
  return replace(
    cfg,
    resampling_time_range=(1e9, 1e9),
    gui=False,
    debug_vis=False,
    rel_heading_envs=0.0,
    rel_standing_envs=0.0,
    rel_world_envs=0.0,
    rel_forward_envs=0.0,
    init_velocity_prob=0.0,
  )


def soccer_env_cfg(settings: Settings) -> ManagerBasedRlEnvCfg:
  cfg = load_env_cfg(settings.policies.locomotion_task, play=True)
  cfg.scene.num_envs = 1
  cfg.observations = {
    "walk": replace(copy.deepcopy(cfg.observations["actor"]), enable_corruption=False)
  }
  twist = cfg.commands["twist"]
  if not isinstance(twist, UniformVelocityCommandCfg):
    raise ValueError("Locomotion must accept a twist command")
  cfg.commands = {"twist": external_twist(twist)}
  for skill, task, command_name in (
    ("jump", JUMP_TASK_ID, JUMP),
    ("kick", KICK_TASK_ID, KICK),
  ):
    source = load_env_cfg(task, play=True)
    if (
      cfg.actions != source.actions
      or cfg.decimation != source.decimation
      or cfg.sim.mujoco.timestep != source.sim.mujoco.timestep
    ):
      raise ValueError(f"{skill} uses different actions or control frequency")
    if (
      cfg.scene.entities["robot"].init_state.joint_pos
      != source.scene.entities["robot"].init_state.joint_pos
    ):
      raise ValueError(f"{skill} uses a different joint rest pose")
    group = replace(
      copy.deepcopy(source.observations["actor"]), enable_corruption=False
    )
    for name, term in group.terms.items():
      if term.params.get("command_name", "motion") == "motion" and (
        "command_name" in term.params or term.func is kick_mdp.ball_contact
      ):
        group.terms[name] = replace(
          term, params={**term.params, "command_name": command_name}
        )
    cfg.observations[skill] = group
    cfg.commands[command_name] = replace(
      copy.deepcopy(source.commands["motion"]),
      resampling_time_range=(1e9, 1e9),
      gui=False,
      debug_vis=False,
      reset_robot_to_clip=False,
    )
    for name, entity in source.scene.entities.items():
      if name != "robot":
        cfg.scene.entities[name] = copy.deepcopy(entity)
    present = {sensor.name for sensor in cfg.scene.sensors or ()}
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + tuple(
      copy.deepcopy(sensor)
      for sensor in source.scene.sensors or ()
      if sensor.name not in present
    )
  scene = settings.scene
  cfg.viewer = replace(
    cfg.viewer,
    origin_type=ViewerConfig.OriginType.WORLD,
    lookat=(
      (scene.robot_position[0] + scene.goal_position[0]) / 2,
      (scene.robot_position[1] + scene.goal_position[1]) / 2,
      0.4,
    ),
    distance=max(scene.pitch_length, scene.pitch_width),
    elevation=-30.0,
    azimuth=scene.goal_heading_degrees + 110.0,
    mesh_lod=False,
  )
  cfg.scene.entities["fallen_robot"] = EntityCfg(
    spec_fn=partial(fallen_spec, scene.fallen_joints),
    init_state=EntityCfg.InitialStateCfg(
      pos=scene.fallen_position,
      rot=rotation(scene.fallen_rotation_degrees),
      joint_pos={},
    ),
  )
  cfg.scene.entities["field"] = EntityCfg(
    spec_fn=partial(field_spec, scene),
    init_state=EntityCfg.InitialStateCfg(joint_pos={}),
  )
  cfg.events = {
    "reset_match": EventTermCfg(func=reset_match, mode="reset", params={"scene": scene})
  }
  cfg.rewards, cfg.metrics, cfg.curriculum, cfg.terminations = {}, {}, {}, {}
  cfg.episode_length_s = 1e9
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 500)
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 1500)
  cfg.sim.contact_sensor_maxmatch = max(cfg.sim.contact_sensor_maxmatch, 500)
  return cfg
