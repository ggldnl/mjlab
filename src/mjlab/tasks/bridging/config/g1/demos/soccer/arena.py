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
from mjlab.tasks.bridging.config.g1.demos.parkour.arena import place_static_entities
from mjlab.tasks.bridging.config.g1.demos.soccer.config import Scene, Settings
from mjlab.tasks.bridging.config.g1.skills.jump import JUMP_TASK_ID
from mjlab.tasks.bridging.config.g1.skills.kick import KICK_TASK_ID
from mjlab.tasks.bridging.config.g1.skills.kick import mdp as kick_mdp
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

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
  x, y = scene.goal_position
  yaw = math.radians(scene.goal_heading_degrees)
  half = scene.goal_width / 2
  posts = []
  for side in (-half, half):
    px, py = x - math.sin(yaw) * side, y + math.cos(yaw) * side
    posts.append(
      f'<geom type="cylinder" size="0.035 {scene.goal_height / 2}" pos="{px} {py} {scene.goal_height / 2}" rgba="1 1 1 1"/>'
    )
  posts.append(
    f'<geom type="capsule" size="0.035" fromto="{x + math.sin(yaw) * half} {y - math.cos(yaw) * half} {scene.goal_height} {x - math.sin(yaw) * half} {y + math.cos(yaw) * half} {scene.goal_height}" rgba="1 1 1 1"/>'
  )
  return mujoco.MjSpec.from_string(
    '<mujoco><worldbody><body name="field">'
    f'<site type="box" size="{scene.pitch_length / 2} {scene.pitch_width / 2} 0.002" pos="{scene.pitch_length / 2 - 1} 0 0.002" rgba="0.15 0.45 0.2 0.5"/>'
    + "".join(posts)
    + "</body></worldbody></mujoco>"
  )


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
  place_static_entities(env, env_ids, ("fallen_robot", "field"))
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
