"""One scene with solid walls, movable boxes and separate skill observations.

Walk and push hold different arm rest poses. The scene keeps the walk one, and the push
observations and actions are shifted onto the push rest pose, so neither policy sees a
change of offsets.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from functools import partial
from typing import Callable

import mujoco
import torch
from tensordict import TensorDict

from mjlab.asset_zoo.objects.box import get_box_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.bridging.config.t1.demos.sokoban.board import Board
from mjlab.tasks.bridging.config.t1.skills.push import PUSH_TASK_ID
from mjlab.tasks.bridging.config.t1.skills.push.command import PushCommandCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

WALK_TASK = "Mjlab-T1-Walk-Natural"


def rest_pose(env: ManagerBasedRlEnv, joint_pos: dict[str, float]) -> torch.Tensor:
  """Default joint positions of the skill that was trained with joint_pos"""
  robot = env.scene["robot"]
  pose = robot.data.default_joint_pos.clone()
  ids, _, values = resolve_matching_names_values(joint_pos, robot.joint_names)
  pose[:, ids] = torch.tensor(values, device=pose.device)
  return pose


def push_joint_pos(
  env: ManagerBasedRlEnv, joint_pos: dict[str, float], biased: bool = True
) -> torch.Tensor:
  robot = env.scene["robot"]
  position = robot.data.joint_pos_biased if biased else robot.data.joint_pos
  return position - rest_pose(env, joint_pos)


def action_shift(env: ManagerBasedRlEnv, joint_pos: dict[str, float]) -> torch.Tensor:
  action = env.action_manager.get_term("joint_pos")
  assert isinstance(action, JointPositionAction)
  return (
    rest_pose(env, joint_pos)[:, action.target_ids] - action.offset
  ) / action.scale


def push_last_action(
  env: ManagerBasedRlEnv, joint_pos: dict[str, float]
) -> torch.Tensor:
  return env.action_manager.action - action_shift(env, joint_pos)


class PushPolicy:
  """Shift push actions from the push rest pose onto the scene one"""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    policy: Callable[[TensorDict], torch.Tensor],
    joint_pos: dict[str, float],
  ):
    self.policy, self.shift = policy, action_shift(env, joint_pos)

  def __call__(self, observations: TensorDict) -> torch.Tensor:
    return self.policy(observations) + self.shift

  def reset(self) -> None:
    reset = getattr(self.policy, "reset", None)
    if callable(reset):
      reset()


def push_rest_pose(push_task: str = PUSH_TASK_ID) -> dict[str, float]:
  return dict(
    load_env_cfg(push_task, play=True).scene.entities["robot"].init_state.joint_pos
    or {}
  )


def box_names(board: Board) -> tuple[str, ...]:
  return tuple(f"box_{index}" for index in range(len(board.boxes)))


def _floor_spec(board: Board) -> mujoco.MjSpec:
  sites = []
  for cell in sorted(board.floor):
    x, y = board.center(cell)
    color = (
      "0.2 0.65 0.3 0.7"
      if cell in board.goals
      else ("0.65 0.65 0.7 0.35" if sum(cell) % 2 else "0.35 0.35 0.4 0.35")
    )
    sites.append(
      f'<site name="cell_{cell[0]}_{cell[1]}" type="box" pos="{x} {y} 0.002" size="0.495 0.495 0.002" rgba="{color}"/>'
    )
  return mujoco.MjSpec.from_string(
    '<mujoco><worldbody><body name="grid">'
    + "".join(sites)
    + "</body></worldbody></mujoco>"
  )


def reset_scene(env: ManagerBasedRlEnv, env_ids: torch.Tensor, board: Board) -> None:
  robot = env.scene["robot"]
  root = robot.data.default_root_state[env_ids].clone()
  root[:, :2] = torch.tensor(board.center(board.robot), device=env.device)
  root[:, :3] += env.scene.env_origins[env_ids]
  root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
  root[:, 7:] = 0.0
  robot.write_root_state_to_sim(root, env_ids)
  robot.write_joint_state_to_sim(
    robot.data.default_joint_pos[env_ids],
    robot.data.default_joint_vel[env_ids],
    env_ids=env_ids,
  )
  for name, cell in zip(box_names(board), board.boxes, strict=True):
    box = env.scene[name]
    root = box.data.default_root_state[env_ids].clone()
    root[:, :2] = torch.tensor(board.center(cell), device=env.device)
    root[:, 2] = 0.5
    root[:, :3] += env.scene.env_origins[env_ids]
    root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    root[:, 7:] = 0.0
    box.write_root_state_to_sim(root, env_ids)


def sokoban_env_cfg(
  board: Board, walk_task: str = WALK_TASK, push_task: str = PUSH_TASK_ID
) -> ManagerBasedRlEnvCfg:
  walk = load_env_cfg(walk_task, play=True)
  push = load_env_cfg(push_task, play=True)
  if (
    walk.actions != push.actions
    or walk.decimation != push.decimation
    or walk.sim.mujoco.timestep != push.sim.mujoco.timestep
  ):
    raise ValueError("Walk and push must use the same joint actions and control rate")
  cfg = copy.deepcopy(push)
  cfg.scene.num_envs = 1
  cfg.scene.env_spacing = float(max(board.width, board.height) + 2)
  pose = dict(push.scene.entities["robot"].init_state.joint_pos or {})
  cfg.scene.entities["robot"].init_state = copy.deepcopy(
    walk.scene.entities["robot"].init_state
  )
  push_obs = replace(copy.deepcopy(push.observations["actor"]), enable_corruption=False)
  push_obs.terms["joint_pos"] = replace(
    push_obs.terms["joint_pos"],
    func=push_joint_pos,
    params={**push_obs.terms["joint_pos"].params, "joint_pos": pose},
  )
  push_obs.terms["actions"] = replace(
    push_obs.terms["actions"], func=push_last_action, params={"joint_pos": pose}
  )
  cfg.observations = {
    "walk": replace(copy.deepcopy(walk.observations["actor"]), enable_corruption=False),
    "push": push_obs,
  }
  twist = copy.deepcopy(walk.commands["twist"])
  if not isinstance(twist, UniformVelocityCommandCfg):
    raise TypeError("Sokoban walk must accept a velocity command named twist")
  cfg.commands["twist"] = replace(
    twist,
    resampling_time_range=(1.0e9, 1.0e9),
    gui=False,
    rel_heading_envs=0.0,
    rel_standing_envs=0.0,
    rel_world_envs=0.0,
    rel_forward_envs=0.0,
    init_velocity_prob=0.0,
  )
  push_cmd = push.commands["push"]
  assert isinstance(push_cmd, PushCommandCfg)
  cfg.commands["push"] = replace(
    push_cmd, box_name="box_0", resampling_time_range=(1.0e9, 1.0e9), gui=False
  )
  # The pace only feeds the training rewards
  cfg.commands.pop("pace", None)
  box = cfg.scene.entities.pop("box")
  sensors = tuple(
    sensor
    for sensor in (cfg.scene.sensors or ())
    if isinstance(sensor, ContactSensorCfg)
    and sensor.secondary is not None
    and sensor.secondary.entity == "box"
  )
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor not in sensors
  )
  for name, cell in zip(box_names(board), board.boxes, strict=True):
    cfg.scene.entities[name] = replace(
      copy.deepcopy(box),
      init_state=EntityCfg.InitialStateCfg(
        pos=(*board.center(cell), 0.5), joint_pos={}
      ),
    )
    for sensor in sensors:
      assert sensor.secondary is not None
      cfg.scene.sensors += (
        replace(
          copy.deepcopy(sensor),
          name=sensor.name.replace("box", name),
          secondary=replace(sensor.secondary, entity=name),
        ),
      )
  for x, y in sorted(board.walls):
    cfg.scene.entities[f"wall_{x}_{y}"] = replace(
      get_box_cfg(half_size=(0.5, 0.5, 0.5), color=(0.3, 0.35, 0.4, 1.0)),
      init_state=EntityCfg.InitialStateCfg(pos=(x + 0.5, y + 0.5, 0.5), joint_pos={}),
    )
  cfg.scene.entities["grid"] = EntityCfg(
    spec_fn=partial(_floor_spec, board),
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={}),
  )
  cfg.events = {
    "reset_scene": EventTermCfg(func=reset_scene, mode="reset", params={"board": board})
  }
  cfg.rewards = {}
  cfg.metrics = {}
  cfg.terminations = {}
  cfg.curriculum = {}
  cfg.episode_length_s = 1.0e9
  cfg.sim.njmax = max(1000, 100 * len(board.boxes))
  return cfg
