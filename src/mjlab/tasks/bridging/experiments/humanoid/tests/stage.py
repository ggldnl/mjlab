"""Shared runtime support for skill transition viewers.

This module merges registered tasks into one scene and loads their policies. Transition
geometry and skill controls belong in the transition script.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import torch
from tensordict import TensorDict

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal import (
  GOAL_CVAE_TASK_ID,
  GoalRunnerCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.model import (
  GoalCvaeModel,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

ROBOT = "robot"
BRIDGE = "bridge"
LOG_ROOT = Path("logs") / "rsl_rl"


def _external_command(cfg: CommandTermCfg) -> CommandTermCfg:
  """Disable bridge-owned sampling while keeping its observation contract."""
  changes: dict[str, Any] = {"debug_vis": True}
  if hasattr(cfg, "dataset_path"):
    changes["dataset_path"] = None
  if hasattr(cfg, "gui"):
    changes["gui"] = False
  return replace(cfg, **changes)


def arena(
  bridge_task: str,
  leaving_task: str,
  entering_task: str,
  base_checkpoint: Path | None = None,
) -> ManagerBasedRlEnvCfg:
  """Merge the bridge and two skill tasks without copying skill behavior."""
  cfg = load_env_cfg(bridge_task, play=True)
  cfg.commands[BRIDGE] = _external_command(cfg.commands[BRIDGE])
  if base_checkpoint is not None:
    action = cfg.actions.get("joint_pos")
    if action is None or not hasattr(action, "imitation_checkpoint"):
      raise ValueError(f"{bridge_task} does not use an imitation base policy")
    cfg.actions["joint_pos"] = replace(action, imitation_checkpoint=base_checkpoint)
  initial = (
    cfg.observations.get("initial") if bridge_task == GOAL_CVAE_TASK_ID else None
  )
  cfg.observations = {BRIDGE: copy.deepcopy(cfg.observations["actor"])}
  if initial is not None:
    cfg.observations["initial"] = copy.deepcopy(initial)
  cfg.events = {}
  cfg.rewards = {}
  cfg.terminations = {}
  cfg.curriculum = {}
  cfg.metrics = {}
  cfg.scene.num_envs = 1
  cfg.episode_length_s = 1.0e9

  tasks = (
    ("leaving", load_env_cfg(leaving_task, play=True)),
    ("entering", load_env_cfg(entering_task, play=True)),
  )
  rate = cfg.sim.mujoco.timestep * cfg.decimation
  for group, task in tasks:
    task_rate = task.sim.mujoco.timestep * task.decimation
    if abs(task_rate - rate) > 1.0e-9:
      raise ValueError(f"{group} skill has a different control rate")
    cfg.observations[group] = replace(
      copy.deepcopy(task.observations["actor"]), enable_corruption=False
    )
    for name, entity in (task.scene.entities or {}).items():
      cfg.scene.entities.setdefault(name, copy.deepcopy(entity))
    for name, command in task.commands.items():
      if name in cfg.commands:
        continue
      command = copy.deepcopy(command)
      changes = {
        "resampling_time_range": (1.0e9, 1.0e9),
        "gui": False,
        "debug_vis": False,
      }
      if hasattr(command, "reset_robot_to_clip"):
        changes["reset_robot_to_clip"] = False
      cfg.commands[name] = replace(command, **changes)
    present = {sensor.name for sensor in (cfg.scene.sensors or ())}
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + tuple(
      copy.deepcopy(sensor)
      for sensor in (task.scene.sensors or ())
      if sensor.name not in present
    )
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, task.sim.nconmax or 0)
    cfg.sim.njmax = max(cfg.sim.njmax or 0, task.sim.njmax or 0)
    cfg.sim.contact_sensor_maxmatch = max(
      cfg.sim.contact_sensor_maxmatch, task.sim.contact_sensor_maxmatch
    )

  cfg.sim.mujoco = copy.deepcopy(tasks[-1][1].sim.mujoco)
  return cfg


def find_checkpoint(experiment: str, explicit: Path | None = None) -> Path:
  """Return an explicit checkpoint or the newest checkpoint in an experiment."""
  if explicit is not None:
    if not explicit.exists():
      raise SystemExit(f"No checkpoint at {explicit}")
    return explicit
  root = LOG_ROOT / experiment
  found = sorted(root.rglob("model_*.pt"), key=lambda path: path.stat().st_mtime)
  if not found:
    raise SystemExit(f"No checkpoint under {root}")
  return found[-1]


class Policy:
  """Frozen policy reading one observation group from the shared scene."""

  def __init__(
    self,
    task: str,
    checkpoint: Path,
    env: RslRlVecEnvWrapper,
    group: str,
    device: str,
    task_runner: bool = True,
  ) -> None:
    agent = load_rl_cfg(task)
    if task == GOAL_CVAE_TASK_ID:
      saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
      weights = saved["student_state_dict"]
      current = cast(torch.Tensor, env.get_observations()[group])
      initial = cast(torch.Tensor, env.get_observations()["initial"])
      route_slots = cast(GoalRunnerCfg, agent).student.route_slots
      route_dim = route_slots + 1 + 3 * route_slots
      posterior_dim = (
        weights["posterior.0.weight"].shape[1] - initial.shape[-1] - route_dim
      )
      sample = TensorDict(
        {
          group: current,
          "initial": initial,
          "posterior": current.new_zeros((env.num_envs, posterior_dim)),
          "route": current.new_zeros((env.num_envs, route_slots + 1)),
        },
        batch_size=[env.num_envs],
      )
      student_cfg = asdict(cast(GoalRunnerCfg, agent).student)
      student_cfg.pop("class_name")
      self._policy = GoalCvaeModel(
        sample,
        {
          "student": [group, "initial"],
          "posterior": ["posterior"],
          "route": ["route"],
        },
        "student",
        env.num_actions,
        **student_cfg,
      ).to(device)
      self._policy.load_state_dict(weights, strict=True)
      self._policy.eval()
      return
    agent.obs_groups = {"actor": (group,), "critic": (group,)}
    runner_cls = (load_runner_cls(task) if task_runner else None) or MjlabOnPolicyRunner
    self._runner = runner_cls(env, asdict(agent), device=device)
    self._runner.load(
      str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    self._policy = self._runner.get_inference_policy(device=device)

  @torch.no_grad()
  def __call__(self, obs: TensorDict) -> torch.Tensor:
    return self._policy(obs)

  def reset(self) -> None:
    if isinstance(self._policy, GoalCvaeModel):
      self._policy.reset()


def load_policy(
  task: str,
  env: RslRlVecEnvWrapper,
  group: str,
  device: str,
  checkpoint: Path | None = None,
  task_runner: bool = True,
) -> Policy:
  agent = load_rl_cfg(task)
  path = find_checkpoint(agent.experiment_name, checkpoint)
  print(f"{group:8s} {path}")
  return Policy(task, path, env, group, device, task_runner)


def state(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Current robot state in the bridge layout."""
  robot: Entity = env.scene[ROBOT]
  data = robot.data
  return torch.cat(
    (
      data.root_link_pos_w,
      data.root_link_quat_w,
      data.root_link_lin_vel_w,
      data.root_link_ang_vel_w,
      data.joint_pos,
      data.joint_vel,
    ),
    dim=-1,
  )


def fresh_obs(env: ManagerBasedRlEnv):
  """Recompute observations after a transition changes commands."""
  env.observation_manager._obs_buffer = None
  return env.observation_manager.compute()
