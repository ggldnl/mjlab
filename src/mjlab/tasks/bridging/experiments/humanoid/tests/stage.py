"""Shared runtime support for skill transition viewers.

This module merges registered tasks into one scene and loads their policies. Transition
geometry and skill controls belong in the transition script.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.bridging.experiments.humanoid.bridges.docking.command import (
  DockingCommand,
  DockingCommandCfg,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

if TYPE_CHECKING:
  from tensordict import TensorDict

ROBOT = "robot"
BRIDGE = "bridge"
LOG_ROOT = Path("logs") / "rsl_rl"
PLAYBACK_COLOR = (0.2, 0.8, 1.0, 0.45)


class TransitionDockingCommand(DockingCommand):
  """Docking command opened by a transition instead of an episode reset."""

  def __init__(self, cfg: DockingCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self._opened = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._playback_opened = torch.zeros_like(self._opened)
    self._playback_ghost = self._make_target_ghost(PLAYBACK_COLOR)
    self.playback_sequence: torch.Tensor | None = None
    self.show_playback = False
    self.aimed = False
    self.active = False

  @property
  def step(self) -> torch.Tensor:
    return (self._env.episode_length_buf - self._opened).clamp(min=0)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    here = self.state_now()[env_ids]
    self.target_sequence[env_ids] = here[:, None]
    self.history[env_ids] = here[:, None]
    self.target_actions[env_ids] = 0.0
    self.window_steps[env_ids] = 1
    self.docking[env_ids] = False
    self.captured[env_ids] = False
    self.capture_step[env_ids] = -1
    self.playback_sequence = None
    self.aimed = False
    self.active = False

  def _update_command(self) -> None:
    if self.active:
      self.advance()
    self.history = torch.roll(self.history, shifts=-1, dims=1)
    self.history[:, -1] = self.state_now()

  def open_window(
    self,
    env_ids: torch.Tensor,
    targets: torch.Tensor,
    duration_s: torch.Tensor,
    target_actions: torch.Tensor | None = None,
  ) -> None:
    self._opened[env_ids] = self._env.episode_length_buf[env_ids]
    self._advanced_at = -1
    super().open_window(env_ids, targets, duration_s, target_actions)
    self.aimed = True
    self.active = True

  def stop(self) -> None:
    self.active = False
    self.docking[:] = False
    self.captured[:] = False

  def start_playback(self, sequence: torch.Tensor) -> None:
    """Play a placed reference trajectory from its first state."""
    if sequence.ndim != 3 or sequence.shape[0] != self.num_envs:
      raise ValueError("playback must have shape (num_envs, time, state)")
    self.playback_sequence = sequence
    self._playback_opened[:] = self._env.episode_length_buf

  def _debug_vis_impl(self, visualizer) -> None:
    if self.aimed:
      super()._debug_vis_impl(visualizer)
    if not self.show_playback or self.playback_sequence is None:
      return
    age = (self._env.episode_length_buf - self._playback_opened).clamp(
      max=self.playback_sequence.shape[1] - 1
    )
    for batch in visualizer.get_env_indices(self.num_envs):
      self._draw_ghost(
        visualizer,
        self.playback_sequence[batch, age[batch]],
        batch,
        "kick_recording",
        model=self._playback_ghost,
        alpha=PLAYBACK_COLOR[3],
      )


@dataclass(kw_only=True)
class TransitionDockingCommandCfg(DockingCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> TransitionDockingCommand:
    return TransitionDockingCommand(self, env)


def _external_command(cfg: DockingCommandCfg) -> TransitionDockingCommandCfg:
  values = {item.name: getattr(cfg, item.name) for item in fields(cfg)}
  values.update(dataset_path=None, debug_vis=True, gui=False)
  return TransitionDockingCommandCfg(**values)


def arena(
  bridge_task: str, leaving_task: str, entering_task: str
) -> ManagerBasedRlEnvCfg:
  """Merge the bridge and two skill tasks without copying skill behavior."""
  cfg = load_env_cfg(bridge_task, play=True)
  bridge_cfg = cfg.commands[BRIDGE]
  if not isinstance(bridge_cfg, DockingCommandCfg):
    raise TypeError(f"{bridge_task} does not use DockingCommandCfg")
  cfg.commands[BRIDGE] = _external_command(bridge_cfg)
  cfg.observations = {BRIDGE: copy.deepcopy(cfg.observations["actor"])}
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
  ) -> None:
    agent = load_rl_cfg(task)
    agent.obs_groups = {"actor": (group,), "critic": (group,)}
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    self._runner = runner_cls(env, asdict(agent), device=device)
    self._runner.load(
      str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    self._policy = self._runner.get_inference_policy(device=device)

  @torch.no_grad()
  def __call__(self, obs: TensorDict) -> torch.Tensor:
    return self._policy(obs)


def load_policy(
  task: str,
  env: RslRlVecEnvWrapper,
  group: str,
  device: str,
  checkpoint: Path | None = None,
) -> Policy:
  agent = load_rl_cfg(task)
  path = find_checkpoint(agent.experiment_name, checkpoint)
  print(f"{group:8s} {path}")
  return Policy(task, path, env, group, device)


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
