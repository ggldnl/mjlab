"""MDP terms wiring the bridge into a manager-based RL environment.

The task is general: only the dataset (skill1's end windows and skill2's start windows)
defines it. Each episode resets the robot to a harvested interrupt, where skill1 leaves
it, and names a goal on skill2's tube. The policy is rewarded for reaching the tube while
spending little effort, with no reference to walls or any diffdrive-specific geometry.

* BridgeCommand  on reset, drops the robot at an interrupt and picks a goal: a state on
                 skill2's representative tube, with a short segment of the tube around it
                 counting as reached.
* TwistAction    turns the policy's twist into wheel torques, the same servo the deployed
                 bridge uses.
* reward and termination functions read the robot and the tube: a bonus for reaching the
  tube, a penalty on effort, and termination on arrival or timeout.

It is loaded only through bridge_env_cfg (its name keeps it out of mjlab's task
auto-import), so importing torch and the simulator here is fine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.skills.experiments.diffdrive.bridge import features
from mjlab.tasks.skills.experiments.diffdrive.bridge.dataset import harvest_dataset
from mjlab.tasks.skills.experiments.diffdrive.bridge.rollouts import representative
from mjlab.tasks.skills.experiments.diffdrive.experiment import CONFIG, corridor_speeds
from mjlab.tasks.skills.experiments.diffdrive.gridworld import GridWorld
from mjlab.tasks.skills.experiments.diffdrive.robot import (
  BASE_HEIGHT,
  OMEGA,
  THETA,
  TRACK,
  WHEEL_RADIUS,
  V,
  X,
  Y,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

COMMAND_NAME = "bridge"


# Command term: reset to an interrupt, aim at a state on the next skill's tube.


@dataclass(kw_only=True)
class BridgeCommandCfg(CommandTermCfg):
  """Configuration for the bridge command term."""

  entity_name: str
  slow: float = 0.5
  fast: float = 1.5
  mode: str = "hold"
  cell: float = 1.0
  seed: int = 0
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> BridgeCommand:
    return BridgeCommand(self, env)


class BridgeCommand(CommandTerm):
  """Per-episode interrupt and tube goal, built entirely from the dataset.

  On reset it samples a transition, places the robot at one of skill1's harvested
  interrupt states, and picks a goal: a state on skill2's representative tube. A short
  segment of the tube ending at that state counts as having reached it, which gives the
  policy a fatter, connected target instead of a single point. The goal is published in
  world frame so the goal-relative observations and rewards need no per-env bookkeeping.
  """

  cfg: BridgeCommandCfg

  def __init__(self, cfg: BridgeCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    world = GridWorld(cell=cfg.cell)
    speeds = corridor_speeds(world, slow=cfg.slow, fast=cfg.fast)
    dataset = harvest_dataset(world, speeds, cfg.mode, seed=cfg.seed)
    # skill1 leaves the robot at the last state of each end-window rollout.
    interrupts = dataset.end_windows[:, :, -1, :]  # [T, N, 5]
    # skill2's tube, reduced to one representative line per transition.
    reps = np.stack(
      [representative(list(sw), CONFIG.representative) for sw in dataset.start_windows]
    )  # [T, L, 5]

    self._interrupts = torch.as_tensor(
      interrupts, dtype=torch.float32, device=self.device
    )
    self._reps = torch.as_tensor(reps, dtype=torch.float32, device=self.device)
    # One scale per transition (each tube's own spread), not pooled across junctions:
    # pooling would mix different corridor places and headings and make the merge test
    # far too lenient.
    tubes = torch.as_tensor(
      dataset.start_windows, dtype=torch.float32, device=self.device
    )
    self._scale = torch.stack([features.tube_scale(tube) for tube in tubes])  # [T, 4]
    self._num_transitions = self._reps.shape[0]
    self._rep_len = self._reps.shape[1]
    self._segment_len = min(CONFIG.merge_segment, self._rep_len)

    self._origin_xy = env.scene.env_origins[:, :2].clone()
    left = self.robot.find_joints("left_wheel")[0][0]
    right = self.robot.find_joints("right_wheel")[0][0]
    self._wheel_ids = torch.tensor([left, right], device=self.device)

    self._goal = torch.zeros(self.num_envs, 5, device=self.device)
    self._segment = torch.zeros(self.num_envs, self._segment_len, 5, device=self.device)
    self._env_scale = torch.zeros(
      self.num_envs, self._scale.shape[1], device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return self._goal

  def reached(self) -> torch.Tensor:
    """Whether the robot is on the tube (within the normalized arrival threshold)."""
    scale = self._env_scale.unsqueeze(1)  # [N, 1, 4], per env's transition
    dist = features.tube_distance(_robot_state(self), self._segment, scale)
    return dist.amin(dim=-1) < CONFIG.arrival_threshold

  def _update_metrics(self) -> None:
    pass

  def _update_command(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    k = len(env_ids)
    if k == 0:
      return
    device = self.device
    t = torch.randint(self._num_transitions, (k,), device=device)
    j = torch.randint(self._interrupts.shape[1], (k,), device=device)
    interrupt = self._interrupts[t, j]  # [k, 5], grid-local

    idx = torch.randint(self._rep_len, (k,), device=device)  # merge target index
    goal_local = self._reps[t, idx]  # [k, 5]
    # The segment of the tube ending at the target (the part that leads into it).
    offsets = torch.arange(self._segment_len - 1, -1, -1, device=device)
    seg_idx = (idx.unsqueeze(1) - offsets).clamp_min(0)  # [k, segment_len]
    segment_local = self._reps[t.unsqueeze(1), seg_idx]  # [k, segment_len, 5]

    origin = self._origin_xy[env_ids]
    goal_world = goal_local.clone()
    goal_world[:, :2] += origin
    self._goal[env_ids] = goal_world
    segment_world = segment_local.clone()
    segment_world[:, :, :2] += origin.unsqueeze(1)
    self._segment[env_ids] = segment_world
    self._env_scale[env_ids] = self._scale[t]

    self._write_interrupt(env_ids, interrupt, origin)

  def _write_interrupt(
    self, env_ids: torch.Tensor, interrupt: torch.Tensor, origin: torch.Tensor
  ) -> None:
    """Place the robot so it realizes the (grid-local) interrupt reduced state."""
    x = interrupt[:, X] + origin[:, 0]
    y = interrupt[:, Y] + origin[:, 1]
    theta = interrupt[:, THETA]
    v = interrupt[:, V]
    omega = interrupt[:, OMEGA]
    z = torch.full_like(x, BASE_HEIGHT)
    zero = torch.zeros_like(x)
    quat = torch.stack([torch.cos(theta / 2), zero, zero, torch.sin(theta / 2)], dim=-1)
    lin_w = torch.stack([v * torch.cos(theta), v * torch.sin(theta), zero], dim=-1)
    ang_w = torch.stack([zero, zero, omega], dim=-1)
    root_state = torch.cat([torch.stack([x, y, z], dim=-1), quat, lin_w, ang_w], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids)

    half = TRACK / 2.0
    left = (v - omega * half) / WHEEL_RADIUS
    right = (v + omega * half) / WHEEL_RADIUS
    self.robot.write_joint_velocity_to_sim(
      torch.stack([left, right], dim=-1), joint_ids=self._wheel_ids, env_ids=env_ids
    )


# Action term: twist to wheel torques.


@dataclass(kw_only=True)
class TwistActionCfg(ActionTermCfg):
  """Configuration for the twist action term."""

  entity_name: str

  def build(self, env: ManagerBasedRlEnv) -> TwistAction:
    return TwistAction(self, env)


class TwistAction(ActionTerm):
  """Drive the wheels from a body-twist action, with the same servo as deployment.

  The policy's 2-vector becomes a clamped twist (forward speed, yaw rate); a velocity
  servo turns that into wheel torques, clamped to the actuator limit. That clamp is
  what gives the robot bounded acceleration, so residual speed cannot vanish at once.
  """

  cfg: TwistActionCfg

  def __init__(self, cfg: TwistActionCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self._entity: Entity = env.scene[cfg.entity_name]
    left = self._entity.find_joints("left_wheel")[0][0]
    right = self._entity.find_joints("right_wheel")[0][0]
    self._wheel_ids = torch.tensor([left, right], device=self.device)
    self._raw = torch.zeros(self.num_envs, CONFIG.action_dim, device=self.device)

  @property
  def action_dim(self) -> int:
    return CONFIG.action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw[:] = actions

  def apply_actions(self) -> None:
    v, omega = features.twist_from_action(self._raw)
    half = TRACK / 2.0
    left_target = (v - omega * half) / WHEEL_RADIUS
    right_target = (v + omega * half) / WHEEL_RADIUS
    target = torch.stack([left_target, right_target], dim=-1)
    wheel_vel = self._entity.data.joint_vel[:, self._wheel_ids]
    torque = torch.clamp(
      CONFIG.kv * (target - wheel_vel), -CONFIG.torque_limit, CONFIG.torque_limit
    )
    self._entity.set_joint_effort_target(torque, joint_ids=self._wheel_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw[env_ids] = 0.0


# Reading the robot and the goal.


def _term(env: ManagerBasedRlEnv) -> BridgeCommand:
  return cast("BridgeCommand", env.command_manager.get_term(COMMAND_NAME))


def _robot_state(term: BridgeCommand) -> torch.Tensor:
  """The robot's reduced state [x, y, theta, v, omega] in world frame."""
  data = term.robot.data
  return torch.stack(
    [
      data.root_link_pos_w[:, 0],
      data.root_link_pos_w[:, 1],
      data.heading_w,
      data.root_link_lin_vel_b[:, 0],
      data.root_link_ang_vel_w[:, 2],
    ],
    dim=-1,
  )


# Observation.


def bridge_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = _term(env)
  return features.observation(_robot_state(term), term.command)


# Rewards: a bonus for reaching the tube, a penalty on effort. Nothing else.


def arrival(env: ManagerBasedRlEnv) -> torch.Tensor:
  """One on the step the robot reaches the tube, else zero (a bonus)."""
  return _term(env).reached().float()


def effort(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared action, to discourage spending energy (a penalty)."""
  return torch.sum(env.action_manager.action**2, dim=-1)


# Terminations: success on reaching the tube, plus the time budget.


def reached_tube(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _term(env).reached()
