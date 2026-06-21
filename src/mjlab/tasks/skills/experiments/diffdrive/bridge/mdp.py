"""MDP terms wiring the bridge into a manager-based RL environment.

This module is the only experiment-specific glue the trainer needs:

* BridgeCommand  a command term that, on every reset, drops the robot at a harvested
                 interrupt state and publishes the goal it should reach, namely the
                 window state of the next skill closest to that interrupt.
* TwistAction    an action term turning the policy's twist into wheel torques, with
                 the same servo the deployed bridge uses.
* observation / reward / termination functions read the robot and the goal and turn
                 them into the learning signal: a smooth pull toward the goal, a bonus
                 for reaching it, a penalty for leaving the corridors.

It is loaded only through bridge_env_cfg (its name keeps it out of mjlab's task
auto-import), so importing torch and the simulator here is fine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.skills.experiments.diffdrive.bridge import config, features, rollouts
from mjlab.tasks.skills.experiments.diffdrive.experiment import corridor_speeds
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


class GridQuery:
  """Vectorized corridor membership test in grid-local coordinates.

  Holds the world's occupancy as a tensor so a whole batch of robot positions can be
  scored at once: is_free is True inside a corridor and False on a wall or off-grid,
  which is exactly the crash test.
  """

  def __init__(self, world: GridWorld, device: str) -> None:
    occupancy = torch.as_tensor(world.grid.grid != 0, device=device)
    self._occupancy = occupancy
    self._cell = world.cell
    self._nrows = world.nrows
    self._ncols = world.ncols

  def is_free(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    c = torch.floor(x / self._cell).long()
    r = self._nrows - 1 - torch.floor(y / self._cell).long()
    in_bounds = (r >= 0) & (r < self._nrows) & (c >= 0) & (c < self._ncols)
    rr = r.clamp(0, self._nrows - 1)
    cc = c.clamp(0, self._ncols - 1)
    return self._occupancy[rr, cc] & in_bounds


# Command term: reset to an interrupt, aim at the matched window state.


@dataclass(kw_only=True)
class BridgeCommandCfg(CommandTermCfg):
  """Configuration for the bridge command term."""

  entity_name: str
  slow: float = 1.3
  fast: float = 2.2
  mode: str = "cruise"
  cell: float = 1.0
  seed: int = 0
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> BridgeCommand:
    return BridgeCommand(self, env)


class BridgeCommand(CommandTerm):
  """Per-episode interrupt and goal for the bridge.

  On reset it samples a corridor transition, places the robot at one of that
  transition's harvested interrupt states, and sets the goal to the next skill's
  window state closest to that interrupt. The goal is published in world frame so
  observations and rewards (which are goal-relative) need no per-env bookkeeping.
  """

  cfg: BridgeCommandCfg

  def __init__(self, cfg: BridgeCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.world = GridWorld(cell=cfg.cell)
    self.grid = GridQuery(self.world, self.device)

    speeds = corridor_speeds(self.world, slow=cfg.slow, fast=cfg.fast)
    harvest = rollouts.harvest_transitions(self.world, speeds, cfg.mode, seed=cfg.seed)
    self._interrupts = torch.as_tensor(
      harvest.interrupts, dtype=torch.float32, device=self.device
    )
    self._windows = torch.as_tensor(
      harvest.windows, dtype=torch.float32, device=self.device
    )
    self._num_transitions = len(harvest.transitions)

    self._origin_xy = env.scene.env_origins[:, :2].clone()
    left = self.robot.find_joints("left_wheel")[0][0]
    right = self.robot.find_joints("right_wheel")[0][0]
    self._wheel_ids = torch.tensor([left, right], device=self.device)

    self._goal = torch.zeros(self.num_envs, 5, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._goal

  def _update_metrics(self) -> None:
    pass

  def _update_command(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    k = len(env_ids)
    if k == 0:
      return
    n_inter = self._interrupts.shape[1]
    n_win = self._windows.shape[1]
    t = torch.randint(self._num_transitions, (k,), device=self.device)
    j = torch.randint(n_inter, (k,), device=self.device)
    interrupt = self._interrupts[t, j]  # [k, 5], grid-local
    window = self._windows[t]  # [k, M, 5]

    dist = features.goal_distance(interrupt.unsqueeze(1).expand(-1, n_win, -1), window)
    goal_local = window[torch.arange(k, device=self.device), dist.argmin(dim=1)]

    origin = self._origin_xy[env_ids]
    goal_world = goal_local.clone()
    goal_world[:, :2] += origin
    self._goal[env_ids] = goal_world

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
    self._raw = torch.zeros(self.num_envs, config.ACTION_DIM, device=self.device)

  @property
  def action_dim(self) -> int:
    return config.ACTION_DIM

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
      config.KV * (target - wheel_vel), -config.TORQUE_LIMIT, config.TORQUE_LIMIT
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


def _crash_mask(term: BridgeCommand) -> torch.Tensor:
  data = term.robot.data
  x = data.root_link_pos_w[:, 0] - term._origin_xy[:, 0]
  y = data.root_link_pos_w[:, 1] - term._origin_xy[:, 1]
  return ~term.grid.is_free(x, y)


def _tolerance(error: torch.Tensor, margin: float) -> torch.Tensor:
  """A smooth bump, 1 at zero error and decaying to about 0.1 at the margin."""
  scaled = error / margin * 2.1460
  return torch.exp(-0.5 * scaled * scaled)


# Observations.


def bridge_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = _term(env)
  return features.observation(_robot_state(term), term.command)


# Rewards.


def goal_tracking(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Smooth pull toward the goal: high when close in position, heading, and speed."""
  term = _term(env)
  state, goal = _robot_state(term), term.command
  forward, lateral = features.position_error(state, goal)
  pos = torch.sqrt(forward * forward + lateral * lateral)
  head = torch.abs(features.heading_error(state, goal))
  spd = torch.abs(features.speed_error(state, goal))
  return (
    _tolerance(pos, config.M_POS)
    * _tolerance(head, config.M_HEAD)
    * _tolerance(spd, config.M_SPEED)
  )


def goal_reached(env: ManagerBasedRlEnv) -> torch.Tensor:
  """One on the step the robot enters the goal tolerance, else zero (a bonus)."""
  term = _term(env)
  return features.success(_robot_state(term), term.command).float()


def crashed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """One on the step the robot leaves the corridors, else zero (a penalty)."""
  return _crash_mask(_term(env)).float()


def action_magnitude(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared action, to discourage needlessly violent twists (a penalty)."""
  return torch.sum(env.action_manager.action**2, dim=-1)


# Terminations.


def reached_goal(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = _term(env)
  return features.success(_robot_state(term), term.command)


def left_corridor(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _crash_mask(_term(env))
