"""MDP terms wiring the bridge into a manager-based RL environment.

The task is defined only by the dataset of couples (one skill1 end trajectory, one skill2
start trajectory per entry). Each episode picks a couple and drops the robot at the
interrupt, where skill1 leaves it. The robot's recent history (the tail of skill1's end
trajectory) is part of the observation, so the policy knows what it was doing. The bridge
then drives onto skill2's recorded start trajectory and is rewarded for tracking it: a
moving reference walks along the recorded trajectory, advancing to the next recorded state
whenever the robot gets close, so the policy learns to reach a join point and then follow
the rest of the trajectory. There is no reference to walls or any diffdrive-specific
geometry; effort is penalized.

* BridgeCommand  on reset, picks a couple and a start index, drops the robot at the
                 interrupt, seeds the history, and publishes the moving reference state.
* TwistAction    turns the policy's twist into wheel torques, the same servo deployment
                 uses.
* reward and termination functions read the robot and the reference: closeness to the
  advancing reference (tracking), an effort penalty, and success when the reference has
  been tracked to the end.

It is loaded only through bridge_env_cfg (its name keeps it out of mjlab's task
auto-import), so importing torch and the simulator here is fine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.skills.experiments.diffdrive.bridge import features
from mjlab.tasks.skills.experiments.diffdrive.bridge.dataset import harvest_dataset
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


# Command term: pick a couple, drop at the interrupt, walk a reference along skill2's tube.


@dataclass(kw_only=True)
class BridgeCommandCfg(CommandTermCfg):
  """Configuration for the bridge command term."""

  entity_name: str
  slow: float = 0.5
  fast: float = 1.5
  cell: float = 1.0
  seed: int = 0
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> BridgeCommand:
    return BridgeCommand(self, env)


class BridgeCommand(CommandTerm):
  """Per-episode couple, interrupt, history, and the moving reference on skill2's tube.

  On reset it samples a couple (one skill1 end trajectory, one skill2 start trajectory) and
  a start index along the start trajectory, places the robot at the interrupt (the last
  state of the end trajectory), and seeds the history from the end trajectory's tail. Each
  step it advances a reference index along the start trajectory whenever the robot is within
  track_tol of the current reference, and publishes that reference state as the goal. The
  goal is in world frame so the goal-relative observation and reward need no per-env
  bookkeeping.
  """

  cfg: BridgeCommandCfg

  def __init__(self, cfg: BridgeCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    world = GridWorld(cell=cfg.cell)
    speeds = corridor_speeds(world, slow=cfg.slow, fast=cfg.fast)
    dataset = harvest_dataset(world, speeds, seed=cfg.seed)
    self._end = torch.as_tensor(
      dataset.end_windows, dtype=torch.float32, device=self.device
    )  # [T, C, L, 5]
    self._start = torch.as_tensor(
      dataset.start_windows, dtype=torch.float32, device=self.device
    )  # [T, C, L, 5]
    # One scale per junction (its tube's own spread), so "close to the reference" is a
    # dimensionless test that does not mix different junctions.
    self._scale = torch.stack(
      [features.tube_scale(tube) for tube in self._start]
    )  # [T, 4]
    self._num_junctions, self._num_couples, self._traj_len = self._start.shape[:3]
    self._history_len = CONFIG.history_len

    self._origin_xy = env.scene.env_origins[:, :2].clone()
    left = self.robot.find_joints("left_wheel")[0][0]
    right = self.robot.find_joints("right_wheel")[0][0]
    self._wheel_ids = torch.tensor([left, right], device=self.device)

    self._t = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._c = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._r = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._goal = torch.zeros(self.num_envs, 5, device=self.device)
    self._history = torch.zeros(self.num_envs, self._history_len, 2, device=self.device)
    self._env_scale = torch.zeros(
      self.num_envs, self._scale.shape[1], device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return self._goal

  @property
  def history(self) -> torch.Tensor:
    """The recent (v, omega) window, flattened: [num_envs, 2 * history_len]."""
    return self._history.reshape(self.num_envs, -1)

  def distance_to_goal(self, state: torch.Tensor) -> torch.Tensor:
    """Normalized distance from each robot state to its current reference: [num_envs]."""
    scale = self._env_scale.unsqueeze(1)  # [N, 1, 4], per env's junction
    return features.tube_distance(state, self._goal.unsqueeze(1), scale).squeeze(1)

  def at_end(self) -> torch.Tensor:
    """Whether the reference has been tracked to the last recorded state."""
    return self._r >= self._traj_len - 1

  def _reference_world(self) -> torch.Tensor:
    """The current reference state per env, in world frame: [num_envs, 5]."""
    ref = self._start[self._t, self._c, self._r].clone()
    ref[:, :2] += self._origin_xy
    return ref

  def _update_metrics(self) -> None:
    pass

  def _update_command(self) -> None:
    state = _robot_state(self)
    close = self.distance_to_goal(state) < CONFIG.track_tol
    self._r = torch.where(close & ~self.at_end(), self._r + 1, self._r)
    self._goal = self._reference_world()
    recent = state[:, [V, OMEGA]].unsqueeze(1)  # [N, 1, 2]
    self._history = torch.cat([self._history[:, 1:, :], recent], dim=1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    k = len(env_ids)
    if k == 0:
      return
    device = self.device
    t = torch.randint(self._num_junctions, (k,), device=device)
    c = torch.randint(self._num_couples, (k,), device=device)
    r = torch.randint(self._traj_len, (k,), device=device)  # where to start tracking
    self._t[env_ids] = t
    self._c[env_ids] = c
    self._r[env_ids] = r
    self._env_scale[env_ids] = self._scale[t]

    couple_end = self._end[t, c]  # [k, L, 5], the skill1 end trajectory
    interrupt = couple_end[:, -1]  # [k, 5], where skill1 hands off
    self._history[env_ids] = couple_end[:, -self._history_len :][:, :, [V, OMEGA]]

    origin = self._origin_xy[env_ids]
    goal = self._start[t, c, r].clone()  # [k, 5], first reference on skill2's tube
    goal[:, :2] += origin
    self._goal[env_ids] = goal

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


# Observation: where the moving reference is, plus the recent-motion history.


def bridge_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  term = _term(env)
  goal_relative = features.observation(_robot_state(term), term.command)
  return torch.cat([goal_relative, term.history], dim=-1)


# Rewards: track the advancing reference, spend little effort. Nothing else.


def tracking(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Closeness to the current reference state, in (0, 1] (a smooth tracking reward)."""
  term = _term(env)
  return torch.exp(-term.distance_to_goal(_robot_state(term)))


def effort(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared action, to discourage spending energy (a penalty)."""
  return torch.sum(env.action_manager.action**2, dim=-1)


# Termination: success once the reference has been tracked to the end.


def tracked_to_end(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _term(env).at_end()
