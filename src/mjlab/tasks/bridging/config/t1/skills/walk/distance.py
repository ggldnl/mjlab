"""Walk a latched distance, brake at the destination and remain standing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import wrap_to_pi
from mjlab.viewer.debug_visualizer import DebugVisualizer


class WalkDistanceCommand(UniformVelocityCommand):
  cfg: WalkDistanceCommandCfg

  def __init__(self, cfg: WalkDistanceCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.target_xy = torch.zeros(self.num_envs, 2, device=self.device)
    self.elapsed = torch.zeros(self.num_envs, device=self.device)
    self.settled_time = torch.zeros_like(self.elapsed)
    self.arrived = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.metrics = {
      name: torch.zeros_like(self.elapsed)
      for name in (
        "remaining_distance",
        "position_error",
        "lateral_error",
        "heading_error",
        "settled_at_goal",
      )
    }

  @property
  def axis(self) -> torch.Tensor:
    return torch.stack((self.heading_target.cos(), self.heading_target.sin()), dim=-1)

  @property
  def remaining(self) -> torch.Tensor:
    offset = self.target_xy - self.robot.data.root_link_pos_w[:, :2]
    return (offset * self.axis).sum(-1)

  @property
  def lateral_error(self) -> torch.Tensor:
    offset = self.robot.data.root_link_pos_w[:, :2] - self.target_xy
    axis = self.axis
    return -offset[:, 0] * axis[:, 1] + offset[:, 1] * axis[:, 0]

  @property
  def yaw_error(self) -> torch.Tensor:
    return wrap_to_pi(self.heading_target - self.robot.data.heading_w)

  @property
  def position_error(self) -> torch.Tensor:
    return (self.target_xy - self.robot.data.root_link_pos_w[:, :2]).norm(dim=-1)

  @property
  def goal(self) -> torch.Tensor:
    return torch.stack((self.remaining, self.lateral_error, self.yaw_error), dim=-1)

  def set_goal(
    self,
    target_xy: torch.Tensor,
    heading: torch.Tensor,
    env_ids: torch.Tensor | None = None,
  ) -> None:
    """Latch a world destination and the straight heading used to approach it"""
    ids = slice(None) if env_ids is None else env_ids
    if (
      target_xy.shape != self.target_xy[ids].shape
      or heading.shape != self.heading_target[ids].shape
      or not bool(torch.isfinite(target_xy).all() & torch.isfinite(heading).all())
    ):
      raise ValueError(
        "Walk goal needs a finite XY position and heading per environment"
      )
    self.target_xy[ids] = target_xy
    self.heading_target[ids] = heading
    self.elapsed[ids] = self.settled_time[ids] = 0.0
    self.arrived[ids] = False
    self.time_left[ids] = 1.0e9
    self._write_velocity()
    observations = getattr(self._env, "observation_manager", None)
    if observations is not None:
      observations._obs_buffer = None

  def set_distance(
    self, distance: torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    """Walk metres forward from the current position and heading"""
    ids = slice(None) if env_ids is None else env_ids
    heading = self.robot.data.heading_w[ids]
    if (
      distance.shape != heading.shape
      or not bool(torch.isfinite(distance).all())
      or bool((distance < 0).any())
    ):
      raise ValueError("Walk distance must be finite and nonnegative per environment")
    axis = torch.stack((heading.cos(), heading.sin()), dim=-1)
    self.set_goal(
      self.robot.data.root_link_pos_w[ids, :2] + distance[:, None] * axis,
      heading,
      env_ids,
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._env.scene.write_data_to_sim()
    self._env.sim.forward()
    count = len(env_ids)
    sample = torch.empty(count, device=self.device)
    distance = sample.uniform_(*self.cfg.distance_range).clone()
    standing = torch.rand(count, device=self.device) < self.cfg.rel_standing_envs
    distance[standing] = 0.0
    heading_offset = sample.uniform_(
      -self.cfg.initial_heading_error, self.cfg.initial_heading_error
    ).clone()
    lateral = sample.uniform_(
      -self.cfg.initial_lateral_error, self.cfg.initial_lateral_error
    ).clone()
    heading_offset[standing] = lateral[standing] = 0.0
    heading = self.robot.data.heading_w[env_ids] + heading_offset
    axis = torch.stack((heading.cos(), heading.sin()), dim=-1)
    left = torch.stack((-axis[:, 1], axis[:, 0]), dim=-1)
    self.set_goal(
      self.robot.data.root_link_pos_w[env_ids, :2]
      + distance[:, None] * axis
      + lateral[:, None] * left,
      heading,
      env_ids,
    )

  def _write_velocity(self) -> None:
    ramp = (self.elapsed / self.cfg.ramp_seconds).clamp(max=1.0)
    brake = torch.sqrt(
      2
      * self.cfg.deceleration
      * (self.remaining - self.cfg.position_tolerance / 2).clamp(min=0)
    )
    self.vel_command_b.zero_()
    self.vel_command_b[:, 0] = torch.where(
      self.arrived, 0.0, torch.minimum(self.cfg.walk_speed * ramp, brake)
    )
    self.is_standing_env[:] = self.vel_command_b[:, 0] == 0
    self.heading_error.copy_(self.yaw_error)

  def _update_command(self) -> None:
    self.elapsed += self._env.step_dt
    ready = (
      (self.position_error <= self.cfg.position_tolerance)
      & (self.yaw_error.abs() <= self.cfg.goal_heading_tolerance)
      & (self.robot.data.root_link_lin_vel_w.norm(dim=-1) <= self.cfg.settled_speed)
      & (self.robot.data.root_link_ang_vel_w.norm(dim=-1) <= self.cfg.settled_speed)
    )
    self.settled_time[:] = torch.where(
      ready, self.settled_time + self._env.step_dt, 0.0
    )
    self.arrived |= self.settled_time >= self.cfg.settle_seconds
    self._write_velocity()

  def _update_metrics(self) -> None:
    self.metrics["remaining_distance"].copy_(self.remaining)
    self.metrics["position_error"].copy_(self.position_error)
    self.metrics["lateral_error"].copy_(self.lateral_error.abs())
    self.metrics["heading_error"].copy_(self.yaw_error.abs())
    self.metrics["settled_at_goal"].copy_(self.arrived.float())

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    super()._debug_vis_impl(visualizer)
    for idx in visualizer.get_env_indices(self.num_envs):
      target = self.robot.data.root_link_pos_w[idx].cpu().numpy().copy()
      target[:2] = self.target_xy[idx].cpu().numpy()
      target[2] = float(self._env.scene.env_origins[idx, 2]) + 0.05
      visualizer.add_sphere(
        target, 0.05, (0.2, 0.9, 0.3, 0.8), label="Walk destination"
      )


@dataclass(kw_only=True)
class WalkDistanceCommandCfg(UniformVelocityCommandCfg):
  distance_range: tuple[float, float] = (0.25, 4.0)
  walk_speed: float = 0.5
  ramp_seconds: float = 0.5
  deceleration: float = 0.4
  position_tolerance: float = 0.04
  goal_heading_tolerance: float = 0.05
  settled_speed: float = 0.08
  settle_seconds: float = 0.4
  initial_lateral_error: float = 0.05
  initial_heading_error: float = 0.05
  max_lateral_error: float = 0.25
  max_heading_error: float = 0.6
  max_overshoot: float = 0.15

  def build(self, env: ManagerBasedRlEnv) -> WalkDistanceCommand:
    values = (
      *self.distance_range,
      self.walk_speed,
      self.ramp_seconds,
      self.deceleration,
      self.position_tolerance,
      self.goal_heading_tolerance,
      self.settled_speed,
      self.settle_seconds,
      self.initial_lateral_error,
      self.initial_heading_error,
      self.max_lateral_error,
      self.max_heading_error,
      self.max_overshoot,
    )
    if any(not math.isfinite(value) or value < 0 for value in values):
      raise ValueError(
        "Walk distances, speeds and tolerances must be finite and nonnegative"
      )
    if (
      self.distance_range[0] > self.distance_range[1]
      or min(
        self.walk_speed,
        self.ramp_seconds,
        self.deceleration,
        self.position_tolerance,
        self.goal_heading_tolerance,
        self.settled_speed,
        self.settle_seconds,
      )
      <= 0
    ):
      raise ValueError("Invalid walk distance range, speed or goal tolerance")
    if (
      self.max_lateral_error <= max(self.position_tolerance, self.initial_lateral_error)
      or self.max_heading_error
      <= max(self.goal_heading_tolerance, self.initial_heading_error)
      or self.max_overshoot <= self.position_tolerance
    ):
      raise ValueError("Walk failure limits must exceed goal and initial tolerances")
    return WalkDistanceCommand(self, env)


def distance_command(
  env: ManagerBasedRlEnv, command_name: str = "twist"
) -> WalkDistanceCommand:
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, WalkDistanceCommand)
  return command


def walk_goal(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
  return distance_command(env, command_name).goal


def goal_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = distance_command(env)
  speed = command.robot.data.root_link_lin_vel_w.norm(dim=-1)
  turn_rate = command.robot.data.root_link_ang_vel_w.norm(dim=-1)
  return torch.exp(
    -(command.position_error / 0.05).square()
    - (speed / 0.1).square()
    - (turn_rate / 0.2).square()
  )


def path_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = distance_command(env)
  return torch.exp(
    -(command.lateral_error / 0.05).square() - (command.yaw_error / 0.1).square()
  )


def invalid_walk(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = distance_command(env)
  return (
    (command.remaining < -command.cfg.max_overshoot)
    | (command.lateral_error.abs() > command.cfg.max_lateral_error)
    | (command.yaw_error.abs() > command.cfg.max_heading_error)
  )
