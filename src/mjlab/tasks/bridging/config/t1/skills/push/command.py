"""Push goal and the walking pace it implies.

PushCommand    a box destination a whole number of cells ahead, latched once. Each step it
               exposes what is left of the push: remaining distance, lane offset, the push
               axis seen from the robot, and a reference speed.
PushPaceCommand  the robot twist that reference speed implies: forward at the box speed,
               no sidestep, yaw steering back onto the axis. Read by the walking rewards
               only, never observed.

The reference speed is what conditions the policy. A distance set at reset does not tell a
state to action map what to do this step, a speed does. It ramps up from rest, holds
push_speed, and brakes at a constant deceleration so it reaches zero on the target:

    speed = min(push_speed * min(t / ramp_seconds, 1), sqrt(2 * decel * remaining))
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply, wrap_to_pi


class PushCommand(CommandTerm):
  cfg: PushCommandCfg

  def __init__(self, cfg: PushCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.box_name = cfg.box_name
    self.target_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.start_w = torch.zeros_like(self.target_w)
    self.direction_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.direction_w[:, 0] = 1.0
    self.cells = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
    self.elapsed = torch.zeros(self.num_envs, device=self.device)
    self.settled_time = torch.zeros(self.num_envs, device=self.device)
    for name in (
      "position_error",
      "forward_progress",
      "lateral_displacement",
      "tilt_degrees",
      "speed_error",
      "settled_at_goal",
    ):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

  @property
  def box(self):
    return self._env.scene[self.box_name]

  @property
  def left_w(self) -> torch.Tensor:
    return torch.stack((-self.direction_w[:, 1], self.direction_w[:, 0]), dim=-1)

  @property
  def axis_yaw(self) -> torch.Tensor:
    return torch.atan2(self.direction_w[:, 1], self.direction_w[:, 0])

  @property
  def remaining(self) -> torch.Tensor:
    """Distance left along the push axis, negative past the target"""
    offset = (self.target_w - self.box.data.root_link_pos_w)[:, :2]
    return (offset * self.direction_w).sum(-1)

  @property
  def lateral_displacement(self) -> torch.Tensor:
    """Box offset from the push line, positive to the left"""
    offset = (self.box.data.root_link_pos_w - self.start_w)[:, :2]
    return (offset * self.left_w).sum(-1)

  @property
  def position_error(self) -> torch.Tensor:
    return (self.target_w - self.box.data.root_link_pos_w)[:, :2].norm(dim=-1)

  @property
  def tilt(self) -> torch.Tensor:
    quat = self.box.data.root_link_quat_w
    up = quat_apply(quat, quat.new_tensor([0.0, 0.0, 1.0]).expand(self.num_envs, -1))
    return torch.acos(up[:, 2].clamp(-1.0, 1.0))

  @property
  def yaw_error(self) -> torch.Tensor:
    """Box yaw against the push axis, folded to a quarter turn since a cube is symmetric"""
    error = self.box.data.heading_w - self.axis_yaw
    return torch.atan2(torch.sin(4.0 * error), torch.cos(4.0 * error)) / 4.0

  @property
  def speed(self) -> torch.Tensor:
    ramp = (self.elapsed / self.cfg.ramp_seconds).clamp(max=1.0)
    brake = torch.sqrt(2.0 * self.cfg.decel * self.remaining.clamp(min=0.0))
    return torch.minimum(self.cfg.push_speed * ramp, brake)

  @property
  def at_goal(self) -> torch.Tensor:
    return (
      (self.position_error < self.cfg.position_tolerance)
      & (self.lateral_displacement.abs() < self.cfg.lateral_tolerance)
      & (self.tilt < self.cfg.tilt_tolerance)
      & (self.box.data.root_link_lin_vel_w.norm(dim=-1) < self.cfg.settled_speed)
      & (self.box.data.root_link_ang_vel_w.norm(dim=-1) < self.cfg.settled_speed)
    )

  @property
  def command(self) -> torch.Tensor:
    """Remaining distance, lane offset, push axis in the robot heading frame, speed"""
    heading = self._env.scene["robot"].data.heading_w
    axis = wrap_to_pi(self.axis_yaw - heading)
    return torch.stack(
      (
        self.remaining,
        self.lateral_displacement,
        torch.cos(axis),
        torch.sin(axis),
        self.speed,
      ),
      dim=-1,
    )

  def set_goal(
    self,
    box_name: str,
    cells: int,
    direction: torch.Tensor,
    start_xy: torch.Tensor | None = None,
  ) -> None:
    """Latch a destination cells ahead of start_xy, the box position by default"""
    if type(cells) is not int or cells < 1:
      raise ValueError("Push distance must be a positive integer number of cells")
    if direction.shape != (self.num_envs, 2):
      raise ValueError("Push direction must have shape (num_envs, 2)")
    if not torch.allclose(direction.norm(dim=-1), direction.new_ones(self.num_envs)):
      raise ValueError("Push direction must be a unit vector")
    self.box_name = box_name
    self.cells.fill_(cells)
    self.elapsed.zero_()
    self.settled_time.zero_()
    self.start_w.copy_(self.box.data.root_link_pos_w)
    if start_xy is not None:
      self.start_w[:, :2] = start_xy
    self.direction_w.copy_(direction)
    self.target_w.copy_(self.start_w)
    self.target_w[:, :2] += cells * self.cfg.cell_size * direction
    self.time_left.fill_(1.0e9)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Reset events have written poses, flush them before latching the destination
    self._env.scene.write_data_to_sim()
    self._env.sim.forward()
    self.box_name = self.cfg.box_name
    self.elapsed[env_ids] = 0.0
    self.settled_time[env_ids] = 0.0
    self.cells[env_ids] = torch.randint(
      self.cfg.min_cells, self.cfg.max_cells + 1, (len(env_ids),), device=self.device
    )
    # The lane is the grid line through the box cell, not the scattered box itself
    origin = self._env.scene.env_origins[env_ids]
    size = self.cfg.cell_size
    cell = self.box.data.root_link_pos_w[env_ids] - origin
    cell[:, :2] = torch.floor(cell[:, :2] / size) * size + 0.5 * size
    self.start_w[env_ids] = cell + origin
    self.direction_w[env_ids] = torch.tensor([1.0, 0.0], device=self.device)
    self.target_w[env_ids] = self.start_w[env_ids]
    self.target_w[env_ids, :2] += (
      self.cells[env_ids, None] * size * self.direction_w[env_ids]
    )

  def _update_command(self) -> None:
    self.elapsed += self._env.step_dt
    self.settled_time[:] = torch.where(
      self.at_goal, self.settled_time + self._env.step_dt, 0.0
    )

  def _update_metrics(self) -> None:
    velocity = self.box.data.root_link_lin_vel_w[:, :2]
    along = (velocity * self.direction_w).sum(-1)
    self.metrics["position_error"] = self.position_error
    self.metrics["forward_progress"] = self.cells * self.cfg.cell_size - self.remaining
    self.metrics["lateral_displacement"] = self.lateral_displacement.abs()
    self.metrics["tilt_degrees"] = torch.rad2deg(self.tilt)
    self.metrics["speed_error"] = (along - self.speed).abs()
    self.metrics["settled_at_goal"] = (
      self.at_goal & (self.settled_time >= self.cfg.settle_seconds)
    ).float()


@dataclass(kw_only=True)
class PushCommandCfg(CommandTermCfg):
  box_name: str = "box"
  cell_size: float = 1.0
  min_cells: int = 1
  max_cells: int = 4
  push_speed: float = 0.4
  ramp_seconds: float = 1.0
  decel: float = 0.4
  position_tolerance: float = 0.05
  lateral_tolerance: float = 0.03
  tilt_tolerance: float = math.radians(3.0)
  yaw_tolerance: float = 0.04
  max_lateral_displacement: float = 0.15
  max_tilt: float = math.radians(10.0)
  max_yaw: float = 0.3
  max_overshoot: float = 0.15
  settled_speed: float = 0.08
  settle_seconds: float = 0.4

  def build(self, env: ManagerBasedRlEnv) -> PushCommand:
    if self.cell_size <= 0 or not 1 <= self.min_cells <= self.max_cells:
      raise ValueError("Invalid push cell size or distance range")
    if (
      min(
        self.push_speed,
        self.ramp_seconds,
        self.decel,
        self.position_tolerance,
        self.lateral_tolerance,
        self.tilt_tolerance,
        self.yaw_tolerance,
        self.settled_speed,
        self.settle_seconds,
      )
      <= 0
    ):
      raise ValueError("Push speeds, ramps and tolerances must be positive")
    if (
      self.max_lateral_displacement <= self.lateral_tolerance
      or not self.tilt_tolerance < self.max_tilt < math.pi / 2
      or self.max_yaw <= self.yaw_tolerance
    ):
      raise ValueError("Push failure limits must exceed success tolerances")
    return PushCommand(self, env)


class PushPaceCommand(CommandTerm):
  """Twist reference for the walking rewards, derived from the push every step"""

  cfg: PushPaceCommandCfg

  @property
  def command(self) -> torch.Tensor:
    push = self._env.command_manager.get_term(self.cfg.push_name)
    assert isinstance(push, PushCommand)
    speed = push.speed
    # Steer back toward the lane: a box left of the line needs a push angled right
    steer = (self.cfg.lane_gain * push.lateral_displacement).clamp(
      -self.cfg.max_steer, self.cfg.max_steer
    )
    heading = self._env.scene["robot"].data.heading_w
    error = wrap_to_pi(push.axis_yaw - steer - heading)
    turn = (self.cfg.heading_gain * error).clamp(
      -self.cfg.max_turn_rate, self.cfg.max_turn_rate
    )
    turn = torch.where(speed > 0, turn, 0.0)
    return torch.stack((speed, torch.zeros_like(speed), turn), dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    pass

  def _update_command(self) -> None:
    pass

  def _update_metrics(self) -> None:
    pass


@dataclass(kw_only=True)
class PushPaceCommandCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  push_name: str = "push"
  heading_gain: float = 1.5
  max_turn_rate: float = 0.5
  lane_gain: float = 2.0
  max_steer: float = 0.2

  def build(self, env: ManagerBasedRlEnv) -> PushPaceCommand:
    return PushPaceCommand(self, env)
