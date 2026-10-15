"""Turn by a relative angle in place."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
  import viser

  from mjlab.viewer.debug_visualizer import DebugVisualizer


class TurnCommand(UniformVelocityCommand):
  cfg: "TurnCommandCfg"

  def __init__(self, cfg: "TurnCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.turn_rate = torch.full((self.num_envs,), cfg.max_turn_rate, device=self.device)
    self.metrics["heading_error"] = torch.zeros(self.num_envs, device=self.device)
    self._heading_gui_enabled: viser.GuiCheckboxHandle | None = None
    self._heading_gui_target: viser.GuiSliderHandle | None = None
    self._heading_gui_rate: viser.GuiSliderHandle | None = None
    self._heading_gui_status: viser.GuiMarkdownHandle | None = None
    self._heading_gui_get_env_idx: Callable[[], int] | None = None
    self._heading_gui_state: tuple[int, float, float] | None = None

  @property
  def remaining(self) -> torch.Tensor:
    return wrap_to_pi(self.heading_target - self.robot.data.heading_w)

  def set_angle(
    self,
    angle: torch.Tensor,
    env_ids: torch.Tensor | None = None,
    max_turn_rate: float | None = None,
  ) -> None:
    ids = slice(None) if env_ids is None else env_ids
    if angle.shape != self.heading_target[ids].shape or not bool(
      torch.isfinite(angle).all()
    ):
      raise ValueError("Turn angles must be a finite value per environment")
    rate = self.cfg.max_turn_rate if max_turn_rate is None else max_turn_rate
    if not math.isfinite(rate) or not 0 < rate <= self.cfg.max_turn_rate:
      raise ValueError("Turn rate must be positive and within the configured maximum")
    self.heading_target[ids] = self.robot.data.heading_w[ids] + angle
    self.turn_rate[ids] = rate
    self.is_standing_env[ids] = False
    self.time_left[ids] = 1.0e9
    self._update_command()
    observations = getattr(self._env, "observation_manager", None)
    if observations is not None:
      observations._obs_buffer = None

  def set_heading(
    self,
    heading: torch.Tensor,
    env_ids: torch.Tensor | None = None,
    max_turn_rate: float | None = None,
  ) -> None:
    """Latch a world heading in radians using the shortest rotation"""
    ids = slice(None) if env_ids is None else env_ids
    if heading.shape != self.heading_target[ids].shape or not bool(
      torch.isfinite(heading).all()
    ):
      raise ValueError("Turn headings must be a finite value per environment")
    self.set_angle(
      wrap_to_pi(heading - self.robot.data.heading_w[ids]), env_ids, max_turn_rate
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Flush reset poses before latching heading
    self._env.scene.write_data_to_sim()
    self._env.sim.forward()
    angle = torch.empty(len(env_ids), device=self.device).uniform_(
      *self.cfg.angle_range
    )
    hold = torch.rand(len(env_ids), device=self.device) < self.cfg.rel_standing_envs
    angle[hold] = 0.0
    self.set_angle(angle, env_ids)
    self.turn_rate[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
      self.cfg.min_turn_rate, self.cfg.max_turn_rate
    )
    slow = torch.rand(len(env_ids), device=self.device) < 0.5
    self.turn_rate[env_ids[slow]] = self.cfg.min_turn_rate
    self.time_left[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
      *self.cfg.resampling_time_range
    )
    self._update_command()

  def _update_command(self) -> None:
    self.vel_command_b.zero_()
    rate = self.turn_rate.clamp(max=self.cfg.max_turn_rate)
    error = self.remaining
    self.is_standing_env &= error.abs() < 2 * self.cfg.heading_tolerance
    self.is_standing_env |= error.abs() <= self.cfg.heading_tolerance
    self.vel_command_b[:, 2] = torch.where(
      self.is_standing_env,
      0.0,
      torch.clamp(self.cfg.heading_control_stiffness * error, min=-rate, max=rate),
    )

  def _update_metrics(self) -> None:
    super()._update_metrics()
    self.metrics["heading_error"].copy_(self.remaining.abs())

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Control world heading in degrees and turn speed in rad/s."""
    with server.gui.add_folder(name.capitalize()):
      self._heading_gui_enabled = server.gui.add_checkbox(
        "Manual heading", initial_value=False
      )
      self._heading_gui_target = server.gui.add_slider(
        "Target heading (deg)", min=-180.0, max=180.0, step=1.0, initial_value=0.0
      )
      self._heading_gui_rate = server.gui.add_slider(
        "Turn speed limit (rad/s)",
        min=min(0.05, self.cfg.max_turn_rate),
        max=self.cfg.max_turn_rate,
        step=0.01,
        initial_value=min(0.2, self.cfg.max_turn_rate),
      )
      self._heading_gui_status = server.gui.add_markdown("")
    self._heading_gui_get_env_idx = get_env_idx
    if on_change is not None:
      for handle in (
        self._heading_gui_enabled,
        self._heading_gui_target,
        self._heading_gui_rate,
      ):
        handle.on_update(lambda _: on_change())

  def compute(self, dt: float) -> None:
    state = None
    if self._heading_gui_enabled is not None and self._heading_gui_enabled.value:
      assert self._heading_gui_get_env_idx is not None
      assert self._heading_gui_target is not None
      assert self._heading_gui_rate is not None
      state = (
        self._heading_gui_get_env_idx(),
        math.radians(self._heading_gui_target.value),
        self._heading_gui_rate.value,
      )
    previous = self._heading_gui_state
    if previous is not None and (state is None or previous[0] != state[0]):
      self.time_left[previous[0]] = 0.0
    if state is not None:
      idx, heading, rate = state
      if state != previous:
        self.is_standing_env[idx] = False
      self.heading_target[idx] = heading
      self.turn_rate[idx] = rate
      self.time_left[idx] = 1.0e9
    self._heading_gui_state = state
    super().compute(dt)
    if self._heading_gui_status is not None:
      assert self._heading_gui_get_env_idx is not None
      idx = self._heading_gui_get_env_idx()
      heading = math.degrees(float(self.robot.data.heading_w[idx]))
      target = math.degrees(float(wrap_to_pi(self.heading_target[idx])))
      error = math.degrees(float(self.remaining[idx]))
      mode = (
        "Manual heading"
        if state is not None
        else f"Random commands · next in {float(self.time_left[idx]):.1f} s"
      )
      self._heading_gui_status.content = (
        f"{mode}\n\nCurrent: {heading:.1f}° · Target: {target:.1f}°"
        f" · Remaining: {error:.1f}°\n\n"
        f"Turn speed limit: {float(self.turn_rate[idx]):.2f} rad/s"
      )

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for idx in visualizer.get_env_indices(self.num_envs):
      start = self.robot.data.root_link_pos_w[idx].cpu().numpy().copy()
      start[2] += self.cfg.viz.z_offset * self.cfg.viz.scale
      for heading, color, label in (
        (self.heading_target[idx], (1.0, 0.5, 0.0, 1.0), "Target heading"),
        (self.robot.data.heading_w[idx], (0.0, 0.8, 1.0, 1.0), "Current heading"),
      ):
        angle = float(heading)
        direction = np.array([math.cos(angle), math.sin(angle), 0.0])
        visualizer.add_arrow(
          start,
          start + 0.5 * self.cfg.viz.scale * direction,
          color=color,
          label=label,
        )


@dataclass(kw_only=True)
class TurnCommandCfg(UniformVelocityCommandCfg):
  angle_range: tuple[float, float] = (-torch.pi, torch.pi)
  min_turn_rate: float = 0.2
  max_turn_rate: float = 0.5
  heading_tolerance: float = 0.05

  def build(self, env: ManagerBasedRlEnv) -> TurnCommand:
    if not -torch.pi <= self.angle_range[0] <= self.angle_range[1] <= torch.pi:
      raise ValueError("Turn angles must be ordered and within [-pi, pi]")
    if not math.isfinite(self.max_turn_rate) or self.max_turn_rate <= 0:
      raise ValueError("Turn rate must be positive")
    if (
      not math.isfinite(self.min_turn_rate)
      or not 0 < self.min_turn_rate <= self.max_turn_rate
    ):
      raise ValueError("Minimum turn rate must be positive and within the maximum")
    if (
      not math.isfinite(self.heading_control_stiffness)
      or self.heading_control_stiffness <= 0
    ):
      raise ValueError("Heading control stiffness must be positive")
    if (
      not math.isfinite(self.heading_tolerance)
      or not 0 <= self.heading_tolerance < math.pi
    ):
      raise ValueError("Heading tolerance must be within [0, pi)")
    return TurnCommand(self, env)


def turn_command(env, command_name: str = "twist") -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, TurnCommand)
  return command.remaining[:, None]


def heading_reward(env, std: float = 0.25) -> torch.Tensor:
  error = turn_command(env)[:, 0]
  coarse = 0.5 * (1.0 + torch.cos(error))
  return 0.5 * (coarse + torch.exp(-torch.square(error / std)))


def turn_rate_penalty(env) -> torch.Tensor:
  command = env.command_manager.get_term("twist")
  assert isinstance(command, TurnCommand)
  actual = command.robot.data.root_link_ang_vel_b[:, 2]
  return torch.square(actual - command.command[:, 2])


def turn_rate_reward(env, std: float) -> torch.Tensor:
  return torch.exp(-turn_rate_penalty(env) / std**2)
