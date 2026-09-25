"""Short-window trajectory command for the diffusion path tracker."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DEFAULT_DATASET,
  ROOT_STATE_DIM,
  Dataset,
  Segments,
  load_dataset,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
  CHANNELS,
  Tolerances,
  channel_errors,
  score,
  upper_body_mask,
)
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_from_angle_axis,
  quat_mul,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

FUTURE_OFFSETS = (1, 2, 3, 4, 5, 8, 12, 16, 24, 32)
"""Dense immediate reference followed by sparse longer lookahead, in control ticks."""

STATE_HISTORY = 4
"""Physical states exposed to the actor, including the current state."""

TARGET_COLOR = (1.0, 0.72, 0.2, 0.45)
REFERENCE_COLOR = (0.35, 0.6, 1.0, 0.35)


def rotation_6d(quaternion: torch.Tensor) -> torch.Tensor:
  """Continuous two-column rotation representation."""
  matrix = matrix_from_quat(quaternion)
  return matrix[..., :, :2].transpose(-1, -2).flatten(-2)


def reference_features(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
  """Encode reference states as errors in the current heading frame.

  current is (N, D). reference may be (N, D) or (N, H, D).
  """
  if current.ndim != 2 or reference.ndim not in (2, 3):
    raise ValueError("current must be (N, D), reference must be (N, D) or (N, H, D)")
  if current.shape[0] != reference.shape[0] or current.shape[-1] != reference.shape[-1]:
    raise ValueError("current and reference state dimensions differ")
  expanded = current if reference.ndim == 2 else current[:, None].expand_as(reference)
  joints = (current.shape[-1] - ROOT_STATE_DIM) // 2
  q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + joints)
  qd = slice(q.stop, q.stop + joints)
  current_quat = expanded[..., 3:7].contiguous()
  reference_quat = reference[..., 3:7].contiguous()
  heading = yaw_quat(current_quat)
  return torch.cat(
    (
      quat_apply_inverse(heading, reference[..., :3] - expanded[..., :3]),
      rotation_6d(quat_mul(quat_conjugate(current_quat), reference_quat)),
      quat_apply_inverse(heading, reference[..., 7:10] - expanded[..., 7:10]),
      quat_apply_inverse(heading, reference[..., 10:13] - expanded[..., 10:13]),
      reference[..., q] - expanded[..., q],
      reference[..., qd] - expanded[..., qd],
    ),
    dim=-1,
  )


class TrackerCommand(CommandTerm):
  """Draw a physical rollout window and expose its moving reference and endpoint."""

  cfg: TrackerCommandCfg

  def __init__(self, cfg: TrackerCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.num_joints = self.robot.data.joint_pos.shape[1]
    self.state_dim = ROOT_STATE_DIM + 2 * self.num_joints
    self.fps = 1.0 / env.step_dt
    self.upper_body = upper_body_mask(tuple(self.robot.joint_names), self.device)
    self.tolerances = cfg.tolerances.tensor(self.device)
    self.offsets = torch.tensor(
      cfg.future_offsets, device=self.device, dtype=torch.long
    )
    self.post_steps = int(self.offsets.max().item())

    self.dataset: Dataset | None = None
    self.windows: Segments | None = None
    minimum = max(1, math.ceil(cfg.duration_s_range[0] * self.fps))
    self.max_steps = max(minimum, math.floor(cfg.duration_s_range[1] * self.fps))
    if cfg.dataset_path is not None:
      self.dataset = load_dataset(cfg.dataset_path, str(self.device), cfg.split)
      if self.dataset.num_joints != self.num_joints:
        raise ValueError("Dataset and robot joint counts differ")
      if not math.isclose(self.dataset.fps, self.fps):
        raise ValueError("Dataset and environment control rates differ")
      self.windows = self.dataset.segments(
        minimum,
        self.max_steps,
        self.dataset.of(cfg.sources),
        before=STATE_HISTORY - 1,
        after=self.post_steps,
      )

    self.route_rows = torch.zeros(
      self.num_envs,
      self.max_steps + self.post_steps + 1,
      dtype=torch.long,
      device=self.device,
    )
    self.route_start = torch.zeros(self.num_envs, 3, device=self.device)
    self.route_origin = torch.zeros(self.num_envs, 3, device=self.device)
    self.route_rotation = torch.zeros(self.num_envs, 4, device=self.device)
    self.route_rotation[:, 0] = 1.0
    self.target = torch.zeros(self.num_envs, self.state_dim, device=self.device)
    self.window_steps = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
    self.final_errors = torch.zeros(self.num_envs, len(CHANNELS), device=self.device)
    self.final_score = torch.zeros(self.num_envs, device=self.device)
    self.arrived = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.actual_history = torch.zeros(
      self.num_envs, STATE_HISTORY, self.state_dim, device=self.device
    )
    self._history_updated_at = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    self._opened = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._advanced_at = -1

    self._target_ghost: mujoco.MjModel | None = None
    self._reference_ghost: mujoco.MjModel | None = None

  @property
  def step(self) -> torch.Tensor:
    return (self._env.common_step_counter - self._opened).clamp(min=0)

  @property
  def deadline(self) -> torch.Tensor:
    return self.step >= self.window_steps

  @property
  def phase(self) -> torch.Tensor:
    return (self.step.float() / self.window_steps.float()).clamp(0.0, 1.0)

  @property
  def command(self) -> torch.Tensor:
    """Hybrid future, exact endpoint, phase, and time remaining."""
    current = self.state_now()
    future = self.reference_at(self.offsets)
    remaining_steps = (self.window_steps - self.step).clamp(min=0)
    future_time = self.offsets[None].expand(self.num_envs, -1).float() / self.fps
    return torch.cat(
      (
        reference_features(current, future).flatten(1),
        future_time,
        reference_features(current, self.target),
        self.phase[:, None],
        (remaining_steps.float() / self.fps)[:, None],
      ),
      dim=-1,
    )

  def state_now(self) -> torch.Tensor:
    data = self.robot.data
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

  def reference_at(self, offsets: int | torch.Tensor) -> torch.Tensor:
    """Placed reference at one or more offsets from the current control tick."""
    if self.dataset is None:
      if isinstance(offsets, int):
        return self.target
      return self.target[:, None].expand(-1, offsets.numel(), -1)
    if isinstance(offsets, int):
      tick = (self.step + offsets).clamp(max=self.max_steps + self.post_steps)
      rows = self.route_rows.gather(1, tick[:, None]).squeeze(1)
      return self._place_state(self.dataset.states[rows])
    tick = (self.step[:, None] + offsets[None]).clamp(
      max=self.max_steps + self.post_steps
    )
    rows = self.route_rows.gather(1, tick)
    flat = self._place_state(self.dataset.states[rows.flatten()])
    return flat.view(self.num_envs, offsets.numel(), self.state_dim)

  def reference_now(self) -> torch.Tensor:
    return self.reference_at(0)

  def target_errors(self) -> torch.Tensor:
    return channel_errors(self.state_now(), self.target, self.upper_body)

  def tracking_score(self) -> torch.Tensor:
    errors = channel_errors(self.state_now(), self.reference_now(), self.upper_body)
    return score(errors, self.tolerances * self.cfg.tracking_tolerance_scale)

  def endpoint_score(self) -> torch.Tensor:
    """Smooth endpoint objective whose influence grows near B."""
    focus = self.phase.pow(self.cfg.endpoint_phase_power)
    return focus * score(self.target_errors(), self.tolerances)

  def terminal_reward(self) -> torch.Tensor:
    """Score exact arrival once at the requested endpoint tick."""
    if self._advanced_at == self._env.common_step_counter:
      return self.final_score
    self._advanced_at = self._env.common_step_counter
    errors = self.target_errors()
    terminal = self.deadline
    terminal_score = score(errors, self.tolerances) * terminal
    self.final_errors = torch.where(terminal[:, None], errors, self.final_errors)
    self.final_score = torch.where(terminal, terminal_score, self.final_score)
    self.arrived |= terminal & (errors <= self.tolerances).all(dim=-1)
    return terminal_score

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.dataset is None or self.windows is None:
      self._opened[env_ids] = self._env.common_step_counter
      current = self.state_now()[env_ids]
      self.target[env_ids] = current
      self.actual_history[env_ids] = current[:, None]
      self._history_updated_at[env_ids] = self._env.common_step_counter
      self.window_steps[env_ids] = 1
      return
    count = env_ids.numel()
    start_rows, target_rows, steps, positions = self.windows.draw(count)
    start = self.dataset.states[start_rows]
    axis = torch.zeros(count, 3, device=self.device)
    axis[:, 2] = 1.0
    rotation = quat_mul(
      quat_from_angle_axis(torch.rand(count, device=self.device) * 2 * math.pi, axis),
      quat_conjugate(yaw_quat(start[:, 3:7])),
    )
    origin = start[:, :3]
    landing = start[:, :3].clone()
    landing[:, :2] = self._env.scene.env_origins[env_ids, :2]

    self.route_rows[env_ids] = self.windows.path(
      positions, steps, self.max_steps, post_steps=self.post_steps
    )
    self.route_start[env_ids] = origin
    self.route_origin[env_ids] = landing
    self.route_rotation[env_ids] = rotation
    self.window_steps[env_ids] = steps
    self._opened[env_ids] = self._env.common_step_counter
    self.target[env_ids] = self._place_state(self.dataset.states[target_rows], env_ids)
    self.final_errors[env_ids] = 0.0
    self.final_score[env_ids] = 0.0
    self.arrived[env_ids] = False
    history_rows = self.windows.history(positions, STATE_HISTORY - 1)
    history = self._place_state(
      self.dataset.states[history_rows.flatten()], env_ids
    ).view(count, STATE_HISTORY, self.state_dim)
    initial = self._write_initial_state(env_ids, self._place_state(start, env_ids))
    history[:, -1] = initial
    self.actual_history[env_ids] = history
    self._history_updated_at[env_ids] = self._env.common_step_counter

  def _place_state(
    self, state: torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> torch.Tensor:
    if env_ids is None:
      source = self.route_start
      destination = self.route_origin
      rotation = self.route_rotation
    else:
      source = self.route_start[env_ids]
      destination = self.route_origin[env_ids]
      rotation = self.route_rotation[env_ids]
    if state.shape[0] != source.shape[0]:
      repeats = state.shape[0] // source.shape[0]
      source = source[:, None].expand(-1, repeats, -1).reshape(-1, 3)
      destination = destination[:, None].expand(-1, repeats, -1).reshape(-1, 3)
      rotation = rotation[:, None].expand(-1, repeats, -1).reshape(-1, 4)
    out = state.clone()
    out[:, :3] = destination + quat_apply(rotation, state[:, :3] - source)
    out[:, 3:7] = quat_mul(rotation, state[:, 3:7])
    out[:, 7:10] = quat_apply(rotation, state[:, 7:10])
    out[:, 10:13] = quat_apply(rotation, state[:, 10:13])
    return out

  def _write_initial_state(
    self, env_ids: torch.Tensor, state: torch.Tensor
  ) -> torch.Tensor:
    q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + self.num_joints)
    qd = slice(q.stop, q.stop + self.num_joints)
    state = state.clone()
    if self.cfg.initial_position_noise > 0:
      state[:, :3] += torch.empty_like(state[:, :3]).uniform_(
        -self.cfg.initial_position_noise, self.cfg.initial_position_noise
      )
      state[:, 2] += torch.empty_like(state[:, 2]).uniform_(
        -0.5 * self.cfg.initial_position_noise,
        0.5 * self.cfg.initial_position_noise,
      )
    if self.cfg.initial_yaw_noise > 0:
      axis = torch.zeros(len(env_ids), 3, device=self.device)
      axis[:, 2] = 1.0
      angle = torch.empty(len(env_ids), device=self.device).uniform_(
        -self.cfg.initial_yaw_noise, self.cfg.initial_yaw_noise
      )
      state[:, 3:7] = quat_mul(quat_from_angle_axis(angle, axis), state[:, 3:7])
    if self.cfg.initial_velocity_noise > 0:
      state[:, 7:13] += torch.empty_like(state[:, 7:13]).uniform_(
        -self.cfg.initial_velocity_noise, self.cfg.initial_velocity_noise
      )
    if self.cfg.initial_joint_position_noise > 0:
      state[:, q] += torch.empty_like(state[:, q]).uniform_(
        -self.cfg.initial_joint_position_noise,
        self.cfg.initial_joint_position_noise,
      )
    if self.cfg.initial_joint_velocity_noise > 0:
      state[:, qd] += torch.empty_like(state[:, qd]).uniform_(
        -self.cfg.initial_joint_velocity_noise,
        self.cfg.initial_joint_velocity_noise,
      )
    limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = state[:, q].clamp(limits[..., 0], limits[..., 1])
    state[:, q] = joint_pos
    self.robot.write_joint_state_to_sim(joint_pos, state[:, qd], env_ids=env_ids)
    self.robot.write_root_state_to_sim(state[:, :ROOT_STATE_DIM], env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)
    return state

  def _update_metrics(self) -> None:
    pass

  def _update_command(self) -> None:
    now = self._env.common_step_counter
    env_ids = (self._history_updated_at != now).nonzero().flatten()
    if env_ids.numel() == 0:
      return
    self.actual_history[env_ids, :-1] = self.actual_history[env_ids, 1:].clone()
    self.actual_history[env_ids, -1] = self.state_now()[env_ids]
    self._history_updated_at[env_ids] = now

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    if self._reference_ghost is None:
      self._reference_ghost = self._make_ghost(REFERENCE_COLOR)
    reference = self.reference_now()
    for batch in visualizer.get_env_indices(self.num_envs):
      self._draw_ghost(visualizer, self.target[batch], batch, "tracker_target")
      self._draw_ghost(
        visualizer,
        reference[batch],
        batch,
        "tracker_reference",
        self._reference_ghost,
        REFERENCE_COLOR[3],
      )

  def _draw_ghost(
    self,
    visualizer: DebugVisualizer,
    target: torch.Tensor,
    batch: int,
    label: str,
    model: mujoco.MjModel | None = None,
    alpha: float = TARGET_COLOR[3],
  ) -> None:
    if self._target_ghost is None:
      self._target_ghost = self._make_ghost(TARGET_COLOR)
    qpos = np.array(self._env.sim.mj_model.qpos0, dtype=np.float64)
    values = target.cpu().numpy()
    free = self.robot.indexing.free_joint_q_adr.cpu().numpy()
    joints = self.robot.indexing.joint_q_adr.cpu().numpy()
    qpos[free[0:3]] = values[0:3]
    qpos[free[3:7]] = values[3:7]
    qpos[joints] = values[ROOT_STATE_DIM : ROOT_STATE_DIM + self.num_joints]
    visualizer.add_ghost_mesh(
      qpos,
      model=self._target_ghost if model is None else model,
      alpha=alpha,
      label=f"{label}_{batch}",
    )

  def _make_ghost(self, color: tuple[float, float, float, float]) -> mujoco.MjModel:
    ghost = copy.deepcopy(self._env.sim.mj_model)
    robot_geoms = set(self.robot.indexing.geom_ids.tolist())
    for geom in range(ghost.ngeom):
      is_collision = ghost.geom_contype[geom] or ghost.geom_conaffinity[geom]
      if geom in robot_geoms and not is_collision:
        ghost.geom_rgba[geom] = color
      else:
        ghost.geom_rgba[geom, 3] = 0.0
    return ghost


@dataclass(kw_only=True)
class TrackerCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  dataset_path: Path | None = DEFAULT_DATASET
  split: str = "train"
  sources: tuple[str, ...] | None = None
  duration_s_range: tuple[float, float] = (0.3, 2.0)
  future_offsets: tuple[int, ...] = FUTURE_OFFSETS
  tracking_tolerance_scale: float = 3.0
  endpoint_phase_power: float = 4.0
  initial_position_noise: float = 0.02
  initial_yaw_noise: float = 0.05
  initial_velocity_noise: float = 0.10
  initial_joint_position_noise: float = 0.02
  initial_joint_velocity_noise: float = 0.15
  tolerances: Tolerances = field(default_factory=Tolerances)

  def build(self, env: ManagerBasedRlEnv) -> TrackerCommand:
    low, high = self.duration_s_range
    if low <= 0 or high < low or high > 2.0:
      raise ValueError(
        "duration_s_range must be positive, ordered, and at most 2 seconds"
      )
    if not self.future_offsets or any(offset < 1 for offset in self.future_offsets):
      raise ValueError("future_offsets must contain positive control ticks")
    if tuple(sorted(set(self.future_offsets))) != self.future_offsets:
      raise ValueError("future_offsets must be unique and increasing")
    if self.tracking_tolerance_scale < 1.0 or self.endpoint_phase_power <= 0:
      raise ValueError(
        "tracking scale must be >= 1 and endpoint power must be positive"
      )
    noises = (
      self.initial_position_noise,
      self.initial_yaw_noise,
      self.initial_velocity_noise,
      self.initial_joint_position_noise,
      self.initial_joint_velocity_noise,
    )
    if any(value < 0 for value in noises):
      raise ValueError("initial-state noise scales cannot be negative")
    return TrackerCommand(self, env)
