"""Time-aligned trajectory command for the imitation bridge."""

from __future__ import annotations

import copy
import math
import re
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
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_error_magnitude,
  quat_from_angle_axis,
  quat_mul,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

TARGET_COLOR = (1.0, 0.72, 0.2, 0.45)
REFERENCE_COLOR = (0.35, 0.6, 1.0, 0.35)


CHANNELS = (
  "root_pos",
  "root_ori",
  "root_lin_vel",
  "root_ang_vel",
  "lower_joint_pos",
  "lower_joint_vel",
  "upper_joint_pos",
  "upper_joint_vel",
)


def upper_body_mask(names: tuple[str, ...], device: str | torch.device) -> torch.Tensor:
  """Return the arm joint mask. The torso remains in the supporting chain."""
  pattern = re.compile(r"shoulder|elbow|wrist")
  return torch.tensor(
    [bool(pattern.search(name)) for name in names], device=device, dtype=torch.bool
  )


@dataclass(frozen=True, kw_only=True)
class Tolerances:
  """Terminal limits in physical units, ordered like CHANNELS."""

  root_pos: float = 0.05
  root_ori: float = 0.05
  root_lin_vel: float = 0.15
  root_ang_vel: float = 0.30
  lower_joint_pos: float = 0.08
  lower_joint_vel: float = 0.80
  upper_joint_pos: float = 0.05
  upper_joint_vel: float = 0.75

  def tensor(self, device: str | torch.device) -> torch.Tensor:
    values = tuple(getattr(self, name) for name in CHANNELS)
    if any(not math.isfinite(value) or value <= 0 for value in values):
      raise ValueError("Tolerances must be finite and positive")
    return torch.tensor(values, device=device)


def _worst(errors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  if not bool(mask.any()):
    return errors.new_zeros(errors.shape[0])
  return errors[:, mask].amax(dim=-1)


def channel_errors(
  actual: torch.Tensor, target: torch.Tensor, upper_body: torch.Tensor
) -> torch.Tensor:
  """Return one physical error for each terminal channel."""
  if actual.shape != target.shape or actual.ndim != 2:
    raise ValueError("actual and target must have matching shape (batch, state)")
  joints = upper_body.numel()
  if actual.shape[1] != ROOT_STATE_DIM + 2 * joints:
    raise ValueError("state and joint mask sizes differ")
  q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + joints)
  qd = slice(q.stop, q.stop + joints)
  position = (actual[:, q] - target[:, q]).abs()
  velocity = (actual[:, qd] - target[:, qd]).abs()
  lower = ~upper_body
  return torch.stack(
    (
      torch.linalg.vector_norm(actual[:, :3] - target[:, :3], dim=-1),
      quat_error_magnitude(actual[:, 3:7], target[:, 3:7]),
      torch.linalg.vector_norm(actual[:, 7:10] - target[:, 7:10], dim=-1),
      torch.linalg.vector_norm(actual[:, 10:13] - target[:, 10:13], dim=-1),
      _worst(position, lower),
      _worst(velocity, lower),
      _worst(position, upper_body),
      _worst(velocity, upper_body),
    ),
    dim=-1,
  )


def score(errors: torch.Tensor, tolerances: torch.Tensor) -> torch.Tensor:
  """Smooth score whose worst normalized channel dominates."""
  reach = torch.log1p(errors / tolerances)
  return 1.0 / (1.0 + 0.7 * reach.amax(dim=-1) + 0.3 * reach.mean(dim=-1))


def _rotation_6d(quat: torch.Tensor) -> torch.Tensor:
  matrix = matrix_from_quat(quat)
  return matrix[..., :, :2].transpose(-1, -2).flatten(-2)


class ImitationCommand(CommandTerm):
  """Sample one demonstrated route and expose its terminal delta and clock."""

  cfg: ImitationCommandCfg

  def __init__(self, cfg: ImitationCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.num_joints = self.robot.data.joint_pos.shape[1]
    self.state_dim = ROOT_STATE_DIM + 2 * self.num_joints
    self.fps = 1.0 / env.step_dt
    self.upper_body = upper_body_mask(tuple(self.robot.joint_names), self.device)
    self.tolerances = cfg.tolerances.tensor(self.device)

    self.dataset: Dataset | None = None
    self.windows: Segments | None = None
    self.max_steps = max(1, math.floor(cfg.duration_s_range[1] * self.fps))
    if cfg.dataset_path is not None:
      self.dataset = load_dataset(cfg.dataset_path, str(self.device), cfg.split)
      if self.dataset.num_joints != self.num_joints:
        raise ValueError("Dataset and robot joint counts differ")
      if not math.isclose(self.dataset.fps, self.fps):
        raise ValueError("Dataset and environment control rates differ")
      minimum = max(1, math.ceil(cfg.duration_s_range[0] * self.fps))
      self.max_steps = max(minimum, self.max_steps)
      self.windows = self.dataset.segments(
        minimum, self.max_steps, self.dataset.of(cfg.sources)
      )

    self.route_rows = torch.zeros(
      self.num_envs, self.max_steps + 1, dtype=torch.long, device=self.device
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
    self._opened = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._advanced_at = -1
    self.active = False
    self.error_names = CHANNELS

    self._target_ghost: mujoco.MjModel | None = None
    self._reference_ghost: mujoco.MjModel | None = None

    for name in CHANNELS:
      self.metrics[f"error_{name}"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["terminal_score"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["arrived"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def step(self) -> torch.Tensor:
    return (self._env.common_step_counter - self._opened).clamp(min=0)

  @property
  def deadline(self) -> torch.Tensor:
    return self.active & (self.step >= self.window_steps)

  @property
  def handoff(self) -> torch.Tensor:
    return self.deadline

  @property
  def command(self) -> torch.Tensor:
    """Terminal-state deltas in the current robot frame plus a bounded clock."""
    current = self.state_now()
    q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + self.num_joints)
    qd = slice(q.stop, q.stop + self.num_joints)
    heading = yaw_quat(current[:, 3:7])
    phase = self.step.float() / self.window_steps.float()
    return torch.cat(
      (
        quat_apply_inverse(heading, self.target[:, :3] - current[:, :3]),
        _rotation_6d(quat_mul(quat_conjugate(current[:, 3:7]), self.target[:, 3:7])),
        quat_apply_inverse(current[:, 3:7], self.target[:, 7:10] - current[:, 7:10]),
        quat_apply_inverse(current[:, 3:7], self.target[:, 10:13] - current[:, 10:13]),
        self.target[:, q] - current[:, q],
        self.target[:, qd] - current[:, qd],
        (1.0 - phase).clamp(min=0.0).unsqueeze(-1),
        phase.clamp(max=1.0).unsqueeze(-1),
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

  def reference_now(self) -> torch.Tensor:
    if self.dataset is None:
      return self.target
    tick = self.step.clamp(max=self.max_steps)
    rows = self.route_rows.gather(1, tick[:, None]).squeeze(1)
    return self._place_state(self.dataset.states[rows])

  def target_errors(self) -> torch.Tensor:
    return channel_errors(self.state_now(), self.target, self.upper_body)

  def aim(self, target: torch.Tensor) -> None:
    """Show a target without starting the bridge clock."""
    if target.shape != self.target.shape:
      raise ValueError("target must have shape (num_envs, state)")
    self.target[:] = target

  def open_window(
    self,
    env_ids: torch.Tensor,
    target: torch.Tensor,
    duration_s: torch.Tensor,
  ) -> None:
    """Start a live transition toward an externally supplied target."""
    if target.shape != self.target[env_ids].shape:
      raise ValueError("target must have shape (selected_envs, state)")
    if duration_s.shape != env_ids.shape:
      raise ValueError("duration_s must have shape (selected_envs,)")
    low, high = self.cfg.duration_s_range
    if bool(((duration_s < low) | (duration_s > high)).any()):
      raise ValueError(f"duration_s must be between {low:g} and {high:g}")
    self._opened[env_ids] = self._env.common_step_counter
    self.target[env_ids] = target
    self.window_steps[env_ids] = torch.clamp(
      torch.round(duration_s * self.fps).long(), min=1
    )
    self.final_errors[env_ids] = 0.0
    self.final_score[env_ids] = 0.0
    self.arrived[env_ids] = False
    self.active = True

  def stop(self) -> None:
    self.active = False

  def tracking_score(self) -> torch.Tensor:
    errors = channel_errors(self.state_now(), self.reference_now(), self.upper_body)
    return score(errors, self.tolerances * self.cfg.tracking_tolerance_scale)

  def terminal_reward(self) -> torch.Tensor:
    """Score the target exactly once, at the requested control tick."""
    if self._advanced_at == self._env.common_step_counter:
      return self.final_score
    self._advanced_at = self._env.common_step_counter
    errors = self.target_errors()
    terminal = self.deadline
    terminal_score = score(errors, self.tolerances) * terminal
    self.final_errors = torch.where(terminal[:, None], errors, self.final_errors)
    self.final_score = torch.where(terminal, terminal_score, self.final_score)
    self.arrived |= terminal & (errors <= self.tolerances).all(dim=-1)
    self.metrics["terminal_score"] = self.final_score
    self.metrics["arrived"] = self.arrived.float()
    for index, name in enumerate(CHANNELS):
      self.metrics[f"error_{name}"] = self.final_errors[:, index]
    return terminal_score

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.dataset is None or self.windows is None:
      self._opened[env_ids] = self._env.common_step_counter
      self.target[env_ids] = self.state_now()[env_ids]
      self.window_steps[env_ids] = 1
      self.active = False
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

    self.route_rows[env_ids] = self.windows.path(positions, steps, self.max_steps)
    self.route_start[env_ids] = origin
    self.route_origin[env_ids] = landing
    self.route_rotation[env_ids] = rotation
    self.window_steps[env_ids] = steps
    self._opened[env_ids] = self._env.common_step_counter
    self.target[env_ids] = self._place_state(self.dataset.states[target_rows], env_ids)
    self.final_errors[env_ids] = 0.0
    self.final_score[env_ids] = 0.0
    self.arrived[env_ids] = False
    self.active = True

    initial = self._place_state(start, env_ids)
    self._write_initial_state(env_ids, initial)

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
    out = state.clone()
    out[:, :3] = destination + quat_apply(rotation, state[:, :3] - source)
    out[:, 3:7] = quat_mul(rotation, state[:, 3:7])
    out[:, 7:10] = quat_apply(rotation, state[:, 7:10])
    out[:, 10:13] = quat_apply(rotation, state[:, 10:13])
    return out

  def _write_initial_state(self, env_ids: torch.Tensor, state: torch.Tensor) -> None:
    q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + self.num_joints)
    qd = slice(q.stop, q.stop + self.num_joints)
    joint_pos = state[:, q]
    limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    self.robot.write_joint_state_to_sim(joint_pos, state[:, qd], env_ids=env_ids)
    self.robot.write_root_state_to_sim(state[:, :ROOT_STATE_DIM], env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _update_metrics(self) -> None:
    pass

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    if self._reference_ghost is None:
      self._reference_ghost = self._make_target_ghost(REFERENCE_COLOR)
    reference = self.reference_now()
    for batch in visualizer.get_env_indices(self.num_envs):
      self._draw_ghost(visualizer, self.target[batch], batch, "bridge_target")
      self._draw_ghost(
        visualizer,
        reference[batch],
        batch,
        "bridge_reference",
        model=self._reference_ghost,
        alpha=REFERENCE_COLOR[3],
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
      self._target_ghost = self._make_target_ghost()
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

  def _make_target_ghost(
    self, color: tuple[float, float, float, float] = TARGET_COLOR
  ) -> mujoco.MjModel:
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
class ImitationCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  dataset_path: Path | None = DEFAULT_DATASET
  split: str = "train"
  sources: tuple[str, ...] | None = None
  duration_s_range: tuple[float, float] = (0.3, 1.2)
  tracking_tolerance_scale: float = 3.0
  tolerances: Tolerances = field(default_factory=Tolerances)

  def build(self, env: ManagerBasedRlEnv) -> ImitationCommand:
    low, high = self.duration_s_range
    if low <= 0 or high < low:
      raise ValueError("duration_s_range must be positive and ordered")
    if self.tracking_tolerance_scale < 1.0:
      raise ValueError("tracking_tolerance_scale must be at least one")
    return ImitationCommand(self, env)
