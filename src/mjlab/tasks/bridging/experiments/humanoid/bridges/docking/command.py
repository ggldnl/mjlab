"""Dataset windows and observations for the docking bridge."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DEFAULT_DATASET,
  Dataset,
  Segments,
  load_dataset,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.docking.bridge import (
  CHANNELS,
  ROOT_STATE_DIM,
  Tolerances,
  arrival_score,
  channel_errors,
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

if TYPE_CHECKING:
  import mujoco

  from mjlab.viewer.debug_visualizer import DebugVisualizer


TARGET_COLOR = (1.0, 0.72, 0.2, 0.45)


def _capture_age(
  step: torch.Tensor, capture_step: torch.Tensor, captured: torch.Tensor
) -> torch.Tensor:
  return torch.where(
    captured, (step - capture_step).clamp(min=0), torch.zeros_like(step)
  )


def _rotation_6d(quat: torch.Tensor) -> torch.Tensor:
  matrix = matrix_from_quat(quat)
  return matrix[..., :, :2].transpose(-1, -2).flatten(-2)


def encode_sequence(states: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
  """Express a state sequence in the latest robot heading frame."""
  if states.ndim != 3 or anchor.ndim != 2 or states.shape[0] != anchor.shape[0]:
    raise ValueError(
      "states and anchor must have shapes (batch, time, state) and (batch, state)"
    )
  batch, time, _ = states.shape
  heading = yaw_quat(anchor[:, 3:7])[:, None].expand(batch, time, 4)
  origin = anchor[:, None, 0:3].expand(batch, time, 3)
  return torch.cat(
    (
      quat_apply_inverse(heading, states[..., 0:3] - origin),
      _rotation_6d(quat_mul(quat_conjugate(heading), states[..., 3:7])),
      quat_apply_inverse(heading, states[..., 7:10]),
      quat_apply_inverse(heading, states[..., 10:ROOT_STATE_DIM]),
      states[..., ROOT_STATE_DIM:],
    ),
    dim=-1,
  )


def _rotate_sequence(
  states: torch.Tensor, rotation: torch.Tensor, origin: torch.Tensor
) -> torch.Tensor:
  batch, time, _ = states.shape
  turn = rotation[:, None].expand(batch, time, 4)
  pivot = origin[:, None].expand(batch, time, 3)
  out = states.clone()
  out[..., 0:3] = pivot + quat_apply(turn, states[..., 0:3] - pivot)
  out[..., 3:7] = quat_mul(turn, states[..., 3:7])
  out[..., 7:10] = quat_apply(turn, states[..., 7:10])
  out[..., 10:ROOT_STATE_DIM] = quat_apply(turn, states[..., 10:ROOT_STATE_DIM])
  return out


def _align_target_route(
  targets: torch.Tensor, route_start: torch.Tensor, actual_start: torch.Tensor
) -> torch.Tensor:
  """Place a demonstrated route at an independently sampled start state."""
  rotation = quat_mul(
    yaw_quat(actual_start[:, 3:7]),
    quat_conjugate(yaw_quat(route_start[:, 3:7])),
  )
  out = _rotate_sequence(targets, rotation, route_start[:, 0:3])
  out[..., 0:3] += (actual_start[:, 0:3] - route_start[:, 0:3])[:, None]
  return out


class DockingCommand(CommandTerm):
  """Draw a reachable bridge window and hold its target trajectory."""

  cfg: DockingCommandCfg

  def __init__(self, cfg: DockingCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.num_joints = self.robot.data.joint_pos.shape[1]
    self.state_dim = ROOT_STATE_DIM + 2 * self.num_joints
    self.fps = 1.0 / env.step_dt
    self.tolerances = cfg.tolerances.tensor(self.device)

    offsets = torch.tensor(
      [round(seconds * self.fps) for seconds in cfg.target_offsets_s],
      dtype=torch.long,
      device=self.device,
    )
    if (
      offsets.ndim != 1
      or offsets.numel() < 3
      or not bool((offsets[1:] > offsets[:-1]).all())
    ):
      raise ValueError(
        "target_offsets_s must map to at least three increasing control steps"
      )
    zero = (offsets == 0).nonzero().flatten()
    if zero.numel() != 1:
      raise ValueError("target_offsets_s must contain one zero")
    self.target_offsets = offsets
    self.target_index = int(zero.item())

    self.dataset: Dataset | None = None
    self.windows: Segments | None = None
    if cfg.dataset_path is not None:
      self.dataset = load_dataset(cfg.dataset_path, str(self.device), cfg.split)
      if self.dataset.num_joints != self.num_joints:
        raise ValueError("Dataset and robot joint counts differ")
      if self.dataset.previous_action is None:
        raise ValueError("Docking training needs the dataset previous_action column")
      if not math.isclose(self.dataset.fps, self.fps):
        raise ValueError(
          f"Dataset is {self.dataset.fps:g} Hz but the task is {self.fps:g} Hz"
        )
      minimum = max(1, math.ceil(cfg.duration_s_range[0] * self.fps))
      maximum = max(minimum, math.floor(cfg.duration_s_range[1] * self.fps))
      if minimum < -int(offsets[0]):
        raise ValueError("The shortest bridge must start before the target prehistory")
      future = int(offsets[-1])
      self.windows = self.dataset.segments(
        minimum + future, maximum + future, self.dataset.of(cfg.sources)
      )

    target_len = offsets.numel()
    self.target_sequence = torch.zeros(
      self.num_envs, target_len, self.state_dim, device=self.device
    )
    self.target_actions = torch.zeros(
      self.num_envs, target_len, self.num_joints, device=self.device
    )
    self.history = torch.zeros(
      self.num_envs, cfg.history_steps, self.state_dim, device=self.device
    )
    self.window_steps = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
    self.docking = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.captured = torch.zeros_like(self.docking)
    self.capture_step = torch.full_like(self.window_steps, -1)
    self.start_distance = torch.zeros(self.num_envs, device=self.device)
    self.best_score = torch.zeros(self.num_envs, device=self.device)
    self.final_errors = torch.zeros(self.num_envs, len(CHANNELS), device=self.device)
    self._advanced_at = -1
    self._progress = torch.zeros(self.num_envs, device=self.device)
    self._new_capture = torch.zeros(self.num_envs, device=self.device)
    self._target_ghost: mujoco.MjModel | None = None

    for name in CHANNELS:
      self.metrics[f"error_{name}"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["score"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["captured"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["docking"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["time_left"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def step(self) -> torch.Tensor:
    return self._env.episode_length_buf

  @property
  def target(self) -> torch.Tensor:
    return self.target_sequence[:, self.target_index]

  @property
  def errors(self) -> torch.Tensor:
    return channel_errors(self.state_now(), self.target)

  @property
  def blend(self) -> torch.Tensor:
    age = _capture_age(self.step, self.capture_step, self.captured)
    return torch.where(
      self.captured,
      (age.float() / self.cfg.blend_steps).clamp(max=1.0),
      torch.zeros_like(age, dtype=torch.float32),
    )

  @property
  def handoff(self) -> torch.Tensor:
    return self.captured & (self.step - self.capture_step >= self.cfg.blend_steps)

  @property
  def deadline(self) -> torch.Tensor:
    return (self.step >= self.window_steps) & ~self.captured

  @property
  def entering_action(self) -> torch.Tensor:
    age = _capture_age(self.step, self.capture_step, self.captured)
    index = (self.target_index + age).clamp(max=self.target_actions.shape[1] - 1)
    return self.target_actions.gather(
      1, index[:, None, None].expand(-1, 1, self.num_joints)
    ).squeeze(1)

  @property
  def command(self) -> torch.Tensor:
    here = self.history[:, -1]
    history = encode_sequence(self.history, here).flatten(1)
    target = encode_sequence(self.target_sequence, here).flatten(1)
    remaining = (self.window_steps - self.step).float() / self.fps
    elapsed = self.step.float() / self.window_steps.float()
    error = channel_errors(here, self.target) / self.tolerances
    return torch.cat(
      (
        history,
        target,
        remaining[:, None],
        elapsed[:, None],
        self.docking.float()[:, None],
        error,
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

  def advance(self) -> torch.Tensor:
    if self._advanced_at == self._env.common_step_counter:
      return self._progress
    self._advanced_at = self._env.common_step_counter
    errors = self.errors
    score = arrival_score(errors, self.tolerances)
    better = score > self.best_score
    self._progress = (score - self.best_score).clamp(min=0.0)
    self.best_score = torch.maximum(self.best_score, score)
    self.final_errors = torch.where(better[:, None], errors, self.final_errors)

    self.docking |= (errors <= self.tolerances * self.cfg.capture_scale).all(dim=-1)
    exact = (errors <= self.tolerances).all(dim=-1)
    newly = exact & ~self.captured & (self.step <= self.window_steps)
    self._new_capture = newly.float()
    self.captured |= newly
    self.capture_step = torch.where(newly, self.step, self.capture_step)

    self.metrics["score"] = self.best_score.clone()
    self.metrics["captured"] = self.captured.float()
    self.metrics["docking"] = self.docking.float()
    for index, name in enumerate(CHANNELS):
      self.metrics[f"error_{name}"] = self.final_errors[:, index].clone()
    return self._progress

  @property
  def new_capture(self) -> torch.Tensor:
    self.advance()
    return self._new_capture

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.windows is None or self.dataset is None:
      raise RuntimeError("A dataset is required to sample training windows")
    count = env_ids.numel()
    route_start_rows, _, total_steps, position = self.windows.draw(count)
    future = int(self.target_offsets[-1])
    steps = total_steps - future
    target_position = position + steps
    target_rows = self.windows.order[
      target_position[:, None] + self.target_offsets[None, :]
    ]
    start_pool = self.dataset.of(self.cfg.sources)
    start_rows = start_pool[
      torch.randint(0, start_pool.numel(), (count,), device=self.device)
    ]

    local = torch.rand(count, device=self.device) < self.cfg.docking_probability
    start_rows[local] = target_rows[local, self.target_index]
    local_steps = max(1, round(self.cfg.docking_duration_s * self.fps))
    steps = torch.where(local, torch.full_like(steps, local_steps), steps)

    start = self.dataset.states[start_rows].clone()
    targets = self.dataset.states[target_rows].clone()
    targets[~local] = _align_target_route(
      targets[~local],
      self.dataset.states[route_start_rows[~local]],
      start[~local],
    )
    previous_action = self.dataset.previous_action
    assert previous_action is not None
    actions = previous_action[target_rows].clone()
    initial_action = previous_action[start_rows].clone()

    up = torch.zeros(count, 3, device=self.device)
    up[:, 2] = 1.0
    facing = quat_from_angle_axis(
      torch.rand(count, device=self.device) * 2 * math.pi, up
    )
    rotation = quat_mul(facing, quat_conjugate(yaw_quat(start[:, 3:7])))
    targets = _rotate_sequence(targets, rotation, start[:, 0:3])
    start = _rotate_sequence(start[:, None], rotation, start[:, 0:3]).squeeze(1)

    origin = self._env.scene.env_origins[env_ids]
    shift = origin[:, :2] - start[:, :2]
    start[:, :2] = origin[:, :2]
    targets[..., :2] += shift[:, None]
    start = self._perturb(start, local)
    initial_action[local] = 0.0
    self.place(env_ids, start, targets, actions, steps, local, initial_action)

  def _perturb(self, start: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    count = start.shape[0]
    scale = torch.where(local, torch.ones_like(local, dtype=torch.float32), 0.35)
    out = start.clone()
    out[:, 0:3] += torch.randn_like(out[:, 0:3]) * (0.02 * scale[:, None])
    if bool(local.any()):
      ids = local.nonzero().flatten()
      angle = torch.rand(ids.numel(), device=self.device) * 2 * math.pi
      radius = torch.empty(ids.numel(), device=self.device).uniform_(
        float(self.tolerances[0]) * 1.2,
        float(self.tolerances[0]) * self.cfg.capture_scale * 0.75,
      )
      out[ids, 0] = start[ids, 0] + radius * torch.cos(angle)
      out[ids, 1] = start[ids, 1] + radius * torch.sin(angle)
    axis = torch.randn(count, 3, device=self.device)
    axis /= axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    turn = torch.randn(count, device=self.device) * 0.03 * scale
    out[:, 3:7] = quat_mul(quat_from_angle_axis(turn, axis), out[:, 3:7])
    out[:, 7:10] += torch.randn_like(out[:, 7:10]) * (0.06 * scale[:, None])
    out[:, 10:ROOT_STATE_DIM] += torch.randn_like(out[:, 10:ROOT_STATE_DIM]) * (
      0.12 * scale[:, None]
    )
    out[:, ROOT_STATE_DIM : ROOT_STATE_DIM + self.num_joints] += torch.randn(
      count, self.num_joints, device=self.device
    ) * (0.03 * scale[:, None])
    out[:, ROOT_STATE_DIM + self.num_joints :] += torch.randn(
      count, self.num_joints, device=self.device
    ) * (0.20 * scale[:, None])
    return out

  def place(
    self,
    env_ids: torch.Tensor,
    start: torch.Tensor,
    targets: torch.Tensor,
    target_actions: torch.Tensor,
    steps: torch.Tensor,
    docking: torch.Tensor,
    initial_action: torch.Tensor,
  ) -> None:
    self._open(env_ids, start, targets, target_actions, steps, docking)

    joints = start[:, ROOT_STATE_DIM : ROOT_STATE_DIM + self.num_joints]
    velocity = start[:, ROOT_STATE_DIM + self.num_joints :]
    limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joints = joints.clamp(limits[..., 0], limits[..., 1])
    self.robot.write_joint_state_to_sim(joints, velocity, env_ids=env_ids)
    self.robot.write_root_state_to_sim(start[:, :ROOT_STATE_DIM], env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)
    self._env.action_manager.action[env_ids] = initial_action

  def _open(
    self,
    env_ids: torch.Tensor,
    start: torch.Tensor,
    targets: torch.Tensor,
    target_actions: torch.Tensor,
    steps: torch.Tensor,
    docking: torch.Tensor,
  ) -> None:
    self.target_sequence[env_ids] = targets
    self.target_actions[env_ids] = target_actions
    self.window_steps[env_ids] = steps
    self.docking[env_ids] = docking
    self.captured[env_ids] = False
    self.capture_step[env_ids] = -1
    self.best_score[env_ids] = 0.0
    self.final_errors[env_ids] = 0.0
    self._progress[env_ids] = 0.0
    self._new_capture[env_ids] = 0.0
    self.start_distance[env_ids] = torch.linalg.vector_norm(
      start[:, 0:3] - targets[:, self.target_index, 0:3], dim=-1
    )
    self.history[env_ids] = start[:, None]

  def open_window(
    self,
    env_ids: torch.Tensor,
    targets: torch.Tensor,
    duration_s: torch.Tensor,
    target_actions: torch.Tensor | None = None,
  ) -> None:
    """Aim from the live robot state without teleporting it."""
    count = env_ids.numel()
    expected = (count, self.target_sequence.shape[1], self.state_dim)
    if targets.shape != expected:
      raise ValueError(f"targets must have shape {expected}")
    if duration_s.shape != (count,) or not bool(
      torch.isfinite(duration_s).all() and (duration_s > 0).all()
    ):
      raise ValueError(f"duration_s must contain {count} finite positive values")
    action_shape = (count, self.target_actions.shape[1], self.num_joints)
    if target_actions is None:
      target_actions = torch.zeros(action_shape, device=self.device)
    elif target_actions.shape != action_shape:
      raise ValueError(f"target_actions must have shape {action_shape}")
    start = self.state_now()[env_ids]
    steps = (duration_s * self.fps).round().long().clamp(min=1)
    self._open(
      env_ids,
      start,
      targets,
      target_actions,
      steps,
      torch.zeros(count, dtype=torch.bool, device=self.device),
    )

  def _update_command(self) -> None:
    self.advance()
    self.history = torch.roll(self.history, shifts=-1, dims=1)
    self.history[:, -1] = self.state_now()

  def _update_metrics(self) -> None:
    self.metrics["time_left"] = (self.window_steps - self.step).float() / self.fps

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for batch in visualizer.get_env_indices(self.num_envs):
      self._draw_ghost(visualizer, self.target[batch], batch, "bridge_target")

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
class DockingCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  dataset_path: Path | None = DEFAULT_DATASET
  split: str = "train"
  sources: tuple[str, ...] | None = None
  duration_s_range: tuple[float, float] = (0.4, 1.2)
  history_steps: int = 4
  target_offsets_s: tuple[float, ...] = (
    -0.12,
    -0.08,
    -0.04,
    -0.02,
    0.0,
    0.02,
    0.04,
    0.06,
    0.08,
  )
  docking_probability: float = 0.7
  docking_duration_s: float = 0.20
  residual_scale: float = 0.25
  capture_scale: float = 3.0
  blend_steps: int = 5
  tolerances: Tolerances = field(default_factory=Tolerances)

  def build(self, env: ManagerBasedRlEnv) -> DockingCommand:
    if self.history_steps < 2:
      raise ValueError("history_steps must be at least two")
    if not 0.0 <= self.docking_probability <= 1.0:
      raise ValueError("docking_probability must be in [0, 1]")
    if self.capture_scale <= 1.0 or self.blend_steps < 1:
      raise ValueError("capture_scale and blend_steps are invalid")
    return DockingCommand(self, env)
