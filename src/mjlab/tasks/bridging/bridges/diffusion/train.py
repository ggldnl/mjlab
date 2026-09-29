"""Improve the diffusion planner with trajectories executed by a frozen tracker."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import torch
from rsl_rl.env import VecEnv

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.bridging.bridges.dataset.dataset import (
  ROOT_STATE_DIM,
  Dataset,
)
from mjlab.tasks.bridging.bridges.diffusion.config import (
  improvement_experiment,
  improvement_task_id,
  motion_patterns,
)
from mjlab.tasks.bridging.bridges.diffusion.dataset.motions import (
  Normalizer,
  Windows,
  bridge_mask,
  encode,
  load_motions,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.bridge import (
  DiffusionBridge,
  checkpoint_metadata,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.model import Denoiser, ModelCfg
from mjlab.tasks.bridging.bridges.diffusion.planner.process import Diffusion, ProcessCfg
from mjlab.tasks.bridging.bridges.diffusion.tracker import tracker_ppo_runner_cfg
from mjlab.tasks.bridging.bridges.diffusion.tracker.command import (
  STATE_HISTORY,
  TrackerCommand,
  TrackerCommandCfg,
)
from mjlab.tasks.bridging.bridges.diffusion.tracker.env_cfg import (
  COMMAND,
  tracker_env_cfg,
)
from mjlab.tasks.bridging.bridges.imitation.command import channel_errors, score
from mjlab.tasks.bridging.config import ROBOTS
from mjlab.tasks.registry import register_mjlab_task
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_from_angle_axis,
  quat_mul,
  yaw_quat,
)

TRAIN_TASK_ID = improvement_task_id("g1")
TRAIN_EXPERIMENT = improvement_experiment("g1")


def _yaw(quaternion: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quaternion.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def _repeat_to_batch(value: torch.Tensor, count: int) -> torch.Tensor:
  if value.shape[0] == count:
    return value
  if value.shape[0] == 0:
    raise ValueError("Cannot pad an empty evaluation batch")
  repeats = math.ceil(count / value.shape[0])
  return value.repeat((repeats,) + (1,) * (value.ndim - 1))[:count]


@dataclass(frozen=True)
class PairCfg:
  min_steps: int = 15
  max_steps: int = 60
  duration_scale_range: tuple[float, float] = (0.95, 1.05)
  empirical_quantile: float = 0.995
  limit_slack: float = 1.25
  position_jitter: float = 0.05
  heading_jitter: float = 0.10
  max_attempts: int = 20

  def __post_init__(self) -> None:
    low, high = self.duration_scale_range
    if self.min_steps < 2 or self.max_steps < self.min_steps:
      raise ValueError("Pair duration bounds are invalid")
    if low <= 0 or high < low:
      raise ValueError("duration_scale_range must be positive and ordered")
    if not 0.5 < self.empirical_quantile < 1 or self.limit_slack < 1:
      raise ValueError("Empirical feasibility settings are invalid")
    if min(self.position_jitter, self.heading_jitter) < 0 or self.max_attempts < 1:
      raise ValueError("Pair jitter must be nonnegative and attempts positive")


@dataclass(frozen=True)
class PairBatch:
  history: torch.Tensor
  target: torch.Tensor
  duration: torch.Tensor
  start_source: torch.Tensor
  target_source: torch.Tensor


class CrossTrajectoryPairs:
  """Draw endpoints from different motion clips and place B feasibly."""

  def __init__(
    self,
    data: Dataset,
    history: int,
    post_steps: int,
    cfg: PairCfg,
    anchor_rows: torch.Tensor | None = None,
  ) -> None:
    self.data = data
    self.history = history
    self.post_steps = post_steps
    self.cfg = cfg
    segments = data.segments(1, 1)
    self.order = segments.order
    self.starts = data.segments(1, 1, start_rows=anchor_rows, before=history - 1).starts
    self.targets = data.segments(1, 1, start_rows=anchor_rows, after=post_steps).starts
    target_trajectories = data.trajectory[self.order[self.targets]].unique()
    if target_trajectories.numel() < 2:
      start_trajectories = data.trajectory[self.order[self.starts]]
      self.starts = self.starts[start_trajectories != target_trajectories[0]]
    if self.starts.numel() == 0 or target_trajectories.numel() == 0:
      raise ValueError("Planner improvement needs at least two motion clips")

    left = self.order[segments.starts]
    right = self.order[segments.starts + 1]
    first = data.states[left]
    second = data.states[right]
    joints = data.num_joints
    qd = slice(ROOT_STATE_DIM + joints, ROOT_STATE_DIM + 2 * joints)
    quantile = cfg.empirical_quantile
    slack = cfg.limit_slack
    self.max_joint_speed = max(
      float(torch.quantile(data.states[:, qd].abs().flatten(), quantile)) * slack,
      1.0,
    )
    self.max_root_acceleration = max(
      float(
        torch.quantile(
          ((second[:, 7:10] - first[:, 7:10]) * data.fps).norm(dim=-1), quantile
        )
      )
      * slack,
      1.0,
    )
    self.max_root_angular_acceleration = max(
      float(
        torch.quantile(
          ((second[:, 10:13] - first[:, 10:13]) * data.fps).norm(dim=-1), quantile
        )
      )
      * slack,
      1.0,
    )
    self.max_vertical_speed = max(
      float(torch.quantile(data.states[:, 9].abs(), quantile)) * slack, 0.25
    )

  def _draw_indexes(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = self.data.states.device
    a = self.starts[torch.randint(self.starts.numel(), (count,), device=device)]
    b = self.targets[torch.randint(self.targets.numel(), (count,), device=device)]
    a_trajectory = self.data.trajectory[self.order[a]]
    b_trajectory = self.data.trajectory[self.order[b]]
    same = a_trajectory == b_trajectory
    for _ in range(self.cfg.max_attempts):
      if not bool(same.any()):
        break
      b[same] = self.targets[
        torch.randint(self.targets.numel(), (int(same.sum()),), device=device)
      ]
      b_trajectory = self.data.trajectory[self.order[b]]
      same = a_trajectory == b_trajectory
    if bool(same.any()):
      raise RuntimeError("Could not draw endpoints from different motion clips")
    return a, b

  def _place(
    self, history: torch.Tensor, target: torch.Tensor, duration: torch.Tensor
  ) -> torch.Tensor:
    count = history.shape[0]
    device = history.device
    a = history[:, -1]
    b = target[:, 0]
    seconds = duration.float() / self.data.fps
    desired_yaw = _yaw(a[:, 3:7]) + 0.5 * (a[:, 12] + b[:, 12]) * seconds
    desired_yaw += torch.empty(count, device=device).uniform_(
      -self.cfg.heading_jitter, self.cfg.heading_jitter
    )
    angle = desired_yaw - _yaw(b[:, 3:7])
    axis = torch.zeros(count, 3, device=device)
    axis[:, 2] = 1.0
    rotation = quat_from_angle_axis(angle, axis)
    repeated = rotation[:, None].expand(-1, target.shape[1], -1).reshape(-1, 4)

    out = target.clone()
    relative = target[..., :3] - b[:, None, :3]
    out[..., :3] = quat_apply(repeated, relative.reshape(-1, 3)).view_as(relative)
    rotated_linear = quat_apply(rotation, b[:, 7:10])
    displacement = 0.5 * (a[:, 7:10] + rotated_linear) * seconds[:, None]
    jitter = torch.empty(count, 3, device=device).uniform_(
      -self.cfg.position_jitter, self.cfg.position_jitter
    )
    jitter[:, 2] = 0.0
    jitter = quat_apply(yaw_quat(a[:, 3:7]), jitter)
    destination = a[:, :3] + displacement + jitter
    destination[:, 2] = b[:, 2]
    out[..., :3] += destination[:, None]
    out[..., 3:7] = quat_mul(repeated, target[..., 3:7].reshape(-1, 4)).view_as(
      target[..., 3:7]
    )
    out[..., 7:10] = quat_apply(repeated, target[..., 7:10].reshape(-1, 3)).view_as(
      target[..., 7:10]
    )
    out[..., 10:13] = quat_apply(repeated, target[..., 10:13].reshape(-1, 3)).view_as(
      target[..., 10:13]
    )
    return out

  def _feasible(
    self, history: torch.Tensor, target: torch.Tensor, duration: torch.Tensor
  ) -> torch.Tensor:
    a = history[:, -1]
    b = target[:, 0]
    seconds = duration.float() / self.data.fps
    joints = self.data.num_joints
    q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + joints)
    joint_rate = (b[:, q] - a[:, q]).abs().amax(dim=-1) / seconds
    root_acceleration = (b[:, 7:10] - a[:, 7:10]).norm(dim=-1) / seconds
    angular_acceleration = (b[:, 10:13] - a[:, 10:13]).norm(dim=-1) / seconds
    vertical_speed = (b[:, 2] - a[:, 2]).abs() / seconds
    return (
      (joint_rate <= self.max_joint_speed)
      & (root_acceleration <= self.max_root_acceleration)
      & (angular_acceleration <= self.max_root_angular_acceleration)
      & (vertical_speed <= self.max_vertical_speed)
    )

  def draw(self, count: int) -> PairBatch:
    device = self.data.states.device
    histories: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    durations: list[torch.Tensor] = []
    starts: list[torch.Tensor] = []
    ends: list[torch.Tensor] = []
    remaining = count
    history_offsets = torch.arange(1 - self.history, 1, device=device)
    target_offsets = torch.arange(self.post_steps + 1, device=device)
    for _ in range(self.cfg.max_attempts):
      if remaining == 0:
        break
      trial = max(remaining * 2, 32)
      a, b = self._draw_indexes(trial)
      history_rows = self.order[a[:, None] + history_offsets]
      target_rows = self.order[b[:, None] + target_offsets]
      history = self.data.states[history_rows]
      target = self.data.states[target_rows]
      base = torch.randint(
        self.cfg.min_steps, self.cfg.max_steps + 1, (trial,), device=device
      )
      low, high = self.cfg.duration_scale_range
      scale = torch.empty(trial, device=device).uniform_(low, high)
      duration = (
        (base.float() * scale)
        .round()
        .long()
        .clamp(self.cfg.min_steps, self.cfg.max_steps)
      )
      target = self._place(history, target, duration)
      keep = self._feasible(history, target, duration).nonzero().flatten()[:remaining]
      if keep.numel() == 0:
        continue
      histories.append(history[keep])
      targets.append(target[keep])
      durations.append(duration[keep])
      starts.append(self.data.skill[history_rows[keep, -1]])
      ends.append(self.data.skill[target_rows[keep, 0]])
      remaining -= keep.numel()
    if remaining:
      raise RuntimeError(
        f"Could only manufacture {count - remaining}/{count} feasible cross-clip pairs"
      )
    return PairBatch(
      torch.cat(histories),
      torch.cat(targets),
      torch.cat(durations),
      torch.cat(starts),
      torch.cat(ends),
    )


class EvaluationTrackerCommand(TrackerCommand):
  """Run an exact externally supplied route once in every environment."""

  def __init__(self, cfg: EvaluationTrackerCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.routes = torch.zeros(
      self.num_envs,
      self.max_steps + self.post_steps + 1,
      self.state_dim,
      device=self.device,
    )
    self.batch_history = torch.zeros(
      self.num_envs, STATE_HISTORY, self.state_dim, device=self.device
    )
    self._batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

  def set_batch(
    self, history: torch.Tensor, routes: torch.Tensor, duration: torch.Tensor
  ) -> None:
    expected = (self.num_envs, self.max_steps + self.post_steps + 1, self.state_dim)
    if history.shape != (self.num_envs, STATE_HISTORY, self.state_dim):
      raise ValueError("Evaluation history has the wrong shape")
    if routes.shape != expected or duration.shape != (self.num_envs,):
      raise ValueError("Evaluation routes or durations have the wrong shape")
    self._batch = history, routes, duration

  def reference_at(self, offsets: int | torch.Tensor) -> torch.Tensor:
    if self._batch is None:
      return super().reference_at(offsets)
    if isinstance(offsets, int):
      tick = (self.step + offsets).clamp(max=self.routes.shape[1] - 1)
      return self.routes[torch.arange(self.num_envs, device=self.device), tick]
    tick = (self.step[:, None] + offsets[None]).clamp(max=self.routes.shape[1] - 1)
    batch = torch.arange(self.num_envs, device=self.device)[:, None]
    return self.routes[batch, tick]

  def _post_target(self) -> torch.Tensor:
    if self._batch is None:
      return super()._post_target()
    batch = torch.arange(self.num_envs, device=self.device)
    post_end = (self.window_steps + self.post_steps).clamp(max=self.routes.shape[1] - 1)
    return self.routes[batch, post_end]

  @torch.no_grad()
  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self._batch is None:
      super()._resample_command(env_ids)
      return
    if env_ids.numel() != self.num_envs:
      raise RuntimeError("Evaluation batches must reset every environment together")
    history, routes, duration = self._batch
    start = history[:, -1]
    self.route_start[:] = start[:, :3]
    self.route_origin[:] = start[:, :3]
    self.route_origin[:, :2] = self._env.scene.env_origins[:, :2]
    self.route_rotation[:] = quat_conjugate(yaw_quat(start[:, 3:7]))
    self.routes[:] = self._place_state(routes.flatten(0, 1)).view_as(routes)
    placed_history = self._place_state(history.flatten(0, 1)).view_as(history)
    initial = self._write_initial_state(env_ids, self.routes[:, 0])
    placed_history[:, -1] = initial
    self.batch_history[:] = placed_history
    self.actual_history[:] = placed_history
    self.window_steps[:] = duration
    self._opened[:] = self._env.common_step_counter
    batch = torch.arange(self.num_envs, device=self.device)
    self.target[:] = self.routes[batch, duration]
    self._history_updated_at[:] = self._env.common_step_counter
    self.final_errors.zero_()
    self.final_score.zero_()
    self.arrived.zero_()


@dataclass(kw_only=True)
class EvaluationTrackerCommandCfg(TrackerCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> EvaluationTrackerCommand:
    self.validate()
    return EvaluationTrackerCommand(self, env)


@dataclass(frozen=True)
class PlanGateCfg:
  max_root_speed: float = 4.0
  max_root_angular_speed: float = 8.0
  max_joint_speed: float = 20.0
  max_root_acceleration: float = 30.0
  max_joint_acceleration: float = 120.0
  max_velocity_mismatch: float = 3.0
  max_foot_speed_in_contact: float = 0.4
  max_foot_reach: float = 0.75
  max_foot_separation: float = 1.0
  allowed_penetration: float = 0.01
  max_penetration: float = 0.05
  max_motion_penalty: float = 1.0
  max_joint_limit_penalty: float = 0.01
  max_forbidden_contact_fraction: float = 0.05
  max_foot_placement_penalty: float = 5.0

  def __post_init__(self) -> None:
    if min(vars(self).values()) < 0:
      raise ValueError("Plan gate settings cannot be negative")


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  while mask.ndim < value.ndim:
    mask = mask.unsqueeze(-1)
  mask = mask.expand_as(value)
  return (value * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)


def _excess(value: torch.Tensor, limit: float) -> torch.Tensor:
  return torch.relu(value / max(limit, 1e-6) - 1).square()


def motion_penalties(
  paths: torch.Tensor, duration: torch.Tensor, fps: float, cfg: PlanGateCfg
) -> dict[str, torch.Tensor]:
  joints = (paths.shape[-1] - ROOT_STATE_DIM) // 2
  q = paths[..., ROOT_STATE_DIM : ROOT_STATE_DIM + joints]
  qd = paths[..., ROOT_STATE_DIM + joints :]
  time_index = torch.arange(paths.shape[1], device=paths.device)[None]
  valid = time_index <= duration[:, None]
  edge = time_index[:, 1:] <= duration[:, None]
  velocity = (
    _masked_mean(_excess(paths[..., 7:10].norm(dim=-1), cfg.max_root_speed), valid)
    + _masked_mean(
      _excess(paths[..., 10:13].norm(dim=-1), cfg.max_root_angular_speed), valid
    )
    + _masked_mean(_excess(qd.abs(), cfg.max_joint_speed), valid)
  )
  root_acceleration = (paths[:, 1:, 7:10] - paths[:, :-1, 7:10]) * fps
  joint_acceleration = (qd[:, 1:] - qd[:, :-1]) * fps
  acceleration = _masked_mean(
    _excess(root_acceleration.norm(dim=-1), cfg.max_root_acceleration), edge
  ) + _masked_mean(_excess(joint_acceleration.abs(), cfg.max_joint_acceleration), edge)
  root_fd = (paths[:, 1:, :3] - paths[:, :-1, :3]) * fps
  joint_fd = (q[:, 1:] - q[:, :-1]) * fps
  consistency = _masked_mean(
    _excess((root_fd - paths[:, :-1, 7:10]).norm(dim=-1), cfg.max_velocity_mismatch),
    edge,
  ) + _masked_mean(
    _excess((joint_fd - qd[:, :-1]).abs(), cfg.max_velocity_mismatch), edge
  )
  return {
    "velocity": velocity,
    "acceleration": acceleration,
    "consistency": consistency,
  }


class KinematicPlanGate:
  """Reject obvious invalid plans without ranking the survivors."""

  def __init__(self, command: EvaluationTrackerCommand, cfg: PlanGateCfg) -> None:
    self.cfg = cfg
    self.model = copy.deepcopy(command._env.sim.mj_model)
    self.data = mujoco.MjData(self.model)
    self.free_qpos = command.robot.indexing.free_joint_q_adr.cpu().numpy()
    self.joint_qpos = command.robot.indexing.joint_q_adr.cpu().numpy()
    self.robot_geoms = set(command.robot.indexing.geom_ids.cpu().tolist())
    self.foot_geoms: dict[int, int] = {}
    for geom in self.robot_geoms:
      name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
      if "left_foot" in name:
        self.foot_geoms[geom] = 0
      elif "right_foot" in name:
        self.foot_geoms[geom] = 1
    if not self.foot_geoms:
      raise ValueError("No foot collision geoms found for plan gating")
    body_ids = command.robot.indexing.body_ids.cpu().tolist()
    self.pelvis = body_ids[0]
    self.feet = tuple(
      mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot")
      for side in ("left", "right")
    )
    if any(site < 0 for site in self.feet):
      raise ValueError("Robot must expose left_foot and right_foot sites")
    self.joint_limits = command.robot.data.soft_joint_pos_limits[0].detach().cpu()

  @torch.no_grad()
  def check(
    self, paths: torch.Tensor, duration: torch.Tensor, fps: float
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    components = motion_penalties(paths, duration, fps, self.cfg)
    joints = (paths.shape[-1] - ROOT_STATE_DIM) // 2
    q = paths[..., ROOT_STATE_DIM : ROOT_STATE_DIM + joints]
    low = self.joint_limits[:, 0].to(q.device)
    high = self.joint_limits[:, 1].to(q.device)
    time_index = torch.arange(paths.shape[1], device=paths.device)[None]
    valid = time_index <= duration[:, None]
    components["joint_limit"] = _masked_mean(
      (torch.relu(low - q) + torch.relu(q - high)).square(), valid
    )

    values = paths.detach().cpu().numpy()
    durations = duration.detach().cpu().numpy()
    penetration = np.zeros(len(values), dtype=np.float32)
    forbidden = np.zeros(len(values), dtype=np.float32)
    placement = np.zeros(len(values), dtype=np.float32)
    for batch, plan in enumerate(values):
      length = int(durations[batch]) + 1
      foot_position = np.zeros((length, 2, 3), dtype=np.float64)
      pelvis_position = np.zeros((length, 3), dtype=np.float64)
      contact = np.zeros((length, 2), dtype=bool)
      for frame, state in enumerate(plan[:length]):
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.free_qpos] = state[:7]
        self.data.qpos[self.joint_qpos] = state[
          ROOT_STATE_DIM : ROOT_STATE_DIM + joints
        ]
        mujoco.mj_forward(self.model, self.data)
        foot_position[frame] = self.data.site_xpos[list(self.feet)]
        pelvis_position[frame] = self.data.xpos[self.pelvis]
        for index in range(self.data.ncon):
          hit = self.data.contact[index]
          first, second = int(hit.geom1), int(hit.geom2)
          first_robot = first in self.robot_geoms
          second_robot = second in self.robot_geoms
          penetration[batch] = max(
            penetration[batch],
            -float(hit.dist) - self.cfg.allowed_penetration,
          )
          if first_robot and second_robot:
            forbidden[batch] += 1.0
          elif first_robot != second_robot:
            robot_geom = first if first_robot else second
            if robot_geom in self.foot_geoms:
              contact[frame, self.foot_geoms[robot_geom]] = True
            else:
              forbidden[batch] += 1.0
      reach = np.linalg.norm(
        foot_position[:, :, :2] - pelvis_position[:, None, :2], axis=-1
      )
      separation = np.linalg.norm(
        foot_position[:, 0, :2] - foot_position[:, 1, :2], axis=-1
      )
      placement[batch] += np.square(
        np.maximum(reach / max(self.cfg.max_foot_reach, 1e-6) - 1.0, 0.0)
      ).mean()
      placement[batch] += np.square(
        np.maximum(separation / max(self.cfg.max_foot_separation, 1e-6) - 1.0, 0.0)
      ).mean()
      if length > 1:
        foot_speed = np.linalg.norm(np.diff(foot_position, axis=0), axis=-1) * fps
        planted = contact[1:] | contact[:-1]
        slide = np.square(
          np.maximum(
            foot_speed / max(self.cfg.max_foot_speed_in_contact, 1e-6) - 1.0,
            0.0,
          )
        )
        if planted.any():
          placement[batch] += float(slide[planted].mean())
      forbidden[batch] /= length
    components["penetration"] = torch.from_numpy(penetration).to(paths.device)
    components["forbidden_contact"] = torch.from_numpy(forbidden).to(paths.device)
    components["foot_placement"] = torch.from_numpy(placement).to(paths.device)
    stacked = torch.stack(tuple(components.values()), dim=-1)
    accepted = torch.isfinite(stacked).all(dim=-1)
    accepted &= components["velocity"] <= self.cfg.max_motion_penalty
    accepted &= components["acceleration"] <= self.cfg.max_motion_penalty
    accepted &= components["consistency"] <= self.cfg.max_motion_penalty
    accepted &= components["joint_limit"] <= self.cfg.max_joint_limit_penalty
    accepted &= components["penetration"] <= self.cfg.max_penetration
    accepted &= (
      components["forbidden_contact"] <= self.cfg.max_forbidden_contact_fraction
    )
    accepted &= components["foot_placement"] <= self.cfg.max_foot_placement_penalty
    return accepted, components


@dataclass(frozen=True)
class EvaluationCfg:
  minimum_evaluator_route_score: float = 0.50
  minimum_evaluator_survival: float = 0.90
  minimum_bridge_score: float = 0.25
  minimum_post_score: float = 0.50
  endpoint_tolerance_scale: float = 2.0
  max_action_saturation_fraction: float = 0.20
  fallen_gravity: float = -0.70
  minimum_root_height: float = 0.45

  def __post_init__(self) -> None:
    probabilities = (
      self.minimum_evaluator_route_score,
      self.minimum_evaluator_survival,
      self.minimum_bridge_score,
      self.minimum_post_score,
      self.max_action_saturation_fraction,
    )
    if any(not 0 <= value <= 1 for value in probabilities):
      raise ValueError("Evaluation fractions and scores must be in [0, 1]")
    if self.endpoint_tolerance_scale < 1 or self.minimum_root_height <= 0:
      raise ValueError("Evaluation tolerances are invalid")


@dataclass(frozen=True)
class RolloutResult:
  history: torch.Tensor
  route: torch.Tensor
  actual: torch.Tensor
  errors: torch.Tensor
  fallen: torch.Tensor
  foot_contact: torch.Tensor
  actions: torch.Tensor
  action_saturation: torch.Tensor
  action_rate: torch.Tensor


@dataclass(frozen=True)
class EvaluatedPlans:
  history: torch.Tensor
  target: torch.Tensor
  duration: torch.Tensor
  planned: torch.Tensor
  physical: torch.Tensor
  tracking_errors: torch.Tensor
  fallen: torch.Tensor
  foot_contact: torch.Tensor
  actions: torch.Tensor
  hard_pass: torch.Tensor
  success: torch.Tensor
  hindsight: torch.Tensor
  endpoint_errors: torch.Tensor
  bridge_score: torch.Tensor
  post_score: torch.Tensor
  action_saturation: torch.Tensor
  action_rate: torch.Tensor
  start_source: torch.Tensor
  target_source: torch.Tensor
  candidates: int

  def checkpoint(self) -> dict[str, torch.Tensor | int]:
    return {
      name: value.cpu() if isinstance(value, torch.Tensor) else value
      for name, value in vars(self).items()
    }


class PhysicalReplay:
  """Bounded replay of quality-gated physical paths with achieved endpoints."""

  def __init__(self, capacity: int) -> None:
    if capacity < 1:
      raise ValueError("Replay capacity must be positive")
    self.capacity = capacity
    self.states: torch.Tensor | None = None
    self.duration: torch.Tensor | None = None

  def __len__(self) -> int:
    return 0 if self.duration is None else self.duration.numel()

  def add(self, states: torch.Tensor, duration: torch.Tensor) -> None:
    if states.shape[0] == 0:
      return
    states = states.detach().cpu().to(torch.float16)
    duration = duration.detach().cpu()
    self.states = states if self.states is None else torch.cat((self.states, states))
    self.duration = (
      duration if self.duration is None else torch.cat((self.duration, duration))
    )
    if len(self) > self.capacity:
      assert self.states is not None and self.duration is not None
      self.states = self.states[-self.capacity :]
      self.duration = self.duration[-self.capacity :]

  def sample(
    self, count: int, device: str, history: int
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if self.states is None or self.duration is None:
      raise RuntimeError("Physical replay is empty")
    picked = torch.randint(len(self), (count,))
    states = self.states[picked].to(device=device, dtype=torch.float32)
    duration = self.duration[picked].to(device)
    return encode(states, states[:, history - 1]), duration

  def checkpoint(self) -> dict[str, torch.Tensor | int | None]:
    return {
      "capacity": self.capacity,
      "states": self.states,
      "duration": self.duration,
    }

  @classmethod
  def from_checkpoint(cls, saved: dict) -> PhysicalReplay:
    replay = cls(int(saved["capacity"]))
    replay.states = saved.get("states")
    replay.duration = saved.get("duration")
    return replay


@dataclass(frozen=True)
class PlannerCfg:
  history: int = 4
  future: int = 1
  min_steps: int = 15
  max_steps: int = 60
  time_scale_range: tuple[float, float] = (0.95, 1.05)
  start_xy_range: float = 0.02
  start_joint_range: float = 0.03
  start_perturb_probability: float = 0.5
  mirror_probability: float = 0.5
  model: ModelCfg = field(default_factory=ModelCfg)
  process: ProcessCfg = field(default_factory=ProcessCfg)
  batch: int = 256
  bootstrap_updates: int = 30_000
  updates_per_cycle: int = 1_000
  learning_rate: float = 2e-4
  fit_batches: int = 32
  max_grad_norm: float = 1.0
  physical_fraction: float = 0.10

  def __post_init__(self) -> None:
    if not 0 <= self.physical_fraction < 1:
      raise ValueError("physical_fraction must be in [0, 1)")


class PlannerTrainer:
  """Train the existing planner on demonstrations and gated physical paths."""

  def __init__(
    self, cfg: PlannerCfg, checkpoint: Path | None, device: str, robot: str = "g1"
  ) -> None:
    self.cfg = cfg
    self.device = device
    columns = cfg.history + cfg.max_steps + cfg.future - 1
    corpus = load_motions(
      motion_patterns(robot, "train"), columns, device, "all", robot=robot
    )
    self.windows = Windows(
      corpus,
      cfg.history,
      cfg.future,
      cfg.min_steps,
      cfg.max_steps,
      cfg.time_scale_range,
      cfg.start_xy_range,
      cfg.start_joint_range,
      cfg.start_perturb_probability,
      cfg.mirror_probability,
    )
    self.warm_started = checkpoint is not None
    if checkpoint is not None:
      self.bridge = DiffusionBridge.load(checkpoint, device)
      if self.bridge.robot != robot:
        raise ValueError(f"Planner checkpoint is for {self.bridge.robot}, not {robot}")
      raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
      self.metadata = dict(raw.get("planner", raw))
    else:
      with torch.no_grad():
        samples = torch.cat(
          [self.windows.sample(cfg.batch)[0] for _ in range(cfg.fit_batches)]
        )
        normalizer = Normalizer.fit(samples)
      model = Denoiser(self.windows.layout.width, self.windows.columns, cfg.model).to(
        device
      )
      process = Diffusion(model, cfg.process).to(device)
      self.bridge = DiffusionBridge(
        process,
        normalizer,
        self.windows.layout,
        cfg.history,
        cfg.future,
        corpus.fps,
        cfg.min_steps,
        robot,
      )
      self.metadata = checkpoint_metadata(
        cfg.model,
        cfg.process,
        self.windows.layout,
        normalizer,
        cfg.history,
        cfg.future,
        cfg.min_steps,
        cfg.max_steps,
        corpus.fps,
        dict(model.state_dict()),
        0,
        robot,
      )
    if self.bridge.layout != self.windows.layout or not math.isclose(
      self.bridge.fps, corpus.fps
    ):
      raise ValueError(
        "Planner checkpoint and motion data use different layouts or rates"
      )
    self.optimizer = torch.optim.AdamW(
      self.bridge.process.denoiser.parameters(), lr=cfg.learning_rate
    )
    if checkpoint is not None:
      raw = torch.load(checkpoint, map_location=device, weights_only=False)
      if raw.get("planner_optimizer_state_dict") is not None:
        self.optimizer.load_state_dict(raw["planner_optimizer_state_dict"])
    self.updates = int(self.metadata.get("iteration", 0))
    self.bridge.process.eval().requires_grad_(False)

  def _loss(self, features: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
    clean = self.bridge.normalizer.normalize(features)
    known = bridge_mask(
      clean.shape[0],
      clean.shape[1],
      self.bridge.layout,
      self.bridge.history,
      self.bridge.future,
      duration,
    )
    rows = self.bridge.history - 1 + duration
    time_index = torch.arange(clean.shape[1], device=clean.device)[None]
    valid = time_index <= rows[:, None] + self.bridge.future - 1
    return self.bridge.process.loss(clean, known, valid)

  def train(self, updates: int, replay: PhysicalReplay) -> dict[str, float]:
    if updates < 1:
      return {"planner/loss": 0.0, "planner/updates": float(self.updates)}
    self.bridge.process.train().requires_grad_(True)
    total = real_total = physical_total = 0.0
    physical_weight = self.cfg.physical_fraction if len(replay) else 0.0
    for _ in range(updates):
      real_features, real_duration = self.windows.sample(self.cfg.batch)
      real_loss = self._loss(real_features, real_duration)
      loss = real_loss
      physical_loss = None
      if physical_weight:
        physical = replay.sample(self.cfg.batch, self.device, self.bridge.history)
        physical_loss = self._loss(*physical)
        loss = (1 - physical_weight) * real_loss + physical_weight * physical_loss
      self.optimizer.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(
        self.bridge.process.denoiser.parameters(), self.cfg.max_grad_norm
      )
      self.optimizer.step()
      total += loss.item()
      real_total += real_loss.item()
      if physical_loss is not None:
        physical_total += physical_loss.item()
    self.updates += updates
    self.bridge.process.eval().requires_grad_(False)
    return {
      "planner/loss": total / updates,
      "planner/real_loss": real_total / updates,
      "planner/physical_loss": physical_total / updates,
      "planner/physical_fraction": physical_weight,
      "planner/updates": float(self.updates),
    }

  def saved(self) -> dict:
    saved = dict(self.metadata)
    saved["ema"] = {
      name: value.detach().cpu()
      for name, value in self.bridge.process.denoiser.state_dict().items()
    }
    saved["iteration"] = self.updates
    return saved

  def load(self, checkpoint: Path) -> None:
    loaded = DiffusionBridge.load(checkpoint, self.device)
    if (
      loaded.layout != self.bridge.layout
      or loaded.history != self.bridge.history
      or loaded.future != self.bridge.future
      or loaded.max_steps != self.bridge.max_steps
    ):
      raise ValueError("Resumed planner shape differs from this task")
    self.bridge.process.denoiser.load_state_dict(loaded.process.denoiser.state_dict())
    self.bridge.normalizer = loaded.normalizer
    raw = torch.load(checkpoint, map_location=self.device, weights_only=False)
    self.metadata = dict(raw.get("planner", raw))
    if raw.get("planner_optimizer_state_dict") is not None:
      self.optimizer.load_state_dict(raw["planner_optimizer_state_dict"])
    self.updates = int(self.metadata.get("iteration", 0))


@dataclass
class PlannerImprovementRunnerCfg(RslRlOnPolicyRunnerCfg):
  robot: str = "g1"
  tracker_checkpoint: str = ""
  planner_checkpoint: str = ""
  planner: PlannerCfg = field(default_factory=PlannerCfg)
  pairs: PairCfg = field(default_factory=PairCfg)
  gate: PlanGateCfg = field(default_factory=PlanGateCfg)
  evaluation: EvaluationCfg = field(default_factory=EvaluationCfg)
  candidates_per_condition: int = 4
  replay_capacity: int = 8192


class PlannerImprovementRunner(MjlabOnPolicyRunner):
  """Keep the tracker fixed and improve only the diffusion planner."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    tracker_checkpoint = str(train_cfg.pop("tracker_checkpoint", ""))
    planner_checkpoint = str(train_cfg.pop("planner_checkpoint", ""))
    robot = str(train_cfg.pop("robot", "g1"))
    planner_values = train_cfg.pop("planner")
    planner_values["model"] = ModelCfg(**planner_values["model"])
    planner_values["process"] = ProcessCfg(**planner_values["process"])
    self.planner_cfg = PlannerCfg(**planner_values)
    self.pair_cfg = PairCfg(**train_cfg.pop("pairs"))
    self.gate_cfg = PlanGateCfg(**train_cfg.pop("gate"))
    self.evaluation_cfg = EvaluationCfg(**train_cfg.pop("evaluation"))
    self.candidates = int(train_cfg.pop("candidates_per_condition"))
    replay_capacity = int(train_cfg.pop("replay_capacity"))
    if self.candidates < 1:
      raise ValueError("candidates_per_condition must be positive")
    super().__init__(env, train_cfg, log_dir, device)
    if self.is_distributed:
      raise ValueError("Diffusion planner improvement supports one GPU")
    command = self.env.unwrapped.command_manager.get_term(COMMAND)
    if not isinstance(command, EvaluationTrackerCommand):
      raise TypeError("PlannerImprovementRunner needs EvaluationTrackerCommand")
    self.command = command
    planner_path = Path(planner_checkpoint) if planner_checkpoint else None
    if planner_path is not None and not planner_path.is_file():
      raise FileNotFoundError(f"Planner checkpoint not found: {planner_path}")
    self.planner = PlannerTrainer(self.planner_cfg, planner_path, device, robot)
    if self.planner.bridge.history < STATE_HISTORY:
      raise ValueError("Planner history is shorter than tracker state history")
    if self.planner.bridge.max_steps != command.max_steps:
      raise ValueError("Planner and evaluator maximum durations differ")
    transitions = self.planner.windows.data.dataset()
    if transitions.num_joints != self.planner.bridge.layout.joints:
      raise ValueError("Motion data and planner use different robots")
    if not math.isclose(transitions.fps, self.planner.bridge.fps):
      raise ValueError("Motion data and planner use different rates")
    self.pairs = CrossTrajectoryPairs(
      transitions,
      self.planner.bridge.history,
      command.post_steps,
      self.pair_cfg,
      self.planner.windows.data.starts + self.planner.bridge.history - 1,
    )
    self.gate = KinematicPlanGate(command, self.gate_cfg)
    self.replay = PhysicalReplay(replay_capacity)
    self.evaluated: EvaluatedPlans | None = None
    self.tracker_loaded = False
    if tracker_checkpoint:
      path = Path(tracker_checkpoint)
      if not path.is_file():
        raise FileNotFoundError(f"Tracker checkpoint not found: {path}")
      super().load(
        str(path), load_cfg={"actor": True}, strict=True, map_location=device
      )
      self.tracker_loaded = True
      self.current_learning_iteration = 0
      self.env.unwrapped.common_step_counter = 0
      print(f"[diffusion] frozen evaluator {path}")
    if planner_path is not None:
      print(f"[diffusion] warm planner {planner_path}")
    self.alg.eval_mode()
    for parameter in self.alg.get_policy().parameters():
      parameter.requires_grad_(False)

  def _routes(self, plans: torch.Tensor, pair: PairBatch) -> torch.Tensor:
    route_length = self.command.max_steps + self.command.post_steps + 1
    routes = plans.new_empty(plans.shape[0], route_length, plans.shape[-1])
    for row, steps in enumerate(pair.duration.tolist()):
      routes[row, : steps + 1] = plans[row, : steps + 1]
      routes[row, steps : steps + self.command.post_steps + 1] = pair.target[
        row, : self.command.post_steps + 1
      ]
      routes[row, steps + self.command.post_steps + 1 :] = routes[
        row, steps + self.command.post_steps
      ]
    return routes

  @torch.no_grad()
  def _execute(
    self, history: torch.Tensor, routes: torch.Tensor, duration: torch.Tensor
  ) -> RolloutResult:
    count = history.shape[0]
    total = self.env.num_envs
    padded_history = _repeat_to_batch(history[:, -STATE_HISTORY:], total)
    padded_routes = _repeat_to_batch(routes, total)
    padded_duration = _repeat_to_batch(duration, total)
    self.command.set_batch(padded_history, padded_routes, padded_duration)
    obs, _ = self.env.reset()
    policy = self.get_inference_policy(device=self.device)
    actual = [self.command.state_now().clone()]
    actions: list[torch.Tensor] = []
    feet = self.env.unwrapped.scene.sensors.get("feet_ground_contact")

    def contact_now() -> torch.Tensor:
      if isinstance(feet, ContactSensor) and feet.data.found is not None:
        return (feet.data.found > 0).clone()
      return torch.zeros(total, 2, dtype=torch.bool, device=self.device)

    contacts = [contact_now()]
    horizon = self.command.max_steps + self.command.post_steps
    gravity = torch.zeros(total, 3, device=self.device)
    gravity[:, 2] = -1.0
    fallen = [
      (
        quat_apply_inverse(actual[0][:, 3:7], gravity)[:, 2]
        > self.evaluation_cfg.fallen_gravity
      )
      | (actual[0][:, 2] < self.evaluation_cfg.minimum_root_height)
    ]
    for _ in range(horizon):
      action = policy(obs)
      obs, _, _, _ = self.env.step(action)
      here = self.command.state_now().clone()
      actual.append(here)
      actions.append(action.clone())
      contacts.append(contact_now())
      fallen.append(
        (
          quat_apply_inverse(here[:, 3:7], gravity)[:, 2]
          > self.evaluation_cfg.fallen_gravity
        )
        | (here[:, 2] < self.evaluation_cfg.minimum_root_height)
      )
    actual_tensor = torch.stack(actual, dim=1)[:count]
    route = self.command.routes[:count].clone()
    history = self.command.batch_history[:count].clone()
    errors = channel_errors(
      actual_tensor.flatten(0, 1), route.flatten(0, 1), self.command.upper_body
    ).view(count, horizon + 1, -1)
    action_tensor = torch.stack(actions, dim=1)[:count]
    clip = float(self.cfg.get("clip_actions") or 100.0)
    saturation = (action_tensor.abs() >= 0.99 * clip).float().mean(dim=(1, 2))
    action_rate = (
      (action_tensor[:, 1:] - action_tensor[:, :-1]).square().mean(dim=(1, 2))
      if horizon > 1
      else torch.zeros(count, device=self.device)
    )
    return RolloutResult(
      history,
      route,
      actual_tensor,
      errors,
      torch.stack(fallen, dim=1)[:count],
      torch.stack(contacts, dim=1)[:count],
      action_tensor,
      saturation,
      action_rate,
    )

  def _rollout_metrics(
    self, rollout: RolloutResult, duration: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    steps = torch.arange(rollout.actual.shape[1], device=self.device)[None]
    bridge_mask = (steps > 0) & (steps <= duration[:, None])
    post_mask = (steps > duration[:, None]) & (
      steps <= duration[:, None] + self.command.post_steps
    )
    tracking = score(
      rollout.errors.flatten(0, 1),
      self.command.tolerances * self.command.cfg.tracking_tolerance_scale,
    ).view_as(bridge_mask)
    bridge_score = _masked_mean(tracking, bridge_mask)
    post_score = _masked_mean(tracking, post_mask)
    batch = torch.arange(duration.numel(), device=self.device)
    endpoint_errors = rollout.errors[batch, duration]
    endpoint = (
      endpoint_errors
      <= self.command.tolerances * self.evaluation_cfg.endpoint_tolerance_scale
    ).all(dim=-1)
    bridge_fall = (rollout.fallen & (steps <= duration[:, None])).any(dim=-1)
    post_fall = (
      rollout.fallen & (steps <= duration[:, None] + self.command.post_steps)
    ).any(dim=-1)
    quality = (
      ~bridge_fall
      & (bridge_score >= self.evaluation_cfg.minimum_bridge_score)
      & (
        rollout.action_saturation <= self.evaluation_cfg.max_action_saturation_fraction
      )
    )
    post_ok = ~post_fall & (post_score >= self.evaluation_cfg.minimum_post_score)
    success = quality & endpoint & post_ok
    hindsight = quality & ~endpoint
    return {
      "endpoint_errors": endpoint_errors,
      "bridge_score": bridge_score,
      "post_score": post_score,
      "success": success,
      "hindsight": hindsight,
      "survived": ~post_fall,
    }

  def _real_routes(self, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if self.command.dataset is None or self.command.windows is None:
      raise RuntimeError("Evaluator has no held-out motion data")
    _, _, duration, position = self.command.windows.draw(count)
    history_rows = self.command.windows.history(position, STATE_HISTORY - 1)
    route_rows = self.command.windows.path(
      position, duration, self.command.max_steps, self.command.post_steps
    )
    return (
      self.command.dataset.states[history_rows],
      self.command.dataset.states[route_rows],
      duration,
    )

  def _validate_evaluator(self) -> dict[str, float]:
    history, route, duration = self._real_routes(self.env.num_envs)
    rollout = self._execute(history, route, duration)
    metrics = self._rollout_metrics(rollout, duration)
    route_score = float(metrics["bridge_score"].mean())
    survival = float(metrics["survived"].float().mean())
    if (
      route_score < self.evaluation_cfg.minimum_evaluator_route_score
      or survival < self.evaluation_cfg.minimum_evaluator_survival
    ):
      raise RuntimeError(
        "Frozen tracker is not a fair evaluator on held-out motion data: "
        f"route score {route_score:.3f}, survival {survival:.3f}"
      )
    return {
      "evaluator/route_score": route_score,
      "evaluator/survival": survival,
    }

  @torch.no_grad()
  def _evaluate_planner(self) -> tuple[EvaluatedPlans, dict[str, float]]:
    candidates = min(self.candidates, self.env.num_envs)
    conditions = max(self.env.num_envs // candidates, 1)
    base = self.pairs.draw(conditions)
    pair = PairBatch(
      base.history.repeat_interleave(candidates, dim=0),
      base.target.repeat_interleave(candidates, dim=0),
      base.duration.repeat_interleave(candidates),
      base.start_source.repeat_interleave(candidates),
      base.target_source.repeat_interleave(candidates),
    )
    planned = self.planner.bridge.generate(
      pair.history, pair.target[:, : self.planner.bridge.future], pair.duration
    ).states
    routes = self._routes(planned, pair)
    hard_pass, components = self.gate.check(
      planned, pair.duration, self.planner.bridge.fps
    )
    survivors = hard_pass.nonzero().flatten()
    horizon = self.command.max_steps + self.command.post_steps + 1
    physical = torch.full(
      (planned.shape[0], horizon, planned.shape[-1]),
      torch.nan,
      dtype=planned.dtype,
      device=planned.device,
    )
    tracking_errors = torch.full(
      (planned.shape[0], horizon, self.command.tolerances.numel()),
      torch.nan,
      device=planned.device,
    )
    fallen = torch.zeros(
      planned.shape[0], horizon, dtype=torch.bool, device=planned.device
    )
    foot_contact = torch.zeros(
      planned.shape[0], horizon, 2, dtype=torch.bool, device=planned.device
    )
    actions = torch.full(
      (planned.shape[0], horizon - 1, self.env.num_actions),
      torch.nan,
      device=planned.device,
    )
    endpoint_errors = torch.full(
      (planned.shape[0], self.command.tolerances.numel()),
      torch.nan,
      device=planned.device,
    )
    bridge_score = torch.full((planned.shape[0],), torch.nan, device=planned.device)
    post_score = bridge_score.clone()
    action_saturation = bridge_score.clone()
    action_rate = bridge_score.clone()
    success = torch.zeros_like(hard_pass)
    hindsight = torch.zeros_like(hard_pass)
    accepted_states = planned.new_empty(
      0,
      self.planner.bridge.history + self.planner.bridge.max_steps,
      planned.shape[-1],
    )
    accepted_duration = pair.duration[:0]
    if survivors.numel():
      rollout = self._execute(
        pair.history[survivors], routes[survivors], pair.duration[survivors]
      )
      measured = self._rollout_metrics(rollout, pair.duration[survivors])
      physical[survivors] = rollout.actual
      tracking_errors[survivors] = rollout.errors
      fallen[survivors] = rollout.fallen
      foot_contact[survivors] = rollout.foot_contact
      actions[survivors] = rollout.actions
      endpoint_errors[survivors] = measured["endpoint_errors"]
      bridge_score[survivors] = measured["bridge_score"]
      post_score[survivors] = measured["post_score"]
      action_saturation[survivors] = rollout.action_saturation
      action_rate[survivors] = rollout.action_rate
      success[survivors] = measured["success"]
      hindsight[survivors] = measured["hindsight"]
      accepted_local = (measured["success"] | measured["hindsight"]).nonzero().flatten()
      if accepted_local.numel():
        accepted = survivors[accepted_local]
        actual = rollout.actual[
          accepted_local, : self.planner.bridge.max_steps + 1
        ].clone()
        accepted_duration = pair.duration[accepted]
        for row, steps in enumerate(accepted_duration.tolist()):
          actual[row, steps:] = actual[row, steps]
        accepted_states = torch.cat(
          (rollout.history[accepted_local, :-1], actual), dim=1
        )
    self.replay.add(accepted_states, accepted_duration)
    evaluated = EvaluatedPlans(
      pair.history.detach().cpu().to(torch.float16),
      pair.target.detach().cpu().to(torch.float16),
      pair.duration.detach().cpu(),
      planned.detach().cpu().to(torch.float16),
      physical.detach().cpu().to(torch.float16),
      tracking_errors.detach().cpu().to(torch.float16),
      fallen.detach().cpu(),
      foot_contact.detach().cpu(),
      actions.detach().cpu().to(torch.float16),
      hard_pass.detach().cpu(),
      success.detach().cpu(),
      hindsight.detach().cpu(),
      endpoint_errors.detach().cpu(),
      bridge_score.detach().cpu(),
      post_score.detach().cpu(),
      action_saturation.detach().cpu(),
      action_rate.detach().cpu(),
      pair.start_source.detach().cpu(),
      pair.target_source.detach().cpu(),
      candidates,
    )
    metrics = {
      "plans/generated": float(planned.shape[0]),
      "plans/hard_pass": float(hard_pass.sum()),
      "plans/hard_pass_fraction": float(hard_pass.float().mean()),
      "plans/success": float(success.sum()),
      "plans/success_fraction": float(success.float().mean()),
      "plans/hindsight": float(hindsight.sum()),
      "plans/hindsight_fraction": float(hindsight.float().mean()),
      "replay/size": float(len(self.replay)),
    }
    metrics.update(
      {f"gate/{name}": float(value.mean()) for name, value in components.items()}
    )
    if survivors.numel():
      metrics.update(
        {
          "tracker/bridge_score": float(bridge_score[survivors].mean()),
          "tracker/post_score": float(post_score[survivors].mean()),
          "tracker/action_saturation": float(action_saturation[survivors].mean()),
          "tracker/action_rate": float(action_rate[survivors].mean()),
        }
      )
    return evaluated, metrics

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    del init_at_random_ep_len
    if not self.tracker_loaded:
      raise ValueError(
        "Planner improvement requires --agent.tracker-checkpoint with a pretrained tracker"
      )
    self.logger.init_logging_writer()
    evaluator_metrics = self._validate_evaluator()
    start = self.current_learning_iteration
    if start == 0 and not self.planner.warm_started:
      print(
        f"[diffusion] planner pretrain: {self.planner_cfg.bootstrap_updates} updates"
      )
      self.planner.train(self.planner_cfg.bootstrap_updates, self.replay)
    for cycle in range(start, num_learning_iterations):
      collect_started = time.time()
      self.evaluated, metrics = self._evaluate_planner()
      collect_time = time.time() - collect_started
      learn_started = time.time()
      metrics.update(
        self.planner.train(self.planner_cfg.updates_per_cycle, self.replay)
      )
      metrics.update(evaluator_metrics)
      learn_time = time.time() - learn_started
      self.current_learning_iteration = cycle + 1
      self.logger.log(
        it=cycle,
        start_it=start,
        total_it=num_learning_iterations,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=metrics,
        learning_rate=self.planner_cfg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=None,
      )
      if (
        self.logger.writer is not None and (cycle + 1) % self.cfg["save_interval"] == 0
      ):
        assert self.logger.log_dir is not None
        self.save(os.path.join(self.logger.log_dir, f"model_{cycle + 1}.pt"))
    if self.logger.writer is not None:
      assert self.logger.log_dir is not None
      self.save(
        os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
      )
      self.logger.stop_logging_writer()

  def save(self, path: str, infos=None) -> None:
    saved = self.alg.save()
    saved["iter"] = self.current_learning_iteration
    saved["infos"] = {
      **(infos or {}),
      "env_state": {"common_step_counter": self.env.unwrapped.common_step_counter},
    }
    saved["planner"] = self.planner.saved()
    saved["planner_optimizer_state_dict"] = self.planner.optimizer.state_dict()
    saved["physical_replay"] = self.replay.checkpoint()
    saved["planner_evaluation"] = (
      self.evaluated.checkpoint() if self.evaluated is not None else None
    )
    torch.save(saved, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)
    if self._eval_video_on_save:
      self._record_eval_video(self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    if load_cfg is None:
      saved = torch.load(path, map_location=map_location, weights_only=False)
      if "planner" not in saved:
        raise ValueError("A planner-improvement checkpoint must contain the planner")
      self.planner.load(Path(path))
      self.planner.warm_started = True
      if saved.get("physical_replay") is not None:
        self.replay = PhysicalReplay.from_checkpoint(saved["physical_replay"])
      self.tracker_loaded = True
    return infos


def training_env_cfg(play: bool = False, robot: str = "g1") -> ManagerBasedRlEnvCfg:
  cfg = tracker_env_cfg(
    play=True,
    split="eval",
    motion_patterns=motion_patterns(robot, "val"),
    robot=robot,
  )
  cfg.scene.num_envs = 1 if play else 512
  original = cfg.commands[COMMAND]
  if not isinstance(original, TrackerCommandCfg):
    raise TypeError("Tracker environment did not create a TrackerCommandCfg")
  values = vars(original).copy()
  values["duration_s_range"] = (0.3, 1.2)
  values["debug_vis"] = play
  cfg.commands[COMMAND] = EvaluationTrackerCommandCfg(**values)
  cfg.rewards = {}
  cfg.terminations = {}
  cfg.metrics = {}
  cfg.episode_length_s = 1.0e9
  return cfg


def training_runner_cfg(robot: str = "g1") -> PlannerImprovementRunnerCfg:
  base = tracker_ppo_runner_cfg(robot)
  return PlannerImprovementRunnerCfg(
    robot=robot,
    seed=base.seed,
    num_steps_per_env=base.num_steps_per_env,
    max_iterations=30,
    obs_groups=base.obs_groups,
    save_interval=1,
    experiment_name=improvement_experiment(robot),
    actor=base.actor,
    critic=base.critic,
    algorithm=base.algorithm,
  )


for _robot in ROBOTS:
  register_mjlab_task(
    task_id=improvement_task_id(_robot),
    env_cfg=training_env_cfg(robot=_robot),
    play_env_cfg=training_env_cfg(play=True, robot=_robot),
    rl_cfg=training_runner_cfg(_robot),
    runner_cls=PlannerImprovementRunner,
  )

__all__ = ["TRAIN_EXPERIMENT", "TRAIN_TASK_ID"]
