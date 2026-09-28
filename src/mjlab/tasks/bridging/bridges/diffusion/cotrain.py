"""Alternate diffusion planning and universal tracking in one training run."""

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
from rsl_rl.utils import check_nan
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg
from mjlab.tasks.bridging.bridges.dataset.dataset import ROOT_STATE_DIM, Dataset
from mjlab.tasks.bridging.bridges.diffusion.dataset.motions import (
  BABEL_EVAL_MOTIONS,
  BABEL_TRAIN_MOTIONS,
  MotionCorpus,
  Normalizer,
  Windows,
  bridge_mask,
  encode,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.bridge import (
  DiffusionBridge,
  checkpoint_metadata,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.model import Denoiser, ModelCfg
from mjlab.tasks.bridging.bridges.diffusion.planner.process import (
  Diffusion,
  ProcessCfg,
)
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
from mjlab.tasks.registry import register_mjlab_task
from mjlab.utils.lab_api.math import (
  quat_conjugate,
  quat_from_angle_axis,
  quat_mul,
  yaw_quat,
)

COTRAIN_TASK_ID = "Mjlab-G1-Diffusion-CoTrain"
COTRAIN_EXPERIMENT = "g1_diffusion_cotrain"


@dataclass(frozen=True)
class LinearRamp:
  start: float = 0.0
  end: float = 0.5
  cycles: int = 10

  def __post_init__(self) -> None:
    if not 0 <= self.start <= self.end <= 1:
      raise ValueError("fractions must satisfy 0 <= start <= end <= 1")
    if self.cycles < 1:
      raise ValueError("cycles must be positive")

  def __call__(self, cycle: int) -> float:
    phase = min(max(cycle, 0) / self.cycles, 1.0)
    return self.start + phase * (self.end - self.start)


class CrossTrajectoryPairs:
  """Draw boundary contexts from different BABEL clips."""

  def __init__(self, data: Dataset, history: int, future: int) -> None:
    self.data = data
    self.history = history
    self.future = future
    self.order = data.segments(1, 1).order
    self.starts = data.segments(1, 1, before=history - 1).starts
    targets = data.segments(1, 1, after=future - 1).starts

    target_trajectory = data.trajectory[self.order[targets]]
    order = torch.argsort(target_trajectory)
    self.targets = targets[order]
    target_trajectory = target_trajectory[order]
    self.target_trajectories, self.target_counts = torch.unique_consecutive(
      target_trajectory, return_counts=True
    )
    self.target_first = self.target_counts.cumsum(0) - self.target_counts

    start_trajectory = data.trajectory[self.order[self.starts]]
    excluded = self._target_counts(start_trajectory)
    self.starts = self.starts[excluded < self.targets.numel()]
    if self.starts.numel() == 0:
      raise ValueError("Cross trajectory training needs at least two usable clips")

  def _target_counts(self, trajectory: torch.Tensor) -> torch.Tensor:
    slot = torch.searchsorted(self.target_trajectories, trajectory)
    bounded = slot.clamp(max=self.target_trajectories.numel() - 1)
    present = (slot < self.target_trajectories.numel()) & (
      self.target_trajectories[bounded] == trajectory
    )
    return torch.where(present, self.target_counts[bounded], 0)

  def draw(
    self, count: int, min_steps: int, max_steps: int
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = self.data.states.device
    a = self.starts[torch.randint(self.starts.numel(), (count,), device=device)]
    a_trajectory = self.data.trajectory[self.order[a]]
    slot = torch.searchsorted(self.target_trajectories, a_trajectory)
    bounded = slot.clamp(max=self.target_trajectories.numel() - 1)
    present = (slot < self.target_trajectories.numel()) & (
      self.target_trajectories[bounded] == a_trajectory
    )
    excluded = torch.where(present, self.target_counts[bounded], 0)
    first = torch.where(present, self.target_first[bounded], self.targets.numel())
    available = self.targets.numel() - excluded
    picked = (torch.rand(count, device=device) * available).long()
    picked += ((picked >= first) & present).long() * excluded
    b = self.targets[picked]

    history_offsets = torch.arange(1 - self.history, 1, device=device)
    future_offsets = torch.arange(self.future, device=device)
    history = self.data.states[self.order[a[:, None] + history_offsets]]
    target = self.data.states[self.order[b[:, None] + future_offsets]]
    duration = torch.randint(min_steps, max_steps + 1, (count,), device=device)
    return history, target, duration


class MixedTrackerCommand(TrackerCommand):
  """Serve recorded BABEL windows and a coordinator supplied generated pool."""

  def __init__(self, cfg: MixedTrackerCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    route_length = self.max_steps + self.post_steps + 1
    self.generated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.generated_routes = torch.zeros(
      self.num_envs, route_length, self.state_dim, device=self.device
    )
    self._pool_history: torch.Tensor | None = None
    self._pool_routes: torch.Tensor | None = None
    self._pool_duration: torch.Tensor | None = None
    self.generated_fraction = 0.0

  def set_generated_pool(
    self,
    history: torch.Tensor,
    routes: torch.Tensor,
    duration: torch.Tensor,
    fraction: float,
  ) -> None:
    expected_route = self.max_steps + self.post_steps + 1
    if history.ndim != 3 or history.shape[1:] != (STATE_HISTORY, self.state_dim):
      raise ValueError("Generated history has the wrong shape")
    if routes.shape != (history.shape[0], expected_route, self.state_dim):
      raise ValueError("Generated routes have the wrong shape")
    if duration.shape != (history.shape[0],):
      raise ValueError("Generated durations have the wrong shape")
    if not 0 <= fraction <= 1:
      raise ValueError("Generated fraction must be in [0, 1]")
    self._pool_history = history
    self._pool_routes = routes
    self._pool_duration = duration
    self.generated_fraction = fraction

  def clear_generated_pool(self) -> None:
    self._pool_history = None
    self._pool_routes = None
    self._pool_duration = None
    self.generated_fraction = 0.0

  def reference_at(self, offsets: int | torch.Tensor) -> torch.Tensor:
    reference = super().reference_at(offsets)
    env_ids = self.generated.nonzero().flatten()
    if env_ids.numel() == 0:
      return reference
    if isinstance(offsets, int):
      ticks = (self.step[env_ids] + offsets).clamp(
        max=self.generated_routes.shape[1] - 1
      )
      reference[env_ids] = self.generated_routes[env_ids, ticks]
      return reference
    ticks = (self.step[env_ids, None] + offsets[None]).clamp(
      max=self.generated_routes.shape[1] - 1
    )
    reference[env_ids] = self.generated_routes[env_ids[:, None], ticks]
    return reference

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    ready = self._pool_routes is not None and self.generated_fraction > 0
    choose_generated = torch.zeros(
      env_ids.numel(), dtype=torch.bool, device=self.device
    )
    if ready:
      choose_generated = (
        torch.rand(env_ids.numel(), device=self.device) < self.generated_fraction
      )
    recorded_ids = env_ids[~choose_generated]
    generated_ids = env_ids[choose_generated]
    if recorded_ids.numel():
      super()._resample_command(recorded_ids)
      self.generated[recorded_ids] = False
    if generated_ids.numel():
      self._resample_generated(generated_ids)

  @torch.no_grad()
  def _resample_generated(self, env_ids: torch.Tensor) -> None:
    assert self._pool_history is not None
    assert self._pool_routes is not None
    assert self._pool_duration is not None
    count = env_ids.numel()
    picked = torch.randint(self._pool_duration.numel(), (count,), device=self.device)
    history = self._pool_history[picked]
    path = self._pool_routes[picked]
    duration = self._pool_duration[picked]

    axis = torch.zeros(count, 3, device=self.device)
    axis[:, 2] = 1.0
    rotation = quat_mul(
      quat_from_angle_axis(torch.rand(count, device=self.device) * 2 * math.pi, axis),
      quat_conjugate(yaw_quat(history[:, -1, 3:7])),
    )
    origin = history[:, -1, :3]
    landing = origin.clone()
    landing[:, :2] = self._env.scene.env_origins[env_ids, :2]
    self.route_start[env_ids] = origin
    self.route_origin[env_ids] = landing
    self.route_rotation[env_ids] = rotation

    placed_path = self._place_state(path.flatten(0, 1), env_ids).view_as(path)
    placed_history = self._place_state(history.flatten(0, 1), env_ids).view_as(history)
    initial = self._write_initial_state(env_ids, placed_path[:, 0])
    placed_history[:, -1] = initial

    self.generated[env_ids] = True
    self.generated_routes[env_ids] = placed_path
    self.window_steps[env_ids] = duration
    self._opened[env_ids] = self._env.common_step_counter
    batch = torch.arange(count, device=self.device)
    self.target[env_ids] = placed_path[batch, duration]
    self.actual_history[env_ids] = placed_history
    self._history_updated_at[env_ids] = self._env.common_step_counter
    self.final_errors[env_ids] = 0
    self.final_score[env_ids] = 0
    self.arrived[env_ids] = False


@dataclass(kw_only=True)
class MixedTrackerCommandCfg(TrackerCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> MixedTrackerCommand:
    self.validate()
    return MixedTrackerCommand(self, env)


@dataclass(frozen=True)
class PlanScoreCfg:
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
  velocity_weight: float = 0.5
  acceleration_weight: float = 0.25
  consistency_weight: float = 0.5
  joint_limit_weight: float = 2.0
  contact_weight: float = 2.0
  foot_placement_weight: float = 1.0

  def __post_init__(self) -> None:
    if min(vars(self).values()) < 0:
      raise ValueError("Plan score settings cannot be negative")


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  while mask.ndim < value.ndim:
    mask = mask.unsqueeze(-1)
  mask = mask.expand_as(value)
  return (value * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)


def _excess(value: torch.Tensor, limit: float) -> torch.Tensor:
  return torch.relu(value / max(limit, 1e-6) - 1).square()


def motion_penalties(
  paths: torch.Tensor, duration: torch.Tensor, fps: float, cfg: PlanScoreCfg
) -> dict[str, torch.Tensor]:
  """Velocity, acceleration, and stored velocity consistency penalties."""
  joints = (paths.shape[-1] - ROOT_STATE_DIM) // 2
  q = paths[..., ROOT_STATE_DIM : ROOT_STATE_DIM + joints]
  qd = paths[..., ROOT_STATE_DIM + joints :]
  time = torch.arange(paths.shape[1], device=paths.device)[None]
  valid = time <= duration[:, None]
  edge = time[:, 1:] <= duration[:, None]

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
  mismatch = (root_fd - paths[:, :-1, 7:10]).norm(dim=-1)
  joint_mismatch = (joint_fd - qd[:, :-1]).abs()
  consistency = _masked_mean(
    _excess(mismatch, cfg.max_velocity_mismatch), edge
  ) + _masked_mean(_excess(joint_mismatch, cfg.max_velocity_mismatch), edge)
  return {
    "velocity": velocity,
    "acceleration": acceleration,
    "consistency": consistency,
  }


class KinematicPlanScorer:
  """Score plan plausibility from its states and MuJoCo forward kinematics."""

  def __init__(self, command: MixedTrackerCommand, cfg: PlanScoreCfg) -> None:
    self.cfg = cfg
    self.model = copy.deepcopy(command._env.sim.mj_model)
    self.data = mujoco.MjData(self.model)
    self.free_qpos = command.robot.indexing.free_joint_q_adr.cpu().numpy()
    self.joint_qpos = command.robot.indexing.joint_q_adr.cpu().numpy()
    self.robot_geoms = set(command.robot.indexing.geom_ids.cpu().tolist())
    self.foot_geoms: dict[int, int] = {}
    for geom in self.robot_geoms:
      name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
      if "left_foot" in name and name.endswith("_collision"):
        self.foot_geoms[geom] = 0
      elif "right_foot" in name and name.endswith("_collision"):
        self.foot_geoms[geom] = 1
    if not self.foot_geoms:
      raise ValueError("No G1 foot collision geoms found for plan scoring")
    body_ids = command.robot.indexing.body_ids.cpu().tolist()
    self.pelvis = body_ids[command.robot.body_names.index("pelvis")]
    self.feet = (
      body_ids[command.robot.body_names.index("left_ankle_roll_link")],
      body_ids[command.robot.body_names.index("right_ankle_roll_link")],
    )
    self.joint_limits = command.robot.data.soft_joint_pos_limits[0].detach().cpu()

  @torch.no_grad()
  def score(
    self, paths: torch.Tensor, duration: torch.Tensor, fps: float
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    components = motion_penalties(paths, duration, fps, self.cfg)
    joints = (paths.shape[-1] - ROOT_STATE_DIM) // 2
    q = paths[..., ROOT_STATE_DIM : ROOT_STATE_DIM + joints]
    low = self.joint_limits[:, 0].to(q.device)
    high = self.joint_limits[:, 1].to(q.device)
    time_index = torch.arange(paths.shape[1], device=paths.device)[None]
    valid = time_index <= duration[:, None]
    joint_limit = (torch.relu(low - q) + torch.relu(q - high)).square()
    components["joint_limit"] = _masked_mean(joint_limit, valid)

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
        foot_position[frame] = self.data.xpos[list(self.feet)]
        pelvis_position[frame] = self.data.xpos[self.pelvis]
        for index in range(self.data.ncon):
          hit = self.data.contact[index]
          first, second = int(hit.geom1), int(hit.geom2)
          first_robot = first in self.robot_geoms
          second_robot = second in self.robot_geoms
          depth = max(-float(hit.dist) - self.cfg.allowed_penetration, 0.0)
          penetration[batch] += depth / max(self.cfg.allowed_penetration, 1e-3)
          if first_robot and second_robot:
            forbidden[batch] += 1.0
            continue
          if first_robot == second_robot:
            continue
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
        placement[batch] += float(slide[planted].mean()) if planted.any() else 0.0
      penetration[batch] /= length
      forbidden[batch] /= length

    components["penetration"] = torch.from_numpy(penetration).to(paths.device)
    components["forbidden_contact"] = torch.from_numpy(forbidden).to(paths.device)
    components["foot_placement"] = torch.from_numpy(placement).to(paths.device)
    total = (
      self.cfg.velocity_weight * components["velocity"]
      + self.cfg.acceleration_weight * components["acceleration"]
      + self.cfg.consistency_weight * components["consistency"]
      + self.cfg.joint_limit_weight * components["joint_limit"]
      + self.cfg.contact_weight
      * (components["penetration"] + components["forbidden_contact"])
      + self.cfg.foot_placement_weight * components["foot_placement"]
    )
    return torch.exp(-total), components


@dataclass(frozen=True)
class ScoredPlans:
  history: torch.Tensor
  target: torch.Tensor
  duration: torch.Tensor
  path: torch.Tensor
  score: torch.Tensor
  candidates: int

  def best_indices(self, minimum: float) -> torch.Tensor:
    grouped = self.score.view(-1, self.candidates)
    best = grouped.argmax(1)
    rows = torch.arange(grouped.shape[0]) * self.candidates + best
    return rows[self.score[rows] >= minimum]

  def accepted_fraction(self, minimum: float) -> float:
    groups = self.score.numel() // self.candidates
    return self.best_indices(minimum).numel() / groups

  def checkpoint(self) -> dict[str, torch.Tensor | int]:
    return {
      "history": self.history.cpu(),
      "target": self.target.cpu(),
      "duration": self.duration.cpu(),
      "path": self.path.cpu(),
      "score": self.score.cpu(),
      "candidates": self.candidates,
    }

  @classmethod
  def from_checkpoint(cls, saved: dict) -> ScoredPlans:
    return cls(
      saved["history"],
      saved["target"],
      saved["duration"],
      saved["path"],
      saved["score"],
      int(saved["candidates"]),
    )


@dataclass(frozen=True)
class CoTrainPlannerCfg:
  history: int = 4
  future: int = 1
  min_steps: int = 15
  max_steps: int = 60
  model: ModelCfg = field(default_factory=ModelCfg)
  process: ProcessCfg = field(default_factory=ProcessCfg)
  batch: int = 256
  bootstrap_updates: int = 30_000
  updates_per_cycle: int = 1_000
  learning_rate: float = 2e-4
  fit_batches: int = 32
  max_grad_norm: float = 1.0


def _motion_windows(
  data: Dataset, history: int, future: int, min_steps: int, max_steps: int
) -> Windows:
  columns = history + max_steps + future - 1
  segments = data.segments(columns - 1, columns - 1)
  corpus = MotionCorpus(
    states=data.states,
    starts=segments.order[segments.starts],
    names=data.names,
    fps=data.fps,
    num_joints=data.num_joints,
  )
  return Windows(corpus, history, future, min_steps, max_steps)


class PlannerTrainer:
  """Train the existing planner on BABEL and selected plausible samples."""

  def __init__(
    self,
    data: Dataset,
    cfg: CoTrainPlannerCfg,
    checkpoint: Path | None,
    device: str,
  ) -> None:
    self.cfg = cfg
    self.device = device
    self.warm_started = checkpoint is not None
    if checkpoint is not None:
      self.bridge = DiffusionBridge.load(checkpoint, device)
      raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
      self.metadata = dict(raw.get("planner", raw))
    else:
      windows = _motion_windows(
        data, cfg.history, cfg.future, cfg.min_steps, cfg.max_steps
      )
      with torch.no_grad():
        samples = torch.cat(
          [windows.sample(cfg.batch)[0] for _ in range(cfg.fit_batches)]
        )
        normalizer = Normalizer.fit(samples)
      model = Denoiser(windows.layout.width, windows.columns, cfg.model).to(device)
      process = Diffusion(model, cfg.process).to(device)
      self.bridge = DiffusionBridge(
        process,
        normalizer,
        windows.layout,
        cfg.history,
        cfg.future,
        data.fps,
        cfg.min_steps,
      )
      self.metadata = checkpoint_metadata(
        cfg.model,
        cfg.process,
        windows.layout,
        normalizer,
        cfg.history,
        cfg.future,
        cfg.min_steps,
        cfg.max_steps,
        data.fps,
        dict(model.state_dict()),
        0,
      )
    if self.bridge.layout.joints != data.num_joints or not math.isclose(
      self.bridge.fps, data.fps
    ):
      raise ValueError(
        "Planner checkpoint and BABEL data use different layouts or rates"
      )
    self.windows = _motion_windows(
      data,
      self.bridge.history,
      self.bridge.future,
      self.bridge.min_steps,
      self.bridge.max_steps,
    )
    self.optimizer = torch.optim.AdamW(
      self.bridge.process.denoiser.parameters(), lr=cfg.learning_rate
    )
    if checkpoint is not None:
      raw = torch.load(checkpoint, map_location=device, weights_only=False)
      state = raw.get("planner_optimizer_state_dict")
      if state is not None:
        self.optimizer.load_state_dict(state)
    self.updates = int(self.metadata.get("iteration", 0))
    self.bridge.process.requires_grad_(False)

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

  def _online_batch(
    self, plans: ScoredPlans, count: int, minimum: float
  ) -> tuple[torch.Tensor, torch.Tensor] | None:
    best = plans.best_indices(minimum)
    if best.numel() == 0:
      return None
    picked = best[
      torch.multinomial(plans.score[best].float().clamp_min(1e-6), count, True)
    ]
    history = plans.history[picked].to(self.device, dtype=torch.float32)
    path = plans.path[picked].to(self.device, dtype=torch.float32)
    duration = plans.duration[picked].to(self.device)
    states = torch.cat((history[:, :-1], path), dim=1)
    return encode(states, history[:, -1]), duration

  def train(
    self,
    updates: int,
    online: ScoredPlans | None,
    online_weight: float,
    minimum_score: float,
  ) -> dict[str, float]:
    if updates < 1:
      return {"planner/loss": 0.0}
    self.bridge.process.train().requires_grad_(True)
    total = real_total = online_total = 0.0
    used_online = 0
    for _ in range(updates):
      features, duration = self.windows.sample(self.cfg.batch)
      real_loss = self._loss(features, duration)
      loss = real_loss
      online_loss = None
      if online is not None and online_weight > 0:
        batch = self._online_batch(online, self.cfg.batch, minimum_score)
        if batch is not None:
          online_loss = self._loss(*batch)
          loss = loss + online_weight * online_loss
      self.optimizer.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(
        self.bridge.process.denoiser.parameters(), self.cfg.max_grad_norm
      )
      self.optimizer.step()
      total += loss.item()
      real_total += real_loss.item()
      if online_loss is not None:
        online_total += online_loss.item()
        used_online += 1
    self.updates += updates
    self.bridge.process.eval().requires_grad_(False)
    return {
      "planner/loss": total / updates,
      "planner/real_loss": real_total / updates,
      "planner/online_loss": online_total / max(used_online, 1),
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
      raise ValueError("Resumed planner shape differs from the task configuration")
    self.bridge.process.denoiser.load_state_dict(loaded.process.denoiser.state_dict())
    self.bridge.normalizer = loaded.normalizer
    raw = torch.load(checkpoint, map_location=self.device, weights_only=False)
    self.metadata = dict(raw.get("planner", raw))
    state = raw.get("planner_optimizer_state_dict")
    if state is not None:
      self.optimizer.load_state_dict(state)
    self.updates = int(self.metadata.get("iteration", 0))


@dataclass
class CoTrainRunnerCfg(RslRlOnPolicyRunnerCfg):
  tracker_checkpoint: str = ""
  planner_checkpoint: str = ""
  tracker_iterations_per_cycle: int = 1_000
  planner: CoTrainPlannerCfg = field(default_factory=CoTrainPlannerCfg)
  score: PlanScoreCfg = field(default_factory=PlanScoreCfg)
  generated_conditions: int = 128
  candidates_per_condition: int = 4
  minimum_plan_score: float = 0.05
  generated_fraction_start: float = 0.0
  generated_fraction_end: float = 0.5
  generated_fraction_ramp_cycles: int = 10


class CoTrainRunner(MjlabOnPolicyRunner):
  """Coordinate planner training, plan scoring, and PPO tracker training."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    tracker_checkpoint = str(train_cfg.pop("tracker_checkpoint", ""))
    planner_checkpoint = str(train_cfg.pop("planner_checkpoint", ""))
    planner_values = train_cfg.pop("planner")
    planner_values["model"] = ModelCfg(**planner_values["model"])
    planner_values["process"] = ProcessCfg(**planner_values["process"])
    self.planner_cfg = CoTrainPlannerCfg(**planner_values)
    self.score_cfg = PlanScoreCfg(**train_cfg.pop("score"))
    self.tracker_iterations = int(train_cfg.pop("tracker_iterations_per_cycle"))
    self.generated_conditions = int(train_cfg.pop("generated_conditions"))
    self.candidates = int(train_cfg.pop("candidates_per_condition"))
    self.minimum_plan_score = float(train_cfg.pop("minimum_plan_score"))
    self.ramp = LinearRamp(
      float(train_cfg.pop("generated_fraction_start")),
      float(train_cfg.pop("generated_fraction_end")),
      int(train_cfg.pop("generated_fraction_ramp_cycles")),
    )
    if min(self.tracker_iterations, self.generated_conditions, self.candidates) < 1:
      raise ValueError("Cycle iteration and candidate counts must be positive")
    if not 0 <= self.minimum_plan_score <= 1:
      raise ValueError("minimum_plan_score must be in [0, 1]")
    super().__init__(env, train_cfg, log_dir, device)
    if self.is_distributed:
      raise ValueError("Diffusion co-training currently supports one GPU")
    command = self.env.unwrapped.command_manager.get_term(COMMAND)
    if not isinstance(command, MixedTrackerCommand) or command.dataset is None:
      raise TypeError("CoTrainRunner requires a kinematic MixedTrackerCommand")
    self.command = command
    planner_path = Path(planner_checkpoint) if planner_checkpoint else None
    if planner_path is not None and not planner_path.is_file():
      raise FileNotFoundError(f"Planner checkpoint not found: {planner_path}")
    self.planner = PlannerTrainer(
      command.dataset, self.planner_cfg, planner_path, device
    )
    if self.planner.bridge.history < STATE_HISTORY:
      raise ValueError("Planner history is shorter than tracker state history")
    if self.planner.bridge.max_steps != command.max_steps:
      raise ValueError("Planner and tracker maximum durations differ")
    self.pairs = CrossTrajectoryPairs(
      command.dataset, self.planner.bridge.history, self.planner.bridge.future
    )
    self.scorer = KinematicPlanScorer(command, self.score_cfg)
    self.scored: ScoredPlans | None = None
    if tracker_checkpoint:
      path = Path(tracker_checkpoint)
      if not path.is_file():
        raise FileNotFoundError(f"Tracker checkpoint not found: {path}")
      super().load(
        str(path),
        load_cfg={"actor": True, "critic": True, "optimizer": True},
        map_location=device,
      )
      self.current_learning_iteration = 0
      self.env.unwrapped.common_step_counter = 0
      print(f"[cotrain] warm tracker {path}")
    if planner_path is not None:
      print(f"[cotrain] warm planner {planner_path}")

  def _score_candidates(self) -> tuple[ScoredPlans, dict[str, float]]:
    conditions = self.generated_conditions
    history, target, duration = self.pairs.draw(
      conditions, self.planner.bridge.min_steps, self.planner.bridge.max_steps
    )
    history = history.repeat_interleave(self.candidates, dim=0)
    target = target.repeat_interleave(self.candidates, dim=0)
    duration = duration.repeat_interleave(self.candidates)
    generated = self.planner.bridge.generate(history, target, duration).states
    score, components = self.scorer.score(generated, duration, self.planner.bridge.fps)
    plans = ScoredPlans(
      history.detach().cpu().to(torch.float16),
      target.detach().cpu().to(torch.float16),
      duration.detach().cpu(),
      generated.detach().cpu().to(torch.float16),
      score.detach().cpu(),
      self.candidates,
    )
    metrics = {
      "plans/score_mean": score.mean().item(),
      "plans/score_max": score.view(-1, self.candidates).max(1).values.mean().item(),
      "plans/score_std": score.view(-1, self.candidates)
      .std(1, unbiased=False)
      .mean()
      .item(),
      "plans/accepted": float(plans.best_indices(self.minimum_plan_score).numel()),
      "plans/accepted_fraction": plans.accepted_fraction(self.minimum_plan_score),
    }
    metrics.update(
      {f"plans/{name}": value.mean().item() for name, value in components.items()}
    )
    return plans, metrics

  def _set_tracker_pool(self, plans: ScoredPlans, fraction: float) -> float:
    best = plans.best_indices(self.minimum_plan_score)
    effective_fraction = fraction * plans.accepted_fraction(self.minimum_plan_score)
    if best.numel() == 0 or effective_fraction == 0:
      self.command.clear_generated_pool()
      return 0.0
    history = plans.history[best, -STATE_HISTORY:].to(self.device, dtype=torch.float32)
    path = plans.path[best].to(self.device, dtype=torch.float32)
    duration = plans.duration[best].to(self.device)
    route_length = self.command.max_steps + self.command.post_steps + 1
    if path.shape[1] < route_length:
      path = torch.cat(
        (path, path[:, -1:].expand(-1, route_length - path.shape[1], -1)), dim=1
      )
    else:
      path = path[:, :route_length]
    self.command.set_generated_pool(history, path, duration, effective_fraction)
    return effective_fraction

  def _tracker_phase(
    self, obs: TensorDict
  ) -> tuple[TensorDict, dict[str, float], float, float]:
    losses: dict[str, float] = {}
    collect_time = learn_time = 0.0
    self.alg.train_mode()
    for _ in range(self.tracker_iterations):
      started = time.time()
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)
          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )
          self.alg.process_env_step(obs, rewards, dones, extras)
          intrinsic = (
            self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
          )
          self.logger.process_env_step(rewards, dones, extras, intrinsic)
        collect_time += time.time() - started
        self.alg.compute_returns(obs)
      started = time.time()
      update = self.alg.update()
      learn_time += time.time() - started
      for name, value in update.items():
        losses[name] = losses.get(name, 0.0) + float(value)
    return (
      obs,
      {name: value / self.tracker_iterations for name, value in losses.items()},
      collect_time,
      learn_time,
    )

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )
    obs = self.env.get_observations().to(self.device)
    self.logger.init_logging_writer()
    start = self.current_learning_iteration
    for cycle in range(start, num_learning_iterations):
      fraction = self.ramp(cycle)
      online_weight = (
        fraction * self.scored.accepted_fraction(self.minimum_plan_score)
        if self.scored is not None
        else 0.0
      )
      planner_updates = (
        self.planner_cfg.bootstrap_updates
        if cycle == 0 and not self.planner.warm_started
        else self.planner_cfg.updates_per_cycle
      )
      planner_started = time.time()
      metrics = self.planner.train(
        planner_updates,
        self.scored,
        online_weight,
        self.minimum_plan_score,
      )
      self.scored, score_metrics = self._score_candidates()
      metrics.update(score_metrics)
      metrics["curriculum/requested_generated_fraction"] = fraction
      metrics["curriculum/planner_online_weight"] = online_weight
      metrics["timing/planner"] = time.time() - planner_started
      metrics["curriculum/generated_fraction"] = self._set_tracker_pool(
        self.scored, fraction
      )

      obs, tracker_losses, collect_time, learn_time = self._tracker_phase(obs)
      metrics.update(tracker_losses)
      self.current_learning_iteration = cycle + 1
      rnd_weight = None
      if self.cfg["algorithm"]["rnd_cfg"]:
        assert self.alg.rnd is not None
        rnd_weight = self.alg.rnd.weight
      self.logger.log(
        it=cycle,
        start_it=start,
        total_it=num_learning_iterations,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=metrics,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=rnd_weight,
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
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    saved = self.alg.save()
    saved["iter"] = self.current_learning_iteration
    saved["infos"] = {**(infos or {}), "env_state": env_state}
    saved["planner"] = self.planner.saved()
    saved["planner_optimizer_state_dict"] = self.planner.optimizer.state_dict()
    saved["cotrain_scored_plans"] = (
      self.scored.checkpoint() if self.scored is not None else None
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
        raise ValueError("A co-training resume checkpoint must contain the planner")
      self.planner.load(Path(path))
      raw_plans = saved.get("cotrain_scored_plans")
      self.scored = (
        ScoredPlans.from_checkpoint(raw_plans) if raw_plans is not None else None
      )
    return infos


def cotrain_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = tracker_env_cfg(
    play=play,
    split="eval" if play else "train",
    motion_patterns=BABEL_EVAL_MOTIONS if play else BABEL_TRAIN_MOTIONS,
  )
  original = cfg.commands[COMMAND]
  if not isinstance(original, TrackerCommandCfg):
    raise TypeError("Tracker environment did not create a TrackerCommandCfg")
  values = vars(original).copy()
  values["duration_s_range"] = (0.3, 1.2)
  cfg.commands[COMMAND] = MixedTrackerCommandCfg(**values)
  return cfg


def cotrain_runner_cfg() -> CoTrainRunnerCfg:
  base = tracker_ppo_runner_cfg()
  return CoTrainRunnerCfg(
    seed=base.seed,
    num_steps_per_env=base.num_steps_per_env,
    max_iterations=30,
    obs_groups=base.obs_groups,
    save_interval=1,
    experiment_name=COTRAIN_EXPERIMENT,
    actor=base.actor,
    critic=base.critic,
    algorithm=base.algorithm,
  )


register_mjlab_task(
  task_id=COTRAIN_TASK_ID,
  env_cfg=cotrain_env_cfg(),
  play_env_cfg=cotrain_env_cfg(play=True),
  rl_cfg=cotrain_runner_cfg(),
  runner_cls=CoTrainRunner,
)

__all__ = ["COTRAIN_EXPERIMENT", "COTRAIN_TASK_ID"]
