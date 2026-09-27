"""Online mixture of recorded references and cross-trajectory diffusion plans."""

# pyright: reportPrivateImportUsage=false, reportIncompatibleVariableOverride=false

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  TRACKER_DATASET,
  Dataset,
  load_dataset,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.planner.bridge import (
  DiffusionBridge,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker.command import (
  STATE_HISTORY,
  TrackerCommand,
  TrackerCommandCfg,
)
from mjlab.utils.lab_api.math import (
  quat_conjugate,
  quat_from_angle_axis,
  quat_mul,
  yaw_quat,
)


@dataclass(frozen=True)
class PlanEpisodes:
  history: torch.Tensor
  target: torch.Tensor
  duration: torch.Tensor
  path: torch.Tensor
  outcome: torch.Tensor


class PlanReplay:
  """Small CPU replay of plans labelled by their physical tracking outcome."""

  def __init__(self, max_batches: int = 8) -> None:
    if max_batches < 1:
      raise ValueError("max_batches must be positive")
    self._batches: deque[PlanEpisodes] = deque(maxlen=max_batches)

  def __len__(self) -> int:
    return sum(batch.duration.numel() for batch in self._batches)

  def add(self, batch: PlanEpisodes) -> None:
    if batch.duration.numel() == 0:
      return
    self._batches.append(
      PlanEpisodes(
        batch.history.detach().to("cpu", torch.float16),
        batch.target.detach().to("cpu", torch.float16),
        batch.duration.detach().to("cpu"),
        batch.path.detach().to("cpu", torch.float16),
        batch.outcome.detach().to("cpu", torch.float32),
      )
    )

  def sample(self, count: int, device: str) -> PlanEpisodes:
    if count < 1 or not self._batches:
      raise ValueError("cannot sample an empty replay")
    joined = PlanEpisodes(
      *(
        torch.cat([getattr(batch, name) for batch in self._batches])
        for name in PlanEpisodes.__dataclass_fields__
      )
    )
    indexes = torch.randint(0, joined.duration.numel(), (count,))
    return PlanEpisodes(
      joined.history[indexes].to(device=device, dtype=torch.float32),
      joined.target[indexes].to(device=device, dtype=torch.float32),
      joined.duration[indexes].to(device),
      joined.path[indexes].to(device=device, dtype=torch.float32),
      joined.outcome[indexes].to(device),
    )


@dataclass(frozen=True)
class LinearRamp:
  start: float = 0.0
  end: float = 0.5
  steps: int = 5_000_000

  def __post_init__(self) -> None:
    if not 0 <= self.start <= self.end <= 1:
      raise ValueError("cross-trajectory fractions must satisfy 0 <= start <= end <= 1")
    if self.steps < 1:
      raise ValueError("ramp steps must be positive")

  def __call__(self, step: int) -> float:
    phase = min(max(step, 0) / self.steps, 1.0)
    return self.start + phase * (self.end - self.start)


class CrossTrajectoryPairs:
  """Draw history and target contexts from different physical trajectories."""

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
      raise ValueError(
        "cross-trajectory dataset needs at least two usable trajectories"
      )

  def _target_counts(self, trajectory: torch.Tensor) -> torch.Tensor:
    """Number of target contexts belonging to each requested trajectory."""
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


class CoTrainingCommand(TrackerCommand):
  """Generate cross-trajectory references online and retain their outcomes."""

  cfg: CoTrainingCommandCfg

  def __init__(self, cfg: CoTrainingCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    if not cfg.planner_checkpoint.is_file():
      raise FileNotFoundError(
        f"Planner checkpoint not found: {cfg.planner_checkpoint}. "
        "Set --env.commands.path.planner-checkpoint."
      )
    self.planner = DiffusionBridge.load(
      cfg.planner_checkpoint, str(self.device), cfg.planner_sample_steps
    )
    if self.planner.history < STATE_HISTORY:
      raise ValueError("planner history is shorter than tracker state history")
    if self.planner.future != self.post_steps + 1:
      raise ValueError("planner future must exactly cover tracker lookahead")
    if self.max_steps != self.planner.max_steps:
      raise ValueError("tracker and planner maximum durations must match")
    cross_data = load_dataset(cfg.cross_dataset_path, str(self.device), cfg.split)
    if cross_data.num_joints != self.num_joints:
      raise ValueError("cross-trajectory dataset and robot joint counts differ")
    if not math.isclose(cross_data.fps, self.fps):
      raise ValueError("cross-trajectory dataset and environment rates differ")
    self.pairs = CrossTrajectoryPairs(
      cross_data, self.planner.history, self.planner.future
    )
    self.ramp = LinearRamp(
      cfg.cross_trajectory_start,
      cfg.cross_trajectory_end,
      cfg.cross_trajectory_ramp_steps,
    )
    self.replay = PlanReplay(cfg.replay_batches)
    route_length = self.max_steps + self.post_steps + 1
    self.online = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.online_routes = torch.zeros(
      self.num_envs, route_length, self.state_dim, device=self.device
    )
    self.plan_history = torch.zeros(
      self.num_envs,
      self.planner.history,
      self.state_dim,
      device=self.device,
    )
    self.plan_target = torch.zeros(
      self.num_envs,
      self.planner.future,
      self.state_dim,
      device=self.device,
    )
    self.plan_path = torch.zeros_like(self.online_routes)
    self.route_sum = torch.zeros(self.num_envs, device=self.device)
    self.route_count = torch.zeros(self.num_envs, device=self.device)

  @property
  def cross_trajectory_fraction(self) -> float:
    return self.ramp(int(self._env.common_step_counter))

  def trajectory_score(self) -> torch.Tensor:
    value = super().trajectory_score()
    self.route_sum += value.detach()
    self.route_count += 1
    return value

  def reference_at(self, offsets: int | torch.Tensor) -> torch.Tensor:
    reference = super().reference_at(offsets)
    env_ids = self.online.nonzero().flatten()
    if env_ids.numel() == 0:
      return reference
    if isinstance(offsets, int):
      ticks = (self.step[env_ids] + offsets).clamp(max=self.online_routes.shape[1] - 1)
      reference[env_ids] = self.online_routes[env_ids, ticks]
      return reference
    ticks = (self.step[env_ids, None] + offsets[None]).clamp(
      max=self.online_routes.shape[1] - 1
    )
    batch = env_ids[:, None].expand_as(ticks)
    reference[env_ids] = self.online_routes[batch, ticks]
    return reference

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._record_outcomes(env_ids)
    choose_online = (
      torch.rand(env_ids.numel(), device=self.device) < self.cross_trajectory_fraction
    )
    recorded_ids = env_ids[~choose_online]
    online_ids = env_ids[choose_online]
    if recorded_ids.numel():
      super()._resample_command(recorded_ids)
      self.online[recorded_ids] = False
      self.route_sum[recorded_ids] = 0
      self.route_count[recorded_ids] = 0
    if online_ids.numel():
      self._resample_online(online_ids)

  def _record_outcomes(self, env_ids: torch.Tensor) -> None:
    finished = env_ids[self.online[env_ids]]
    if finished.numel() == 0:
      return
    route = self.route_sum[finished] / self.route_count[finished].clamp_min(1)
    reached_deadline = self.step[finished] >= self.window_steps[finished]
    outcome = torch.where(
      reached_deadline,
      0.8 * self.final_score[finished] + 0.2 * route,
      0.1 * route,
    ).clamp(0.0, 1.0)
    self.replay.add(
      PlanEpisodes(
        self.plan_history[finished],
        self.plan_target[finished],
        self.window_steps[finished],
        self.plan_path[finished],
        outcome,
      )
    )

  @torch.no_grad()
  def _resample_online(self, env_ids: torch.Tensor) -> None:
    count = env_ids.numel()
    history, target, duration = self.pairs.draw(
      count, self.planner.min_steps, self.max_steps
    )
    path = self.planner.generate(history, target, duration).states
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
    placed_history = self._place_state(
      history[:, -STATE_HISTORY:].flatten(0, 1), env_ids
    ).view(count, STATE_HISTORY, self.state_dim)
    initial = self._write_initial_state(env_ids, placed_path[:, 0])
    placed_history[:, -1] = initial

    self.online[env_ids] = True
    self.online_routes[env_ids] = placed_path
    self.window_steps[env_ids] = duration
    self._opened[env_ids] = self._env.common_step_counter
    batch = torch.arange(count, device=self.device)
    self.target[env_ids] = placed_path[batch, duration]
    self.actual_history[env_ids] = placed_history
    self._history_updated_at[env_ids] = self._env.common_step_counter
    self.final_errors[env_ids] = 0
    self.final_score[env_ids] = 0
    self.arrived[env_ids] = False
    self.route_sum[env_ids] = 0
    self.route_count[env_ids] = 0
    self.plan_history[env_ids] = history
    self.plan_target[env_ids] = target
    self.plan_path[env_ids] = path


@dataclass(kw_only=True)
class CoTrainingCommandCfg(TrackerCommandCfg):
  planner_checkpoint: Path = Path("planner.pt")
  cross_dataset_path: Path = TRACKER_DATASET
  planner_sample_steps: int | None = 10
  cross_trajectory_start: float = 0.0
  cross_trajectory_end: float = 0.5
  cross_trajectory_ramp_steps: int = 360_000
  replay_batches: int = 8

  def build(self, env: ManagerBasedRlEnv) -> CoTrainingCommand:
    LinearRamp(
      self.cross_trajectory_start,
      self.cross_trajectory_end,
      self.cross_trajectory_ramp_steps,
    )
    if self.replay_batches < 1:
      raise ValueError("replay_batches must be positive")
    return CoTrainingCommand(self, env)
