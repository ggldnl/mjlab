"""Small-perturbation command for terminal residual training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DEFAULT_DATASET,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
  ImitationCommand,
  ImitationCommandCfg,
)
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)


def _rotation_6d(quat: torch.Tensor) -> torch.Tensor:
  matrix = matrix_from_quat(quat)
  return matrix[..., :, :2].transpose(-1, -2).flatten(-2)


def imitation_command(command: ImitationCommand) -> torch.Tensor:
  """Encode the target exactly like the frozen imitation bridge expects."""
  current = command.state_now()
  joints = command.num_joints
  q = slice(command.state_dim - 2 * joints, command.state_dim - joints)
  qd = slice(q.stop, q.stop + joints)
  heading = yaw_quat(current[:, 3:7])
  phase = command.step.float() / command.window_steps.float()
  return torch.cat(
    (
      quat_apply_inverse(heading, command.target[:, :3] - current[:, :3]),
      _rotation_6d(quat_mul(quat_conjugate(current[:, 3:7]), command.target[:, 3:7])),
      quat_apply_inverse(current[:, 3:7], command.target[:, 7:10] - current[:, 7:10]),
      quat_apply_inverse(current[:, 3:7], command.target[:, 10:13] - current[:, 10:13]),
      command.target[:, q] - current[:, q],
      command.target[:, qd] - current[:, qd],
      (1.0 - phase).clamp(min=0.0).unsqueeze(-1),
      phase.clamp(max=1.0).unsqueeze(-1),
    ),
    dim=-1,
  )


def correction_weight(
  step: torch.Tensor,
  window_steps: torch.Tensor,
  docking: torch.Tensor,
  enabled: torch.Tensor,
  correction_steps: int,
) -> torch.Tensor:
  """Gate corrections to the capture region and final control steps."""
  remaining = window_steps - step
  active = docking & enabled & (remaining <= correction_steps) & (remaining > 0)
  return active.to(dtype=torch.float32)


class ResidualCommand(ImitationCommand):
  """Sample only recoverable perturbations around demonstrated target states."""

  cfg: ResidualCommandCfg

  def __init__(self, cfg: ResidualCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.start_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.entering_action = torch.zeros(
      self.num_envs, self.num_joints, device=self.device
    )
    self._best_score = torch.zeros(self.num_envs, device=self.device)
    self._captured = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._progress = torch.zeros(self.num_envs, device=self.device)
    self._new_capture = torch.zeros(self.num_envs, device=self.device)
    self._advanced_at = -1

  @property
  def docking(self) -> torch.Tensor:
    return (self.target_errors() <= self.tolerances * 3.0).all(dim=-1)

  @property
  def captured(self) -> torch.Tensor:
    within = self.active & (self.target_errors() <= self.tolerances).all(dim=-1)
    return self._captured | within

  @property
  def new_capture(self) -> torch.Tensor:
    return self._new_capture

  @property
  def handoff(self) -> torch.Tensor:
    return self.captured | self.deadline

  @property
  def step(self) -> torch.Tensor:
    return super().step + self.start_step

  @property
  def command(self) -> torch.Tensor:
    return imitation_command(self)

  @property
  def residual_weight(self) -> torch.Tensor:
    return correction_weight(
      self.step,
      self.window_steps,
      self.docking,
      torch.ones_like(self.docking),
      self.cfg.correction_steps(self.fps),
    )

  def reference_now(self) -> torch.Tensor:
    return self.target

  def advance(self) -> torch.Tensor:
    """Update residual progress and capture once per simulation step."""
    if self._advanced_at == self._env.common_step_counter:
      return self._progress
    self._advanced_at = self._env.common_step_counter
    errors = self.target_errors()
    current = 1.0 / (1.0 + (errors / self.tolerances).amax(dim=-1))
    self._progress = (current - self._best_score).clamp(min=0.0)
    self._best_score = torch.maximum(self._best_score, current)
    captured = self.active & (errors <= self.tolerances).all(dim=-1)
    self._new_capture = (captured & ~self._captured).float()
    self._captured |= captured
    return self._progress

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self._best_score[env_ids] = 0.0
    self._captured[env_ids] = False
    self._progress[env_ids] = 0.0
    self._new_capture[env_ids] = 0.0
    self.entering_action[env_ids] = 0.0
    if self.dataset is None:
      self.start_step[env_ids] = 0
      return
    local_steps = self.cfg.correction_steps(self.fps)
    self.start_step[env_ids] = (self.window_steps[env_ids] - local_steps).clamp(min=0)
    rows = self.route_rows[env_ids].gather(1, self.start_step[env_ids, None]).squeeze(1)
    if self.dataset.previous_action is not None:
      self.entering_action[env_ids] = self.dataset.previous_action[rows]
    self._write_initial_state(
      env_ids, self._place_state(self.dataset.states[rows], env_ids)
    )

  def open_window(
    self,
    env_ids: torch.Tensor,
    target: torch.Tensor,
    duration_s: torch.Tensor,
  ) -> None:
    self.start_step[env_ids] = 0
    self._best_score[env_ids] = 0.0
    self._captured[env_ids] = False
    self._progress[env_ids] = 0.0
    self._new_capture[env_ids] = 0.0
    super().open_window(env_ids, target, duration_s)


@dataclass(kw_only=True)
class ResidualCommandCfg(ImitationCommandCfg):
  """Residual correction horizon and capture-region sampler."""

  dataset_path: Path | None = DEFAULT_DATASET
  duration_s_range: tuple[float, float] = (0.3, 1.2)
  docking_probability: float = 1.0
  docking_duration_s: float = 0.20
  residual_scale: float = 0.25

  def correction_steps(self, fps: float) -> int:
    return max(1, round(self.docking_duration_s * fps))

  def build(self, env: ManagerBasedRlEnv) -> ResidualCommand:
    if self.docking_probability != 1.0:
      raise ValueError("Residual training samples only local target perturbations")
    return ResidualCommand(self, env)
