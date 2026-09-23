from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import torch

from mjlab.tasks.tracking.mdp.commands import (
  MotionCommand,
  MotionCommandCfg,
  MotionLoader,
)
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_error_magnitude,
  quat_inv,
  quat_mul,
  yaw_quat,
)


def resolve_motion_files(specs: tuple[str, ...]) -> tuple[str, ...]:
  """Expand motion paths and globs in stable order."""
  files: dict[str, None] = {}
  for spec in specs:
    matches = sorted(glob.glob(spec))
    if not matches and Path(spec).is_file():
      matches = [spec]
    for match in matches:
      files[str(Path(match))] = None
  if not files:
    raise FileNotFoundError(f"No motion files match {specs}")
  return tuple(files)


class MotionSetLoader(MotionLoader):
  """Concatenate compatible motion files while retaining clip boundaries."""

  def __init__(
    self, motion_files: tuple[str, ...], body_indexes: torch.Tensor, device: str
  ) -> None:
    motions = [MotionLoader(path, body_indexes, device=device) for path in motion_files]
    lengths = [motion.time_step_total for motion in motions]
    self.motion_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    self.motion_starts = torch.cat(
      (
        torch.zeros(1, dtype=torch.long, device=device),
        self.motion_lengths.cumsum(0)[:-1],
      )
    )
    self.motion_ends = self.motion_starts + self.motion_lengths
    self.motion_files = motion_files
    self.joint_pos = torch.cat([motion.joint_pos for motion in motions])
    self.joint_vel = torch.cat([motion.joint_vel for motion in motions])
    self.body_pos_w = torch.cat([motion.body_pos_w for motion in motions])
    self.body_quat_w = torch.cat([motion.body_quat_w for motion in motions])
    self.body_lin_vel_w = torch.cat([motion.body_lin_vel_w for motion in motions])
    self.body_ang_vel_w = torch.cat([motion.body_ang_vel_w for motion in motions])
    self.time_step_total = self.joint_pos.shape[0]


class MotionSetCommand(MotionCommand):
  motion: MotionSetLoader

  def __init__(self, cfg: MotionSetCommandCfg, env) -> None:
    if cfg.sampling_mode != "uniform":
      raise ValueError("The oracle motion set currently supports uniform sampling only")
    if (
      not cfg.future_steps
      or cfg.future_steps[0] != 1
      or any(
        right <= left
        for left, right in zip(cfg.future_steps, cfg.future_steps[1:], strict=False)
      )
    ):
      raise ValueError("future_steps must start at 1 and be strictly increasing")
    motion_files = resolve_motion_files(cfg.motion_files)
    cfg.motion_file = motion_files[0]
    super().__init__(cfg, env)
    self.oracle_cfg = cfg
    self.motion = MotionSetLoader(motion_files, self.body_indexes, self.device)
    self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._sample_cursor = 0
    self.motion_end_steps = torch.full(
      (self.num_envs,), int(self.motion.motion_ends[0]), device=self.device
    )
    self.bin_count = len(motion_files)
    self.bin_failed_count = torch.zeros(self.bin_count, device=self.device)
    self._current_bin_failed = torch.zeros(self.bin_count, device=self.device)
    self.metrics["motion_id"] = torch.zeros(self.num_envs, device=self.device)
    self._foot_indexes = [
      self.cfg.body_names.index(name) for name in cfg.foot_body_names
    ]
    for name in (
      "error_foot_pos",
      "error_foot_rot",
      "error_foot_lin_vel",
      "error_foot_ang_vel",
    ):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

  @property
  def future_time_steps(self) -> torch.Tensor:
    offsets = torch.tensor(
      self.oracle_cfg.future_steps, dtype=torch.long, device=self.device
    )
    return torch.minimum(
      self.time_steps[:, None] + offsets, self.motion_end_steps[:, None] - 1
    )

  @property
  def goal_time_steps(self) -> torch.Tensor:
    """Reference frame reached by the action being selected."""
    return self.future_time_steps[:, 0]

  @property
  def future_joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.future_time_steps]

  @property
  def future_joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.future_time_steps]

  @property
  def future_body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.future_time_steps]
      + self._env.scene.env_origins[:, None, None, :]
    )

  @property
  def future_body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.future_time_steps]

  @property
  def future_body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.future_time_steps]

  @property
  def future_body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.future_time_steps]

  @property
  def goal_joint_pos(self) -> torch.Tensor:
    return self.future_joint_pos[:, 0]

  @property
  def goal_joint_vel(self) -> torch.Tensor:
    return self.future_joint_vel[:, 0]

  @property
  def goal_body_pos_w(self) -> torch.Tensor:
    return self.future_body_pos_w[:, 0]

  @property
  def goal_body_quat_w(self) -> torch.Tensor:
    return self.future_body_quat_w[:, 0]

  @property
  def goal_body_lin_vel_w(self) -> torch.Tensor:
    return self.future_body_lin_vel_w[:, 0]

  @property
  def goal_body_ang_vel_w(self) -> torch.Tensor:
    return self.future_body_ang_vel_w[:, 0]

  def _relative_goal_body_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
    anchor = self.motion_anchor_body_index
    reference_anchor_pos = self.goal_body_pos_w[:, anchor, None]
    reference_anchor_quat = self.goal_body_quat_w[:, anchor, None]
    robot_anchor_pos = self.robot_anchor_pos_w[:, None]
    robot_anchor_quat = self.robot_anchor_quat_w[:, None]
    origin = robot_anchor_pos.clone()
    origin[..., 2] = reference_anchor_pos[..., 2]
    rotation = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(reference_anchor_quat)))
    rotation = rotation.expand(-1, len(self.cfg.body_names), -1)
    return (
      origin + quat_apply(rotation, self.goal_body_pos_w - reference_anchor_pos),
      quat_mul(rotation, self.goal_body_quat_w),
    )

  def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
    if self.oracle_cfg.balanced_sampling:
      motion_ids = (
        torch.arange(len(env_ids), device=self.device) + self._sample_cursor
      ) % len(self.motion.motion_files)
      self._sample_cursor += len(env_ids)
    else:
      motion_ids = torch.randint(
        len(self.motion.motion_files), (len(env_ids),), device=self.device
      )
    lengths = self.motion.motion_lengths[motion_ids]
    available = (lengths - self.oracle_cfg.minimum_remaining_steps + 1).clamp(min=1)
    offsets = (torch.rand(len(env_ids), device=self.device) * available).long()
    self.motion_ids[env_ids] = motion_ids
    self.time_steps[env_ids] = self.motion.motion_starts[motion_ids] + offsets
    self.motion_end_steps[env_ids] = self.motion.motion_ends[motion_ids]
    self.metrics["sampling_entropy"][:] = 1.0
    self.metrics["sampling_top1_prob"][:] = 1.0 / len(self.motion.motion_files)
    self.metrics["sampling_top1_bin"][:] = 0.5

  def _update_metrics(self) -> None:
    goal_pos_relative_w, goal_quat_relative_w = self._relative_goal_body_pose()
    goal_anchor = self.motion_anchor_body_index
    self.metrics["error_anchor_pos"] = torch.linalg.vector_norm(
      self.goal_body_pos_w[:, goal_anchor] - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.goal_body_quat_w[:, goal_anchor], self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.linalg.vector_norm(
      self.goal_body_lin_vel_w[:, goal_anchor] - self.robot_anchor_lin_vel_w,
      dim=-1,
    )
    self.metrics["error_anchor_ang_vel"] = torch.linalg.vector_norm(
      self.goal_body_ang_vel_w[:, goal_anchor] - self.robot_anchor_ang_vel_w,
      dim=-1,
    )
    self.metrics["error_body_pos"] = torch.linalg.vector_norm(
      goal_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      goal_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)
    self.metrics["error_body_lin_vel"] = torch.linalg.vector_norm(
      self.goal_body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.linalg.vector_norm(
      self.goal_body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_joint_pos"] = torch.linalg.vector_norm(
      self.goal_joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.linalg.vector_norm(
      self.goal_joint_vel - self.robot_joint_vel, dim=-1
    )
    feet = self._foot_indexes
    self.metrics["motion_id"] = self.motion_ids.float()
    self.metrics["error_foot_pos"] = torch.linalg.vector_norm(
      self.goal_body_pos_w[:, feet] - self.robot_body_pos_w[:, feet], dim=-1
    ).amax(dim=-1)
    self.metrics["error_foot_rot"] = quat_error_magnitude(
      self.goal_body_quat_w[:, feet], self.robot_body_quat_w[:, feet]
    ).amax(dim=-1)
    self.metrics["error_foot_lin_vel"] = torch.linalg.vector_norm(
      self.goal_body_lin_vel_w[:, feet] - self.robot_body_lin_vel_w[:, feet],
      dim=-1,
    ).amax(dim=-1)
    self.metrics["error_foot_ang_vel"] = torch.linalg.vector_norm(
      self.goal_body_ang_vel_w[:, feet] - self.robot_body_ang_vel_w[:, feet],
      dim=-1,
    ).amax(dim=-1)

  def _update_command(self) -> None:
    self.time_steps += 1
    env_ids = torch.where(self.time_steps >= self.motion_end_steps)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)
      self._env.sim.forward()
    self.update_relative_body_poses()


@dataclass(kw_only=True)
class MotionSetCommandCfg(MotionCommandCfg):
  motion_files: tuple[str, ...]
  foot_body_names: tuple[str, ...]
  future_steps: tuple[int, ...] = (1, 2, 3, 4, 5)
  minimum_remaining_steps: int = 1
  balanced_sampling: bool = False

  def build(self, env) -> MotionSetCommand:
    return MotionSetCommand(self, env)
