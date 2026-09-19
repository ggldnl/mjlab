"""Endpoint command and privileged path for CVAE distillation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DEFAULT_DATASET,
  ROOT_STATE_DIM,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
  ImitationCommand,
  ImitationCommandCfg,
)
from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)


def _rotation_6d(quaternion: torch.Tensor) -> torch.Tensor:
  from mjlab.utils.lab_api.math import matrix_from_quat

  matrix = matrix_from_quat(quaternion)
  return matrix[..., :, :2].transpose(-1, -2).flatten(-2)


class CvaeCommand(ImitationCommand):
  """Sample a reachable endpoint and retain its demonstrated crossing."""

  def __init__(self, cfg: CvaeCommandCfg, env: ManagerBasedRlEnv) -> None:
    if cfg.dataset_path is not None:
      motion_cfg = env.cfg.commands[cfg.motion_name]
      motion_file = getattr(motion_cfg, "motion_file", "")
      if cfg.sources is None and motion_file:
        cfg.sources = (Path(motion_file).stem,)
    super().__init__(cfg, env)
    if self.dataset is not None:
      # ponytail: one teacher per run, add source-routed teachers after this baseline works
      if cfg.sources is None or len(cfg.sources) != 1:
        raise ValueError("CVAE DAgger currently trains one tracker source per run")
      if self.dataset.previous_action is None:
        raise ValueError("Rebuild the tracker dataset with previous_action metadata")
      if self.dataset.foot_contact is None:
        raise ValueError("Rebuild the tracker dataset with foot_contact metadata")
      if self.dataset.phase is None:
        raise ValueError("Rebuild the tracker dataset with motion phase metadata")

  @property
  def motion(self) -> MotionCommand:
    term = self._env.command_manager.get_term(self.cvae_cfg.motion_name)
    if not isinstance(term, MotionCommand):
      raise TypeError(f"'{self.cvae_cfg.motion_name}' is not a MotionCommand")
    return term

  def _target_contacts(self, rows: torch.Tensor) -> torch.Tensor:
    if self.dataset is None or self.dataset.foot_contact is None:
      return self.target.new_zeros((rows.shape[0], 2))
    return self.dataset.foot_contact[rows]

  def _encode_target(
    self,
    current: torch.Tensor,
    target: torch.Tensor,
    contact: torch.Tensor,
    time_to_target: torch.Tensor,
    total_time: torch.Tensor,
  ) -> torch.Tensor:
    q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + self.num_joints)
    qd = slice(q.stop, q.stop + self.num_joints)
    heading = yaw_quat(current[:, 3:7])
    target_linear_velocity = quat_apply_inverse(heading, target[:, 7:10])
    target_angular_velocity = quat_apply_inverse(heading, target[:, 10:13])
    current_linear_velocity = quat_apply_inverse(heading, current[:, 7:10])
    current_angular_velocity = quat_apply_inverse(heading, current[:, 10:13])
    default_joint_pos = self.robot.data.default_joint_pos
    assert default_joint_pos is not None
    root_height = target[:, 2] - self._env.scene.env_origins[:, 2]
    return torch.cat(
      (
        quat_apply_inverse(heading, target[:, :3] - current[:, :3]),
        _rotation_6d(quat_mul(quat_conjugate(current[:, 3:7]), target[:, 3:7])),
        root_height[:, None],
        target_linear_velocity,
        target_linear_velocity - current_linear_velocity,
        target_angular_velocity,
        target_angular_velocity - current_angular_velocity,
        target[:, q] - default_joint_pos,
        target[:, q] - current[:, q],
        target[:, qd],
        target[:, qd] - current[:, qd],
        contact,
        time_to_target[:, None],
        total_time[:, None],
      ),
      dim=-1,
    )

  @property
  def command(self) -> torch.Tensor:
    current = self.state_now()
    target_rows = self.route_rows.gather(1, self.window_steps[:, None]).squeeze(1)
    remaining = (self.window_steps - self.step).clamp(min=0).float() / self.fps
    total = self.window_steps.float() / self.fps
    return self._encode_target(
      current, self.target, self._target_contacts(target_rows), remaining, total
    )

  def posterior_path(self) -> torch.Tensor:
    """Evenly spaced demonstrated waypoints, expressed from the live state."""
    if self.dataset is None:
      return self.target.new_zeros((self.num_envs, 0))
    current = self.state_now()
    remaining = (self.window_steps - self.step).clamp(min=0)
    fractions = (
      torch.arange(
        1,
        self.cvae_cfg.posterior_waypoints + 1,
        device=self.device,
        dtype=torch.float32,
      )
      / self.cvae_cfg.posterior_waypoints
    )
    offsets = torch.ceil(remaining[:, None] * fractions).long()
    ticks = (self.step[:, None] + offsets).clamp(max=self.max_steps)
    rows = self.route_rows.gather(1, ticks)
    total = remaining.float() / self.fps
    encoded = []
    for index in range(self.cvae_cfg.posterior_waypoints):
      target = self._place_state(self.dataset.states[rows[:, index]])
      encoded.append(
        self._encode_target(
          current,
          target,
          self._target_contacts(rows[:, index]),
          offsets[:, index].float() / self.fps,
          total,
        )
      )
    return torch.cat(encoded, dim=-1)

  @property
  def cvae_cfg(self) -> CvaeCommandCfg:
    assert isinstance(self.cfg, CvaeCommandCfg)
    return self.cfg

  def teacher_anchor(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Original motion anchor placed in the sampled bridge frame."""
    anchor_pos = self.motion.motion.body_pos_w[
      self.motion.time_steps, self.motion.motion_anchor_body_index
    ]
    anchor_quat = self.motion.motion.body_quat_w[
      self.motion.time_steps, self.motion.motion_anchor_body_index
    ]
    placed_pos = self.route_origin + quat_apply(
      self.route_rotation, anchor_pos - self.route_start
    )
    placed_quat = quat_mul(self.route_rotation, anchor_quat)
    return placed_pos, placed_quat

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    if self.dataset is None or self.dataset.phase is None:
      return
    start_rows = self.route_rows[env_ids, 0]
    self.motion.time_steps[env_ids] = self.dataset.phase[start_rows]
    assert self.dataset.previous_action is not None
    self._env.action_manager.initialize_action(
      self.dataset.previous_action[start_rows], env_ids
    )


@dataclass(kw_only=True)
class CvaeCommandCfg(ImitationCommandCfg):
  dataset_path: Path | None = DEFAULT_DATASET
  duration_s_range: tuple[float, float] = (0.5, 2.0)
  posterior_waypoints: int = 8
  motion_name: str = "motion"

  def build(self, env: ManagerBasedRlEnv) -> CvaeCommand:
    if self.posterior_waypoints < 1:
      raise ValueError("posterior_waypoints must be positive")
    return CvaeCommand(self, env)
