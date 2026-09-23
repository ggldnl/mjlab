"""Endpoint command with measured feet and an aligned oracle reference."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.collect import (
  DEFAULT_ORACLE_DATASET,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.data import (
  GoalBodies,
  load_bodies,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.route import (
  touchdown_route,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  ROOT_STATE_DIM,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
  ImitationCommand,
  ImitationCommandCfg,
  Tolerances,
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


def rotation_6d(quat: torch.Tensor) -> torch.Tensor:
  return matrix_from_quat(quat)[..., :2, :].flatten(-2)


def foot_geometry(
  root: torch.Tensor,
  pos: torch.Tensor,
  quat: torch.Tensor,
  lin_vel: torch.Tensor,
  ang_vel: torch.Tensor,
) -> torch.Tensor:
  """Foot pose and velocity relative to the moving root frame."""
  root_quat = root[:, None, 3:7].expand_as(quat)
  offset = pos - root[:, None, :3]
  local_pos = quat_apply_inverse(root_quat, offset)
  local_quat = quat_mul(quat_conjugate(root_quat), quat)
  local_lin_vel = quat_apply_inverse(
    root_quat,
    lin_vel
    - root[:, None, 7:10]
    - torch.cross(root[:, None, 10:13].expand_as(offset), offset, dim=-1),
  )
  local_ang_vel = quat_apply_inverse(root_quat, ang_vel - root[:, None, 10:13])
  return torch.cat(
    (local_pos, rotation_6d(local_quat), local_lin_vel, local_ang_vel), dim=-1
  ).flatten(1)


def post_goal_features(
  target: torch.Tensor,
  future: torch.Tensor,
  contact: torch.Tensor,
  valid: torch.Tensor,
) -> torch.Tensor:
  """Encode a known entering-skill continuation in the terminal root frame."""
  orientation = target[:, None, 3:7].expand(*future.shape[:2], 4)
  frames = torch.cat(
    (
      quat_apply_inverse(orientation, future[..., :3] - target[:, None, :3]),
      rotation_6d(quat_mul(quat_conjugate(orientation), future[..., 3:7])),
      quat_apply_inverse(orientation, future[..., 7:10]),
      quat_apply_inverse(orientation, future[..., 10:13]),
      contact,
      valid[..., None].float(),
    ),
    dim=-1,
  )
  return (frames * valid[..., None]).flatten(1)


class GoalCommand(ImitationCommand):
  """Sample one physical crossing and expose its endpoint and dense route."""

  cfg: GoalCommandCfg

  def __init__(self, cfg: GoalCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    self._foot_indexes = [cfg.body_names.index(name) for name in cfg.foot_body_names]
    self.foot_body_indexes = self.body_indexes[self._foot_indexes]
    sensor = env.scene[cfg.contact_sensor]
    if not isinstance(sensor, ContactSensor):
      raise TypeError(f"{cfg.contact_sensor} must be a contact sensor")
    self.sensor = sensor
    self.bodies: GoalBodies | None = None
    if self.dataset is not None:
      assert cfg.dataset_path is not None
      self.bodies = load_bodies(cfg.dataset_path, str(self.device), cfg.split)
      if self.bodies.names != cfg.body_names:
        raise ValueError("Oracle dataset body order differs from the oracle model")
      if self.bodies.pos.shape[0] != self.dataset.states.shape[0]:
        raise ValueError("Oracle body and state row counts differ")
      if self.dataset.previous_action is None:
        raise ValueError("Oracle dataset needs previous_action")
    self._row_position: torch.Tensor | None = None
    if self.windows is not None:
      order = self.windows.order
      self._row_position = torch.empty_like(order)
      self._row_position[order] = torch.arange(len(order), device=self.device)

    self.target_foot = torch.zeros(self.num_envs, 30, device=self.device)
    self.target_foot_quat = torch.zeros(self.num_envs, 2, 4, device=self.device)
    self.target_foot_quat[..., 0] = 1.0
    self.target_contact = torch.zeros(self.num_envs, 2, device=self.device)
    self.target_future = torch.zeros(
      self.num_envs, cfg.post_goal_frames * 18, device=self.device
    )
    self.history = torch.zeros(
      self.num_envs, cfg.history_frames, 32, device=self.device
    )
    self.route_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.route_feet = torch.zeros(
      self.num_envs, cfg.route_slots, dtype=torch.long, device=self.device
    )
    self.posterior = torch.zeros(
      self.num_envs, cfg.posterior_waypoints * 41, device=self.device
    )
    self.initial: torch.Tensor | None = None

  @property
  def oracle_cfg(self) -> GoalCommandCfg:
    return self.cfg

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot_body_pos_w[:, 0]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot_body_quat_w[:, 0]

  def _future_rows(self) -> torch.Tensor:
    if self.dataset is None:
      raise ValueError("External bridge windows have no oracle route")
    offsets = torch.tensor(self.cfg.future_steps, device=self.device)
    ticks = torch.minimum(
      self.step[:, None] + offsets[None], self.window_steps[:, None]
    ).clamp(max=self.max_steps)
    return self.route_rows.gather(1, ticks)

  @property
  def future_joint_pos(self) -> torch.Tensor:
    assert self.dataset is not None
    return self.dataset.states[
      self._future_rows(), ROOT_STATE_DIM : ROOT_STATE_DIM + self.num_joints
    ]

  @property
  def future_joint_vel(self) -> torch.Tensor:
    assert self.dataset is not None
    start = ROOT_STATE_DIM + self.num_joints
    return self.dataset.states[self._future_rows(), start : start + self.num_joints]

  def _future_body(self, name: str) -> torch.Tensor:
    assert self.bodies is not None
    value = getattr(self.bodies, name)[self._future_rows()]
    rotation = self.route_rotation[:, None, None, :].expand(*value.shape[:-1], 4)
    if name == "pos":
      return self.route_origin[:, None, None] + quat_apply(
        rotation, value - self.route_start[:, None, None]
      )
    if name == "quat":
      return quat_mul(rotation, value)
    return quat_apply(rotation, value)

  @property
  def future_body_pos_w(self) -> torch.Tensor:
    return self._future_body("pos")

  @property
  def future_body_quat_w(self) -> torch.Tensor:
    return self._future_body("quat")

  @property
  def future_body_lin_vel_w(self) -> torch.Tensor:
    return self._future_body("lin_vel")

  @property
  def future_body_ang_vel_w(self) -> torch.Tensor:
    return self._future_body("ang_vel")

  def foot_contact_now(self) -> torch.Tensor:
    if self.sensor.data.found is None:
      raise ValueError("Foot contact sensor has no found data")
    return (self.sensor.data.found > 0).float()

  def foot_now(self) -> torch.Tensor:
    data = self.robot.data
    feet = self.foot_body_indexes
    return foot_geometry(
      self.state_now(),
      data.body_link_pos_w[:, feet],
      data.body_link_quat_w[:, feet],
      data.body_link_lin_vel_w[:, feet],
      data.body_link_ang_vel_w[:, feet],
    )

  def foot_errors(self) -> torch.Tensor:
    """Worst foot position, orientation, linear and angular velocity errors."""
    actual = self.foot_now().reshape(self.num_envs, 2, 15)
    target = self.target_foot.reshape(self.num_envs, 2, 15)
    root_quat = self.state_now()[:, None, 3:7].expand(self.num_envs, 2, 4)
    foot_quat = self.robot.data.body_link_quat_w[:, self.foot_body_indexes]
    local_quat = quat_mul(quat_conjugate(root_quat), foot_quat)
    return torch.stack(
      (
        torch.linalg.vector_norm(actual[..., :3] - target[..., :3], dim=-1).amax(-1),
        quat_error_magnitude(local_quat, self.target_foot_quat).amax(-1),
        torch.linalg.vector_norm(actual[..., 9:12] - target[..., 9:12], dim=-1).amax(
          -1
        ),
        torch.linalg.vector_norm(actual[..., 12:15] - target[..., 12:15], dim=-1).amax(
          -1
        ),
      ),
      dim=-1,
    )

  def actor_condition(self) -> torch.Tensor:
    current = self.state_now()
    target = self.target
    joints = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + self.num_joints)
    speeds = slice(joints.stop, joints.stop + self.num_joints)
    heading = yaw_quat(current[:, 3:7])
    remaining = (self.window_steps - self.step).clamp(min=0).float() / self.fps
    total = self.window_steps.float() / self.fps
    return torch.cat(
      (
        quat_apply_inverse(heading, target[:, :3] - current[:, :3]),
        rotation_6d(quat_mul(quat_conjugate(current[:, 3:7]), target[:, 3:7])),
        quat_apply_inverse(heading, target[:, 7:10] - current[:, 7:10]),
        quat_apply_inverse(heading, target[:, 10:13] - current[:, 10:13]),
        target[:, joints] - current[:, joints],
        target[:, speeds] - current[:, speeds],
        self.foot_now(),
        self.target_foot,
        self.foot_contact_now(),
        self.target_contact,
        self.target_future,
        self.history.flatten(1),
        remaining[:, None],
        total[:, None],
      ),
      dim=-1,
    )

  @property
  def command(self) -> torch.Tensor:
    return self.actor_condition()

  def initial_condition(self) -> torch.Tensor:
    if self.initial is None:
      self.initial = self._initial_features().clone()
    return self.initial

  def _initial_features(self) -> torch.Tensor:
    state = self.state_now()
    quat = state[:, 3:7]
    return torch.cat(
      (
        self.actor_condition(),
        state[:, 2:3],
        quat_apply_inverse(quat, state[:, 7:10]),
        quat_apply_inverse(quat, state[:, 10:13]),
        self.robot.data.projected_gravity_b,
        state[:, ROOT_STATE_DIM:],
        self._env.action_manager.action,
      ),
      dim=-1,
    )

  def route_label(self) -> torch.Tensor:
    return torch.cat(
      (self.route_count[:, None].float(), self.route_feet.float()), dim=-1
    )

  def posterior_path(self) -> torch.Tensor:
    return self.posterior

  def _record_route(self, env_ids: torch.Tensor) -> None:
    assert self.dataset is not None and self.bodies is not None
    path_rows = self.route_rows[env_ids]
    contact = self.bodies.contact[path_rows]
    count, feet = touchdown_route(
      contact, self.window_steps[env_ids], self.cfg.route_slots
    )
    self.route_count[env_ids] = count
    self.route_feet[env_ids] = feet
    target_rows = path_rows.gather(1, self.window_steps[env_ids, None]).squeeze(1)
    target_raw = self.dataset.states[target_rows]
    indexes = self._foot_indexes
    self.target_foot[env_ids] = foot_geometry(
      target_raw,
      self.bodies.pos[target_rows][:, indexes],
      self.bodies.quat[target_rows][:, indexes],
      self.bodies.lin_vel[target_rows][:, indexes],
      self.bodies.ang_vel[target_rows][:, indexes],
    )
    self.target_contact[env_ids] = self.bodies.contact[target_rows]
    assert self.windows is not None and self._row_position is not None
    order = self.windows.order
    offsets_after = torch.arange(1, self.cfg.post_goal_frames + 1, device=self.device)
    positions = self._row_position[target_rows, None] + offsets_after[None]
    after_rows = order[positions.clamp(max=len(order) - 1)]
    valid = (
      (positions < len(order))
      & (
        self.dataset.trajectory[after_rows]
        == self.dataset.trajectory[target_rows, None]
      )
      & (
        self.dataset.frame[after_rows]
        == self.dataset.frame[target_rows, None] + offsets_after[None]
      )
    )
    after_rows = torch.where(valid, after_rows, target_rows[:, None])
    self.target_future[env_ids] = post_goal_features(
      target_raw,
      self.dataset.states[after_rows],
      self.bodies.contact[after_rows],
      valid,
    )
    root_quat = target_raw[:, None, 3:7].expand(len(env_ids), 2, 4)
    self.target_foot_quat[env_ids] = quat_mul(
      quat_conjugate(root_quat), self.bodies.quat[target_rows][:, indexes]
    )

    fractions = torch.linspace(
      1 / self.cfg.posterior_waypoints,
      1,
      self.cfg.posterior_waypoints,
      device=self.device,
    )
    offsets = torch.ceil(self.window_steps[env_ids, None] * fractions).long()
    rows = path_rows.gather(1, offsets)
    start = self.dataset.states[path_rows[:, 0]]
    waypoints = self.dataset.states[rows]
    start_quat = start[:, None, 3:7].expand(*waypoints.shape[:2], 4)
    root_pos = quat_apply_inverse(start_quat, waypoints[..., :3] - start[:, None, :3])
    root_ori = rotation_6d(quat_mul(quat_conjugate(start_quat), waypoints[..., 3:7]))
    foot_pos = self.bodies.pos[rows][:, :, indexes]
    foot_quat = self.bodies.quat[rows][:, :, indexes]
    foot_lin = self.bodies.lin_vel[rows][:, :, indexes]
    foot_ang = self.bodies.ang_vel[rows][:, :, indexes]
    path_quat = start[:, None, None, 3:7].expand_as(foot_quat)
    foot_path = torch.cat(
      (
        quat_apply_inverse(path_quat, foot_pos - start[:, None, None, :3]),
        rotation_6d(quat_mul(quat_conjugate(path_quat), foot_quat)),
        quat_apply_inverse(path_quat, foot_lin),
        quat_apply_inverse(path_quat, foot_ang),
      ),
      dim=-1,
    ).flatten(2)
    self.posterior[env_ids] = torch.cat(
      (root_pos, root_ori, foot_path, self.bodies.contact[rows]), dim=-1
    ).flatten(1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self.history[env_ids] = 0
    if self.dataset is not None:
      assert self.dataset.previous_action is not None
      self._record_route(env_ids)
      starts = self.route_rows[env_ids, 0]
      self._env.action_manager.initialize_action(
        self.dataset.previous_action[starts], env_ids
      )
      self._perturb_start(env_ids)
    self._env.sim.forward()
    latest = torch.cat((self.foot_now(), self.foot_contact_now()), dim=-1)
    self.history[env_ids] = latest[env_ids, None]
    if self.initial is not None:
      self.initial[env_ids] = self._initial_features()[env_ids]

  def _perturb_start(self, env_ids: torch.Tensor) -> None:
    cfg = self.cfg
    if cfg.start_perturb_prob <= 0:
      return
    selected = env_ids[
      torch.rand(len(env_ids), device=self.device) < cfg.start_perturb_prob
    ]
    if not len(selected):
      return
    state = self.state_now()[selected].clone()
    state[:, :2] += torch.randn(len(selected), 2, device=self.device) * cfg.start_xy_std
    axis = torch.zeros(len(selected), 3, device=self.device)
    axis[:, 2] = 1.0
    yaw = torch.randn(len(selected), device=self.device) * cfg.start_yaw_std
    state[:, 3:7] = quat_mul(quat_from_angle_axis(yaw, axis), state[:, 3:7])
    state[:, 7:13] += (
      torch.randn(len(selected), 6, device=self.device) * cfg.start_root_velocity_std
    )
    joints = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + self.num_joints)
    speeds = slice(joints.stop, joints.stop + self.num_joints)
    state[:, joints] += (
      torch.randn(len(selected), self.num_joints, device=self.device)
      * cfg.start_joint_pos_std
    )
    state[:, speeds] += (
      torch.randn(len(selected), self.num_joints, device=self.device)
      * cfg.start_joint_vel_std
    )
    self._write_initial_state(selected, state)

  def _update_command(self) -> None:
    self.history[:, :-1] = self.history[:, 1:].clone()
    self.history[:, -1] = torch.cat((self.foot_now(), self.foot_contact_now()), dim=-1)

  def aim(
    self,
    target: torch.Tensor,
    *,
    target_foot: torch.Tensor | None = None,
    target_foot_quat: torch.Tensor | None = None,
    target_contact: torch.Tensor | None = None,
    target_future: torch.Tensor | None = None,
  ) -> None:
    if self.dataset is None:
      if target_foot is None or target_foot_quat is None or target_contact is None:
        raise ValueError("External goals need target foot state and contact")
      if target_foot.shape != self.target_foot.shape:
        raise ValueError("target_foot must have shape (num_envs, 30)")
      if target_contact.shape != self.target_contact.shape:
        raise ValueError("target_contact must have shape (num_envs, 2)")
      if target_foot_quat.shape != self.target_foot_quat.shape:
        raise ValueError("target_foot_quat must have shape (num_envs, 2, 4)")
      self.target_foot[:] = target_foot
      self.target_foot_quat[:] = target_foot_quat
      self.target_contact[:] = target_contact
      if target_future is not None:
        if target_future.shape != self.target_future.shape:
          raise ValueError("target_future has the wrong shape")
        self.target_future[:] = target_future
      else:
        self.target_future.zero_()
    super().aim(target)

  def open_window(
    self, env_ids: torch.Tensor, target: torch.Tensor, duration_s: torch.Tensor
  ) -> None:
    super().open_window(env_ids, target, duration_s)
    if self.initial is None:
      self.initial = self._initial_features().clone()
    else:
      self.initial[env_ids] = self._initial_features()[env_ids]


@dataclass(kw_only=True)
class GoalCommandCfg(ImitationCommandCfg):
  dataset_path: Path | None = DEFAULT_ORACLE_DATASET
  tolerances: Tolerances = field(
    default_factory=lambda: Tolerances(upper_joint_pos=0.3, upper_joint_vel=3.0)
  )
  duration_s_range: tuple[float, float] = (0.3, 2.0)
  body_names: tuple[str, ...]
  foot_body_names: tuple[str, str]
  contact_sensor: str = "feet_ground_contact"
  future_steps: tuple[int, ...] = (1, 2, 3, 4, 5)
  history_frames: int = 8
  posterior_waypoints: int = 8
  route_slots: int = 6
  post_goal_frames: int = 5
  foot_tolerances: tuple[float, float, float, float] = (0.06, 0.10, 0.30, 0.60)
  start_perturb_prob: float = 0.5
  start_xy_std: float = 0.015
  start_yaw_std: float = 0.03
  start_root_velocity_std: float = 0.05
  start_joint_pos_std: float = 0.015
  start_joint_vel_std: float = 0.10

  def build(self, env: ManagerBasedRlEnv) -> GoalCommand:
    if (
      min(
        self.history_frames,
        self.posterior_waypoints,
        self.route_slots,
        self.post_goal_frames,
      )
      < 1
    ):
      raise ValueError(
        "History, posterior waypoints, route slots and post-goal frames must be positive"
      )
    if not self.future_steps or self.future_steps[0] != 1:
      raise ValueError("future_steps must start at 1")
    if not math.isfinite(self.duration_s_range[1]):
      raise ValueError("duration range must be finite")
    return GoalCommand(self, env)
