"""The MDP terms a goal-conditioned primitive motion skill is assembled from.

A skill here is a DeepMimic-style tracker with a goal bolted on. It is handed a reference
clip from its own skill's library and paid for reproducing it (pose, velocity,
end-effectors, root), and it is *also* handed the goal that clip realizes and paid for
realizing it. The tracking term is what makes the motion look like walking or running or
jumping; the goal term is what makes the skill addressable, so a controller or a bridge
can later ask for a particular walk rather than for whatever the reference happened to be
doing.

Both halves matter, and the second is the one that quietly goes missing. A goal that
reaches the observation but never the reward is not a goal: the policy is free to ignore
the channel and reproduce whatever motion the proprioception already implies, and the
result looks conditioned right up until the moment something tries to steer it. So every
goal channel here has a reward term that reads it, and `parkour_env_cfg` wires the ones
each skill actually has.

Everything is expressed without global position or heading, which is what `dataset.py`
normalized the clips into. The reference carries root height, roll and pitch, and
velocities in the root's own yaw frame; the goal carries a displacement relative to where
the episode started. Nothing anywhere says where on the plane the robot is or which way
it is pointing, so the same skill means the same thing everywhere.

The pieces:

- `SkillMotionCommand` owns the per-env clip, the frame cursor, and the goal. It writes
  the reference pose into the sim on reset (with the randomization the cfg asks for),
  advances the cursor each step, and reports when a clip has run out.
- the `reference_*` and `goal_*` observations are what the policy reads.
- the `track_*` rewards are the DeepMimic terms; `goal_*` are the conditioning terms.
- `motion_finished` ends the episode when the clip does, which is not optional: a
  reference that silently loops back to frame 0 hands the policy a teleport it is then
  penalized for failing to track.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.skills.experiments.parkour.motions import ClipLibrary, ReferenceFrames
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")

# How a goal channel's name says what it means. A clip library states its channel names,
# so the command term works out which reward terms have anything to read rather than the
# experiment having to keep an index table in step with the dataset.
_DX_NAMES = ("goal_dx", "land_dx")
_DY_NAMES = ("goal_dy", "land_dy")
_DYAW_NAMES = ("goal_dyaw",)
_APEX_NAMES = ("apex_height",)


def _yaw_of(quat: torch.Tensor) -> torch.Tensor:
  """Yaw angle [rad] of a batch of wxyz quaternions."""
  w, x, y, z = quat.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate_z(vectors: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  """Rotate xy(z) vectors about z by `yaw`."""
  cos, sin = torch.cos(yaw), torch.sin(yaw)
  x, y = vectors[..., 0], vectors[..., 1]
  out = torch.empty_like(vectors)
  out[..., 0] = cos * x - sin * y
  out[..., 1] = sin * x + cos * y
  if vectors.shape[-1] > 2:
    out[..., 2] = vectors[..., 2]
  return out


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
  return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


##
# The command: which clip, where in it, and what goal it realizes
##


class SkillMotionCommand(CommandTerm):
  """A reference clip, a cursor into it, and the goal that clip realizes.

  One skill's whole conditioning lives here. On reset an env draws a clip and a start
  frame, the robot is placed at that frame (perturbed by whatever the cfg asks for), and
  the goal is read off the clip *at that frame* and then held fixed for the episode --
  which is what makes the randomization a randomization rather than a different task: the
  robot starts from a different place each time and is asked for the same thing.
  """

  cfg: SkillMotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: SkillMotionCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.library = ClipLibrary(cfg.clip_dir, str(self.device))
    print(f"[motion] {cfg.clip_dir}: {self.library.describe()}")

    if self.library.joint_dim != self.robot.data.joint_pos.shape[1]:
      raise ValueError(
        f"The clips in {cfg.clip_dir} have {self.library.joint_dim} joints but the "
        f"robot has {self.robot.data.joint_pos.shape[1]}. Rebuild the dataset."
      )

    self.ee_indexes = torch.tensor(
      [self.robot.body_names.index(name) for name in cfg.end_effector_names],
      dtype=torch.long,
      device=self.device,
    )
    if len(self.ee_indexes) != self.library.num_ee:
      raise ValueError(
        f"The clips store {self.library.num_ee} end-effector offsets but the cfg names "
        f"{len(self.ee_indexes)}. They must be the same bodies, in the same order."
      )

    channels = self.library.goal_channels
    self.dx_channel = _first_index(channels, _DX_NAMES)
    self.dy_channel = _first_index(channels, _DY_NAMES)
    self.dyaw_channel = _first_index(channels, _DYAW_NAMES)
    self.apex_channel = _first_index(channels, _APEX_NAMES)

    num_envs = self.num_envs
    self.clip_id = torch.zeros(num_envs, dtype=torch.long, device=self.device)
    self.frame = torch.zeros(num_envs, dtype=torch.long, device=self.device)
    self.start_frame = torch.zeros(num_envs, dtype=torch.long, device=self.device)
    self.goal = torch.zeros(num_envs, self.library.goal_dim, device=self.device)
    self.finished = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
    # `compute` runs `_update_command` on every call, including the `compute(dt=0.0)`
    # that follows a reset. Without this the cursor would advance a frame before the
    # first observation, so the robot would be standing at the reference frame it was
    # placed at while the reference already pointed at the next one -- a free tracking
    # error at every episode start, on the frames a jump can least afford it.
    self._just_resampled = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    # Where the episode started, so a displacement goal can be measured against it.
    self.spawn_pos_w = torch.zeros(num_envs, 3, device=self.device)
    self.spawn_yaw = torch.zeros(num_envs, device=self.device)
    # The reference root height at the start frame, which is what an apex goal is above.
    self.stance_height = torch.zeros(num_envs, device=self.device)
    self.peak_height = torch.zeros(num_envs, device=self.device)

    self.metrics["error_joint_pos"] = torch.zeros(num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(num_envs, device=self.device)
    self.metrics["error_ee_pos"] = torch.zeros(num_envs, device=self.device)
    self.metrics["error_root_height"] = torch.zeros(num_envs, device=self.device)
    self.metrics["goal_remaining"] = torch.zeros(num_envs, device=self.device)
    self.metrics["clip_progress"] = torch.zeros(num_envs, device=self.device)

  ##
  # What the policy is conditioned on.
  ##

  @property
  def command(self) -> torch.Tensor:
    """The goal, fixed for the episode. This is the conditioning signal."""
    return self.goal

  @property
  def reference(self) -> ReferenceFrames:
    """The reference state at each env's current frame."""
    return self.library.frames_at(self.clip_id, self.frame)

  @property
  def clip_length(self) -> torch.Tensor:
    return self.library.lengths[self.clip_id]

  @property
  def phase(self) -> torch.Tensor:
    """(N,) how far through its clip each env is, in [0, 1]."""
    return self.frame.float() / (self.clip_length.float() - 1.0).clamp_min(1.0)

  def future_joint_pos(self, offset: int) -> torch.Tensor:
    """(N, J) reference joint positions `offset` control steps ahead."""
    return self.library.joint_pos_at(self.clip_id, self.frame + offset)

  ##
  # Where the robot is, in the frames the goal is stated in.
  ##

  @property
  def robot_yaw(self) -> torch.Tensor:
    return _yaw_of(self.robot.data.root_link_quat_w)

  @property
  def achieved_displacement(self) -> torch.Tensor:
    """(N, 2) how far the robot has travelled since reset, in the spawn yaw frame."""
    delta = self.robot.data.root_link_pos_w - self.spawn_pos_w
    return _rotate_z(delta[:, :2], -self.spawn_yaw)

  @property
  def achieved_heading(self) -> torch.Tensor:
    """(N,) how far the robot has turned since reset."""
    return _wrap_to_pi(self.robot_yaw - self.spawn_yaw)

  @property
  def goal_displacement(self) -> torch.Tensor:
    """(N, 2) the displacement the goal asks for, in the spawn yaw frame."""
    if self.dx_channel is None or self.dy_channel is None:
      return torch.zeros(self.num_envs, 2, device=self.device)
    return self.goal[:, [self.dx_channel, self.dy_channel]]

  @property
  def displacement_error(self) -> torch.Tensor:
    """(N, 2) how much of the displacement goal is left, in the robot's own yaw frame.

    In the robot's frame rather than the spawn frame because that is the only one the
    policy can act in: "two metres that way" has to be stated relative to where the robot
    is facing now, or it is asking the policy to know its own heading, which the
    observation deliberately does not tell it.
    """
    remaining_spawn = self.goal_displacement - self.achieved_displacement
    return _rotate_z(remaining_spawn, self.spawn_yaw - self.robot_yaw)

  @property
  def heading_error(self) -> torch.Tensor:
    """(N,) how much of the heading goal is left."""
    if self.dyaw_channel is None:
      return torch.zeros(self.num_envs, device=self.device)
    return _wrap_to_pi(self.goal[:, self.dyaw_channel] - self.achieved_heading)

  @property
  def apex_error(self) -> torch.Tensor:
    """(N,) commanded apex minus the highest the root has actually reached.

    Positive means the robot has not jumped high enough yet, and it shrinks to zero as
    the jump happens rather than being a one-off event, so it can be rewarded densely.
    Negative means it overshot, which is penalized the same way.
    """
    if self.apex_channel is None:
      return torch.zeros(self.num_envs, device=self.device)
    achieved = self.peak_height - self.stance_height
    return self.goal[:, self.apex_channel] - achieved

  ##
  # Lifecycle.
  ##

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    num = len(env_ids)
    if num == 0:
      return

    clip_ids = self.library.sample_clips(num)
    frames = self.library.sample_start_frames(clip_ids, self.cfg.max_start_fraction)
    self.clip_id[env_ids] = clip_ids
    self.frame[env_ids] = frames
    self.start_frame[env_ids] = frames
    self.goal[env_ids] = self.library.goal_at(clip_ids, frames)
    self.finished[env_ids] = False

    reference = self.library.frames_at(clip_ids, frames)
    root_pos, root_quat = self._write_reference_state(env_ids, reference)

    # Taken from what was written, not read back off the entity. Derived quantities like
    # `root_link_pos_w` are stale until the env's next `sim.forward()`, and reading them
    # in the same call that wrote the state is the one thing mjlab's own docs tell you
    # not to do -- it would silently anchor the goal to the *previous* episode's pose.
    self.spawn_pos_w[env_ids] = root_pos
    self.spawn_yaw[env_ids] = _yaw_of(root_quat)
    self.stance_height[env_ids] = reference.root_height
    self.peak_height[env_ids] = root_pos[:, 2]
    self._just_resampled[env_ids] = True

  def _write_reference_state(
    self, env_ids: torch.Tensor, reference: ReferenceFrames
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Put the robot at a reference frame, perturbed by the cfg's randomization.

    The clip is normalized, so it says nothing about where on the plane to stand or
    which way to face. Both are chosen here: the env's own origin, and a yaw drawn
    uniformly. Neither is observed and neither is rewarded, so the draw costs nothing and
    keeps a thousand envs from being visually identical.

    Returns the root position and orientation it wrote, since the caller needs them and
    cannot read them back off the entity until the env forwards the sim.
    """
    num = len(env_ids)
    origins = self._env.scene.env_origins[env_ids]

    root_pos = origins.clone()
    root_pos[:, 2] = reference.root_height
    heading = sample_uniform(-torch.pi, torch.pi, (num,), device=self.device)
    root_quat = quat_mul(_quat_from_yaw(heading), reference.root_quat)

    pose_noise = _sampled_range(self.cfg.pose_range, num, self.device)
    root_pos = root_pos + pose_noise[:, 0:3]
    root_quat = quat_mul(
      quat_from_euler_xyz(pose_noise[:, 3], pose_noise[:, 4], pose_noise[:, 5]),
      root_quat,
    )

    # The clip's velocities are in its own yaw frame; the sim wants them in the world,
    # so they follow whatever heading was just drawn.
    lin_vel = _rotate_z(reference.root_lin_vel_b, heading)
    ang_vel = _rotate_z(reference.root_ang_vel_b, heading)
    vel_noise = _sampled_range(self.cfg.velocity_range, num, self.device)
    lin_vel = lin_vel + vel_noise[:, 0:3]
    ang_vel = ang_vel + vel_noise[:, 3:6]

    joint_pos = reference.joint_pos + sample_uniform(
      self.cfg.joint_position_range[0],
      self.cfg.joint_position_range[1],
      reference.joint_pos.shape,
      device=self.device,
    )
    limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(limits[:, :, 0], limits[:, :, 1])

    self.robot.write_joint_state_to_sim(joint_pos, reference.joint_vel, env_ids=env_ids)
    self.robot.write_root_state_to_sim(
      torch.cat([root_pos, root_quat, lin_vel, ang_vel], dim=-1), env_ids=env_ids
    )
    self.robot.reset(env_ids=env_ids)
    return root_pos, root_quat

  def _update_command(self) -> None:
    # An env resampled on this very call stays where it was put: it has not lived a step
    # yet, so there is no step to advance over.
    advance = ~self._just_resampled
    self.frame = self.frame + advance.long()
    self._just_resampled[:] = False
    # A clip that has run out is an episode that is over. The cursor is held at the last
    # frame so the reference is still well defined for the step between this and the
    # termination manager acting on it; `motion_finished` is what ends the episode.
    self.finished = self.frame >= self.clip_length
    self.frame = torch.minimum(self.frame, self.clip_length - 1)
    self.peak_height = torch.maximum(
      self.peak_height, self.robot.data.root_link_pos_w[:, 2]
    )

  def _update_metrics(self) -> None:
    reference = self.reference
    self.metrics["error_joint_pos"] = torch.norm(
      reference.joint_pos - self.robot.data.joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      reference.joint_vel - self.robot.data.joint_vel, dim=-1
    )
    self.metrics["error_ee_pos"] = torch.norm(
      reference.ee_offsets - robot_ee_offsets(self.robot, self.ee_indexes), dim=-1
    ).mean(dim=-1)
    self.metrics["error_root_height"] = (
      reference.root_height - self.robot.data.root_link_pos_w[:, 2]
    ).abs()
    self.metrics["goal_remaining"] = torch.norm(self.displacement_error, dim=-1)
    self.metrics["clip_progress"] = self.phase


def _first_index(channels: tuple[str, ...], names: tuple[str, ...]) -> int | None:
  for name in names:
    if name in channels:
      return channels.index(name)
  return None


def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
  half = 0.5 * yaw
  zeros = torch.zeros_like(half)
  return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)


def _sampled_range(
  ranges: dict[str, tuple[float, float]], num: int, device: str
) -> torch.Tensor:
  """(num, 6) samples for x, y, z, roll, pitch, yaw from a partially-filled dict."""
  keys = ("x", "y", "z", "roll", "pitch", "yaw")
  bounds = torch.tensor(
    [ranges.get(key, (0.0, 0.0)) for key in keys], device=device, dtype=torch.float32
  )
  return sample_uniform(bounds[:, 0], bounds[:, 1], (num, 6), device=device)


@dataclass(kw_only=True)
class SkillMotionCommandCfg(CommandTermCfg):
  entity_name: str
  clip_dir: str
  """Folder of normalized clips for this skill, written by `dataset.py`."""
  end_effector_names: tuple[str, ...]
  """The bodies whose offsets the clips store, in the same order."""

  max_start_fraction: float = 0.7
  """Episodes start uniformly in the first this-much of a clip. Short of the end so
  there is always motion left to track and goal left to realize."""

  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  """Per-axis noise on the spawn pose. Small: it is meant to stop the policy from only
  ever seeing the exact reference pose, not to teach recovery."""

  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  """Per-axis noise on the spawn velocity."""

  joint_position_range: tuple[float, float] = (-0.05, 0.05)
  """Noise on the spawn joint positions [rad]."""

  def build(self, env: ManagerBasedRlEnv) -> SkillMotionCommand:
    return SkillMotionCommand(self, env)


##
# Shared readings off the live robot
##


def robot_ee_offsets(robot: Entity, ee_indexes: torch.Tensor) -> torch.Tensor:
  """(N, E, 3) end-effector positions relative to the root, in the root's yaw frame.

  The same quantity `dataset.py` stored per reference frame, computed the same way, so
  comparing them compares the pose and not two different derivations of it.
  """
  root_pos = robot.data.root_link_pos_w
  heading = yaw_quat(robot.data.root_link_quat_w)
  offsets = robot.data.body_link_pos_w[:, ee_indexes] - root_pos[:, None, :]
  num_ee = offsets.shape[1]
  return quat_apply_inverse(heading.unsqueeze(1).expand(-1, num_ee, -1), offsets)


def _command(env: ManagerBasedRlEnv, command_name: str) -> SkillMotionCommand:
  return cast(SkillMotionCommand, env.command_manager.get_term(command_name))


##
# Observations
##


def reference_joint_pos_error(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """(N, J) reference joint positions minus the robot's.

  The error rather than the raw reference, because the policy needs the difference and
  handing it the subtraction already done is one fewer thing for the first layer to
  spend capacity on.
  """
  command = _command(env, command_name)
  return command.reference.joint_pos - command.robot.data.joint_pos


def reference_joint_vel_error(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  command = _command(env, command_name)
  return command.reference.joint_vel - command.robot.data.joint_vel


def reference_root_velocity(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """(N, 6) reference root linear and angular velocity, in the root's own yaw frame."""
  reference = _command(env, command_name).reference
  return torch.cat([reference.root_lin_vel_b, reference.root_ang_vel_b], dim=-1)


def reference_root_pose(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """(N, 4) reference root height and the gravity direction in the reference root frame.

  Height and gravity rather than a quaternion: they are what the reward compares, they
  are heading-free by construction, and they are the same two quantities the robot's own
  proprioception reports.
  """
  command = _command(env, command_name)
  reference = command.reference
  gravity = command.robot.data.gravity_vec_w
  return torch.cat(
    [
      reference.root_height.unsqueeze(-1),
      quat_apply_inverse(reference.root_quat, gravity),
    ],
    dim=-1,
  )


def reference_future_joint_pos(
  env: ManagerBasedRlEnv, command_name: str, offsets: tuple[int, ...]
) -> torch.Tensor:
  """(N, J * len(offsets)) where the reference is heading, as errors from the pose now.

  Showing the policy a few frames ahead rather than only the current one is what lets it
  prepare a motion instead of chasing it, and it is the single change that moves tracking
  quality the most on fast behaviors -- a jump's takeoff has to be set up before the
  frame that asks for it.
  """
  command = _command(env, command_name)
  current = command.robot.data.joint_pos
  return torch.cat(
    [command.future_joint_pos(offset) - current for offset in offsets], dim=-1
  )


def motion_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """(N, 3) where in the clip this is: sin, cos, and the fraction still to come.

  Sin and cos rather than the raw fraction so the signal is smooth, and the remaining
  fraction alongside because a jump is not periodic and "how much is left" is the part
  that matters there.
  """
  command = _command(env, command_name)
  phase = command.phase
  angle = 2.0 * torch.pi * phase
  return torch.stack([torch.sin(angle), torch.cos(angle), 1.0 - phase], dim=-1)


def goal_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """(N, G) the goal this episode was given, unchanged throughout it."""
  return _command(env, command_name).command


def goal_remaining(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """(N, 4) how much of the goal is left: displacement, heading and apex.

  Always four channels whatever the skill's goal actually holds, so every skill agrees on
  its observation width. That matters here specifically: the bridging experiment runs all
  three frozen skills against one shared env, and a skill whose observation is a
  different size cannot be evaluated there at all.
  """
  command = _command(env, command_name)
  return torch.cat(
    [
      command.displacement_error,
      command.heading_error.unsqueeze(-1),
      command.apex_error.unsqueeze(-1),
    ],
    dim=-1,
  )


def reference_contacts(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """(N, 2) which feet the reference has on the ground."""
  return _command(env, command_name).reference.contacts


##
# Rewards: the DeepMimic half
##


def track_joint_pos_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """The pose term: how close every joint is to the reference."""
  command = _command(env, command_name)
  error = torch.sum(
    torch.square(command.reference.joint_pos - command.robot.data.joint_pos), dim=-1
  )
  return torch.exp(-error / std**2)


def track_joint_vel_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """The velocity term. Weighted well below the pose term, as in DeepMimic: joint
  velocity is noisy and matching it exactly is neither achievable nor necessary."""
  command = _command(env, command_name)
  error = torch.sum(
    torch.square(command.reference.joint_vel - command.robot.data.joint_vel), dim=-1
  )
  return torch.exp(-error / std**2)


def track_end_effector_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """The end-effector term: feet and hands where the reference puts them.

  Joint angles alone let a small error at the hip become a large one at the foot, and the
  foot is where it shows. This is the term that makes a gait land in the right place.
  """
  command = _command(env, command_name)
  actual = robot_ee_offsets(command.robot, command.ee_indexes)
  error = torch.sum(torch.square(command.reference.ee_offsets - actual), dim=-1).mean(
    dim=-1
  )
  return torch.exp(-error / std**2)


def track_root_pose_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """The root term: height and orientation, which is what standing upright means here.

  No world position and no heading, on purpose: those are exactly what the clips were
  normalized to remove, and rewarding them would tie the skill back to where its
  reference happened to be recorded.
  """
  command = _command(env, command_name)
  robot = command.robot
  reference = command.reference
  height_error = torch.square(reference.root_height - robot.data.root_link_pos_w[:, 2])
  reference_gravity = quat_apply_inverse(reference.root_quat, robot.data.gravity_vec_w)
  orientation_error = torch.sum(
    torch.square(reference_gravity - robot.data.projected_gravity_b), dim=-1
  )
  return torch.exp(-(height_error + orientation_error) / std**2)


def track_root_velocity_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """The root velocity term, in the root's own yaw frame."""
  command = _command(env, command_name)
  robot = command.robot
  reference = command.reference
  heading = yaw_quat(robot.data.root_link_quat_w)
  lin_vel = quat_apply_inverse(heading, robot.data.root_link_lin_vel_w)
  ang_vel = quat_apply_inverse(heading, robot.data.root_link_ang_vel_w)
  error = torch.sum(
    torch.square(reference.root_lin_vel_b - lin_vel), dim=-1
  ) + torch.sum(torch.square(reference.root_ang_vel_b - ang_vel), dim=-1)
  return torch.exp(-error / std**2)


def track_foot_contact(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Share of feet whose ground contact matches the reference's.

  The contact pattern is what a gait *is*: a walk and a run differ in whether both feet
  are ever down at once far more legibly than they differ in any joint angle. Matching it
  directly is cheap, and it is the term that keeps a jump's flight phase real rather than
  a deep crouch that traces the same root height.
  """
  from mjlab.sensor import ContactSensor

  command = _command(env, command_name)
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None, f"Contact sensor '{sensor_name}' reports no force."
  actual = (torch.norm(force, dim=-1) > force_threshold).float()
  return (actual == command.reference.contacts).float().mean(dim=-1)


##
# Rewards: the goal half
##


def goal_displacement_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """How close the robot is to having travelled what the goal asked for.

  This is what stops the goal from being decoration. Without it a policy can reproduce
  the reference motion perfectly, ignore the goal channel entirely, and score full marks;
  the skill then looks conditioned and steers nowhere.
  """
  command = _command(env, command_name)
  error = torch.sum(torch.square(command.displacement_error), dim=-1)
  return torch.exp(-error / std**2)


def goal_heading_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """How close the robot is to having turned what the goal asked for."""
  command = _command(env, command_name)
  return torch.exp(-torch.square(command.heading_error) / std**2)


def goal_apex_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """How close the highest point reached is to the commanded apex.

  Read off a running maximum, so it climbs as the jump happens and then holds, rather
  than being a single frame's worth of reward the policy has to hit exactly.
  """
  command = _command(env, command_name)
  return torch.exp(-torch.square(command.apex_error) / std**2)


##
# Terminations
##


def motion_finished(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """The clip has run out.

  Registered as a time-out rather than a failure, because reaching the end of the
  reference is success. It has to exist at all: without it the cursor wraps and the
  policy is handed a teleport from the clip's last pose to its first, then penalized for
  not tracking it. The symptom is a skill that trains to a plateau and never gets past
  it, with nothing in the logs pointing at why.
  """
  return _command(env, command_name).finished


def bad_motion_tracking(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  """The robot has lost the reference badly enough that the episode is not worth finishing.

  Measured on the end-effectors, where losing the reference shows first and worst.
  """
  command = _command(env, command_name)
  actual = robot_ee_offsets(command.robot, command.ee_indexes)
  error = torch.norm(command.reference.ee_offsets - actual, dim=-1)
  return torch.any(error > threshold, dim=-1)


def bad_root_height(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  """The root is nowhere near the height the reference says it should be at."""
  command = _command(env, command_name)
  error = (
    command.reference.root_height - command.robot.data.root_link_pos_w[:, 2]
  ).abs()
  return error > threshold


def fell_over(
  env: ManagerBasedRlEnv, limit_angle: float, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Tilted past recovery. Independent of the reference, so it still fires when the
  reference itself is mid-jump and the tracking terms are legitimately loose."""
  asset: Entity = env.scene[asset_cfg.name]
  tilt = torch.acos((-asset.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
  return tilt > limit_angle
