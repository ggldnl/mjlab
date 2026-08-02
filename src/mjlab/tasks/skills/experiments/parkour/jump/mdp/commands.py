"""The goal-conditioned jump command.

This is ASAP's phase-based motion tracking, with two changes.

The first is that it carries a set of clips instead of one. ASAP trains a policy per
motion; the phase scalar in the observation is enough because there is only ever one
thing the phase could refer to. Here five forward jumps of increasing length share a
policy, so the observation also carries the goal: where this jump ends up. The clip
tells the policy how to jump, the goal tells it which jump this is.

The second is stretching. Five clips is five distances, and a policy asked for 1.2 m
would have to interpolate between two points it has only ever seen exactly. So each
episode samples a scale, and the reference is stretched horizontally by it: every
body gets the same time-varying horizontal offset, proportional to how far the root
has travelled since the clip began. The pose is untouched, only the translation
changes, which keeps the reference a physically reachable trajectory. During flight
this is exactly a different takeoff velocity; on the ground it is a longer stride.
The result is continuous goal coverage from a handful of clips.

The command exposes the same property names as mjlab's `MotionCommand`, so the
tracking task's reward and termination terms work against it unchanged.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.tasks.skills.experiments.parkour.jump.motion_lib import MotionLibrary
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from collections.abc import Callable
  from typing import Any

  import viser

  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class JumpCommand(CommandTerm):
  """Multi-clip, goal-conditioned motion reference."""

  cfg: JumpCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = MotionLibrary(cfg.motion_files, self.body_indexes, device=self.device)
    print(self.motion.describe())

    self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.scales = torch.ones(self.num_envs, device=self.device)
    self.motion_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # Where the clip is pinned. Zero is the trained default (clip at the env origin,
    # facing +x); `anchor_to_robot` moves it. `_anchored` keeps the whole transform
    # out of the hot path until something actually uses it, since training never does.
    self.anchor_pos = torch.zeros(self.num_envs, 2, device=self.device)
    self.anchor_yaw = torch.zeros(self.num_envs, device=self.device)
    self._anchored = False

    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    # Adaptive sampling works over a flat (motion, time bin) grid, so a clip that
    # keeps failing gets resampled more often for the same reason a moment within
    # a clip does: that is where the policy is losing episodes.
    self.bins_per_motion = max(int(cfg.adaptive_bins), 1)
    self.num_bins = self.motion.num_motions * self.bins_per_motion
    self.bin_failed_count = torch.zeros(self.num_bins, device=self.device)
    self._current_bin_failed = torch.zeros(self.num_bins, device=self.device)
    kernel = torch.tensor(
      [cfg.adaptive_lambda**i for i in range(cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = kernel / kernel.sum()

    # Set by the viewer GUI or by `request_goal`; consumed at the next resample.
    self._requested_goal: tuple[int, float] | None = None

    for name in (
      "error_anchor_pos",
      "error_anchor_rot",
      "error_body_pos",
      "error_body_rot",
      "error_joint_pos",
      "error_joint_vel",
      "error_goal_pos",
      "goal_distance",
      "goal_reached",
      "sampling_entropy",
    ):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    self._scrubber_handles: tuple = ()

  ##
  # Reference, stretched by the per-episode scale.
  ##

  @property
  def _index(self) -> tuple[torch.Tensor, torch.Tensor]:
    return self.motion_ids, self.time_steps

  @property
  def stretch_offset(self) -> torch.Tensor:
    """Horizontal offset applied to every body this frame, shape [B, 2]."""
    mid, t = self._index
    travelled = self.motion.root_pos_w[mid, t, :2] - self.motion.root_pos_w[mid, 0, :2]
    return (self.scales - 1.0).unsqueeze(-1) * travelled

  @property
  def stretch_vel_offset(self) -> torch.Tensor:
    """Time derivative of `stretch_offset`, shape [B, 2]."""
    mid, t = self._index
    return (self.scales - 1.0).unsqueeze(-1) * self.motion.root_lin_vel_w[mid, t, :2]

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self._index]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self._index]

  ##
  # Where the clip is pinned in the world.
  #
  # Clips are canonicalized to start at the origin facing +x, and by default that is
  # where the reference lives: anchor at zero, and the environment's reset teleports
  # the robot onto it. That is right for training and wrong for composition, where
  # the jump has to happen wherever the robot has walked to. Re-anchoring pins the
  # clip to the robot's current pose instead of moving the robot to the clip.
  #
  # The policy cannot tell the difference. Everything it observes is either in its
  # own body frame or expressed relative to the reference anchor, and a rigid
  # rotation-plus-translation of the whole reference leaves all of that unchanged.
  ##

  def _rotate(self, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a [B, ..., 3] field about z by each env's anchor yaw."""
    if not self._anchored:
      return vec
    quat = self.anchor_yaw_quat
    shape = vec.shape
    flat = vec.reshape(shape[0], -1, 3)
    quat = quat[:, None, :].expand(-1, flat.shape[1], -1)
    return quat_apply(quat.reshape(-1, 4), flat.reshape(-1, 3)).reshape(shape)

  @property
  def anchor_yaw_quat(self) -> torch.Tensor:
    half = 0.5 * self.anchor_yaw
    zero = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zero, zero, torch.sin(half)], dim=-1)

  @property
  def body_pos_w(self) -> torch.Tensor:
    pos = self.motion.body_pos_w[self._index].clone()
    pos[..., :2] += self.stretch_offset[:, None, :]
    pos = self._rotate(pos)
    pos[..., :2] += self.anchor_pos[:, None, :]
    return pos + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    quat = self.motion.body_quat_w[self._index]
    if not self._anchored:
      return quat
    yaw = self.anchor_yaw_quat[:, None, :].expand(-1, quat.shape[1], -1)
    return quat_mul(yaw, quat)

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    vel = self.motion.body_lin_vel_w[self._index].clone()
    vel[..., :2] += self.stretch_vel_offset[:, None, :]
    return self._rotate(vel)

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._rotate(self.motion.body_ang_vel_w[self._index])

  def anchor_to_robot(self, env_ids: torch.Tensor) -> None:
    """Pin the clip to where the robot is now and rewind it to its first frame.

    What a skill's `reset` does for this skill: called when the composition hands
    control to the jump, so the reference starts from the robot's current position
    and heading rather than from the origin. Nothing is written to the simulation --
    the robot is not moved, the reference is.
    """
    self.time_steps[env_ids] = 0
    self.motion_done[env_ids] = False

    root_quat = self.robot_root_quat_w[env_ids]
    # The clip's own first frame carries a heading (its pelvis is not exactly aligned
    # with the direction it travels), so the anchor rotation is the robot's heading
    # relative to the clip's, not the robot's heading outright.
    clip_quat = self.motion.body_quat_w[self.motion_ids[env_ids], 0, 0]
    delta = quat_mul(yaw_quat(root_quat), quat_inv(yaw_quat(clip_quat)))
    self.anchor_yaw[env_ids] = torch.atan2(
      2.0 * (delta[:, 0] * delta[:, 3]), 1.0 - 2.0 * delta[:, 3] ** 2
    )

    self._anchored = True

    # Position last: it is measured against the freshly rotated clip origin, so the
    # robot ends up exactly on the reference's first frame rather than near it.
    clip_root = self.motion.body_pos_w[self.motion_ids[env_ids], 0, 0].clone()
    rotated = quat_apply(self.anchor_yaw_quat[env_ids], clip_root)
    robot_xy = (
      self.robot_root_pos_w[env_ids, :2] - self._env.scene.env_origins[env_ids, :2]
    )
    self.anchor_pos[env_ids] = robot_xy - rotated[:, :2]

    self.update_relative_body_poses()

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self.body_pos_w[:, self.motion_anchor_body_index]

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.body_quat_w[:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.body_lin_vel_w[:, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.body_ang_vel_w[:, self.motion_anchor_body_index]

  ##
  # Robot state, named to match the reference above.
  ##

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
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_root_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, 0]

  @property
  def robot_root_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, 0]

  ##
  # The goal.
  ##

  @property
  def phase(self) -> torch.Tensor:
    """Progress through the clip in [0, 1], shape [B, 1]."""
    lengths = self.motion.time_step_total_per_motion[self.motion_ids]
    return (self.time_steps / (lengths - 1).clamp(min=1)).unsqueeze(-1)

  @property
  def goal(self) -> torch.Tensor:
    """What this jump is: (dx, dy, dyaw, apex), shape [B, 4].

    The translation components carry the episode's stretch, which is the whole
    point: this is the number the policy is asked to hit, and the number a user
    sets at inference time.
    """
    g = self.motion.goals[self.motion_ids].clone()
    g[:, :2] *= self.scales.unsqueeze(-1)
    return g

  @property
  def goal_distance(self) -> torch.Tensor:
    return torch.norm(self.goal[:, :2], dim=-1)

  @property
  def landing_step(self) -> torch.Tensor:
    """Frame the reference touches down on, which is when the goal starts to pay."""
    return self.motion.land_steps[self.motion_ids]

  @property
  def goal_pos_w(self) -> torch.Tensor:
    """Where the reference root ends up, shape [B, 3].

    The clip's last frame, not its touchdown frame, so this is the same quantity
    the goal descriptor reports. Retargeting leaves some clips drifting a little
    after landing, and having the number the policy is shown disagree with the
    number it is paid for would be a slow, quiet source of bias.
    """
    mid = self.motion_ids
    last = self.motion.time_step_total_per_motion[mid] - 1
    pos = self.motion.root_pos_w[mid, last].clone()
    travelled = (
      self.motion.root_pos_w[mid, last, :2] - self.motion.root_pos_w[mid, 0, :2]
    )
    pos[:, :2] += (self.scales - 1.0).unsqueeze(-1) * travelled
    return pos + self._env.scene.env_origins

  @property
  def has_landed(self) -> torch.Tensor:
    return self.time_steps >= self.landing_step

  @property
  def goal_pos_error(self) -> torch.Tensor:
    """Horizontal distance from the robot's root to the goal, [B]."""
    return torch.norm(self.goal_pos_w[:, :2] - self.robot_root_pos_w[:, :2], dim=-1)

  @property
  def goal_b(self) -> torch.Tensor:
    """The goal as the policy sees it: [dx, dy, dyaw, apex, time_to_landing].

    The displacement is what is *left* to cover, in the robot's own heading frame,
    so it stays meaningful under reference-state initialization: an environment
    dropped in mid-flight is told how much further to go, not how far the clip
    travels in total.
    """
    remaining_w = self.goal_pos_w - self.robot_root_pos_w
    heading = yaw_quat(self.robot_root_quat_w)
    remaining_b = quat_apply_inverse(heading, remaining_w)

    goal = self.goal
    time_to_landing = (
      (self.landing_step - self.time_steps).float() * self._env.step_dt
    ).clamp(min=0.0)

    return torch.cat(
      [
        remaining_b[:, :2],
        goal[:, 2:3],
        goal[:, 3:4],
        time_to_landing.unsqueeze(-1),
      ],
      dim=-1,
    )

  @property
  def command(self) -> torch.Tensor:
    """Actor-visible reference: joint targets, phase, and the goal descriptor."""
    return torch.cat([self.joint_pos, self.joint_vel, self.phase, self.goal], dim=1)

  ##
  # Inference-time goal selection.
  ##

  def solve_goal(self, distance: float) -> tuple[int, float]:
    """Pick the clip and stretch that jump `distance` metres.

    Every clip can serve every distance in principle; the one whose own distance is
    closest needs the least stretching, and less stretching means a reference that
    is closer to something a human actually did.
    """
    lo, hi = self.cfg.scale_range
    best: tuple[float, float, int, float] | None = None
    for i, base in enumerate(self.motion.distances.tolist()):
      if base < 1e-6:
        continue
      scale = float(np.clip(distance / base, lo, hi))
      # Reachability first, then how little the clip had to be distorted.
      candidate = (abs(scale * base - distance), abs(scale - 1.0), i, scale)
      if best is None or candidate[:2] < best[:2]:
        best = candidate
    if best is None:
      return 0, 1.0
    return best[2], best[3]

  @property
  def landing_distances(self) -> torch.Tensor:
    """How far each clip has travelled by the frame it touches down on, [num_motions].

    Not the same quantity as `motion.distances`, and the difference is what an
    obstacle cares about. A clip's goal is where its last frame is, and every clip
    here keeps walking for half a metre after it lands: level 5 travels 1.97 m in
    total but touches down at 1.47 m. Something that has to come down past a box
    has to be commanded on the second number.
    """
    starts = self.motion.root_pos_w[:, 0, :2]
    lands = self.motion.root_pos_w[
      torch.arange(self.motion.num_motions, device=self.device), self.land_steps_all, :2
    ]
    return torch.norm(lands - starts, dim=-1)

  @property
  def takeoff_distances(self) -> torch.Tensor:
    """How far each clip has travelled by the frame it leaves the ground."""
    starts = self.motion.root_pos_w[:, 0, :2]
    takeoffs = self.motion.root_pos_w[
      torch.arange(self.motion.num_motions, device=self.device),
      self.takeoff_steps_all,
      :2,
    ]
    return torch.norm(takeoffs - starts, dim=-1)

  @property
  def land_steps_all(self) -> torch.Tensor:
    return self.motion.land_steps

  @property
  def takeoff_steps_all(self) -> torch.Tensor:
    return torch.tensor(
      [m.takeoff_step for m in self.motion.metadata],
      dtype=torch.long,
      device=self.device,
    )

  def solve_landing(
    self, distance: float, takeoff_before: float | None = None
  ) -> tuple[int, float]:
    """Pick the clip and stretch that touch down `distance` metres ahead.

    The counterpart of `solve_goal` for a caller that has an obstacle rather than a
    target: what has to be cleared is decided by where the reference lands, and by
    where it leaves the ground. `takeoff_before` is the distance to whatever must be
    airborne over, and clips whose stretched run-up would still be on the ground at
    that point are rejected outright rather than merely penalized -- a jump that takes
    off on top of the box is not a worse jump, it is a trip.
    """
    lo, hi = self.cfg.scale_range
    landings = self.landing_distances.tolist()
    takeoffs = self.takeoff_distances.tolist()
    best: tuple[float, float, int, float] | None = None
    fallback: tuple[float, float, int, float] | None = None
    for i, (land, takeoff) in enumerate(zip(landings, takeoffs, strict=True)):
      if land < 1e-6:
        continue
      scale = float(np.clip(distance / land, lo, hi))
      candidate = (abs(scale * land - distance), abs(scale - 1.0), i, scale)
      if fallback is None or candidate[:2] < fallback[:2]:
        fallback = candidate
      if takeoff_before is not None and scale * takeoff > takeoff_before:
        continue
      if best is None or candidate[:2] < best[:2]:
        best = candidate
    best = best or fallback
    if best is None:
      return 0, 1.0
    return best[2], best[3]

  def request_goal(self, distance: float) -> None:
    """Ask for a jump distance; applied at the next reset."""
    self._requested_goal = self.solve_goal(distance)

  def set_goal(self, env_ids: torch.Tensor, motion_id: int, scale: float) -> None:
    self.motion_ids[env_ids] = motion_id
    self.scales[env_ids] = scale

  def apply_goals(self, env_ids: torch.Tensor, distances: torch.Tensor) -> None:
    """Give each environment its own commanded distance and restart its clip.

    The scripted counterpart to the viewer's slider: assigns per-environment goals
    and snaps the robots onto the first frame of the clip that serves them, so an
    evaluation can sweep distances without waiting for random resampling to happen
    to produce the ones it wants.
    """
    solved = [self.solve_goal(float(d)) for d in distances]
    self.motion_ids[env_ids] = torch.tensor(
      [m for m, _ in solved], dtype=torch.long, device=self.device
    )
    self.scales[env_ids] = torch.tensor(
      [s for _, s in solved], dtype=torch.float32, device=self.device
    )
    self.time_steps[env_ids] = 0
    self.motion_done[env_ids] = False

    self._write_reference_state_to_sim(
      env_ids,
      self.body_pos_w[env_ids, 0],
      self.body_quat_w[env_ids, 0],
      self.body_lin_vel_w[env_ids, 0],
      self.body_ang_vel_w[env_ids, 0],
      self.joint_pos[env_ids],
      self.joint_vel[env_ids],
    )
    self.update_relative_body_poses()

  ##
  # Sampling.
  ##

  def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
    failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(failed):
      bin_index = self._bin_index()
      self._current_bin_failed[:] = torch.bincount(
        bin_index[env_ids][failed], minlength=self.num_bins
      )

    probs = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / self.num_bins
    probs = torch.nn.functional.pad(
      probs.view(1, 1, -1), (0, self.cfg.adaptive_kernel_size - 1), mode="replicate"
    )
    probs = torch.nn.functional.conv1d(probs, self.kernel.view(1, 1, -1)).view(-1)
    probs = probs / probs.sum()

    sampled = torch.multinomial(probs, len(env_ids), replacement=True)
    motion_ids = torch.div(sampled, self.bins_per_motion, rounding_mode="floor")
    bins = sampled % self.bins_per_motion

    self.motion_ids[env_ids] = motion_ids
    lengths = self.motion.time_step_total_per_motion[motion_ids]
    jitter = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
    self.time_steps[env_ids] = (
      (bins + jitter) / self.bins_per_motion * (lengths - 1)
    ).long()

    entropy = -(probs * (probs + 1e-12).log()).sum() / math.log(self.num_bins)
    self.metrics["sampling_entropy"][:] = entropy

  def _bin_index(self) -> torch.Tensor:
    lengths = self.motion.time_step_total_per_motion[self.motion_ids].clamp(min=1)
    within = torch.clamp(
      (self.time_steps * self.bins_per_motion) // lengths, 0, self.bins_per_motion - 1
    )
    return self.motion_ids * self.bins_per_motion + within

  def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
    self.motion_ids[env_ids] = torch.randint(
      0, self.motion.num_motions, (len(env_ids),), device=self.device
    )
    lengths = self.motion.time_step_total_per_motion[self.motion_ids[env_ids]]
    frac = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
    self.time_steps[env_ids] = (frac * (lengths - 1)).long()
    self.metrics["sampling_entropy"][:] = 1.0

  def _start_sampling(self, env_ids: torch.Tensor) -> None:
    self.motion_ids[env_ids] = torch.randint(
      0, self.motion.num_motions, (len(env_ids),), device=self.device
    )
    self.time_steps[env_ids] = 0
    self.metrics["sampling_entropy"][:] = 1.0

  def _sample_scales(self, env_ids: torch.Tensor) -> None:
    lo, hi = self.cfg.scale_range
    self.scales[env_ids] = sample_uniform(lo, hi, (len(env_ids),), device=self.device)

  ##
  # Reset.
  ##

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.sampling_mode == "start":
      self._start_sampling(env_ids)
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      self._adaptive_sampling(env_ids)

    self._sample_scales(env_ids)

    if self._requested_goal is not None:
      motion_id, scale = self._requested_goal
      self.set_goal(env_ids, motion_id, scale)
      self.time_steps[env_ids] = 0

    self.motion_done[env_ids] = False
    # A resample puts the clip back at the origin: the reference is about to be
    # written into the simulation, so the robot goes to the clip, not the reverse.
    self.anchor_pos[env_ids] = 0.0
    self.anchor_yaw[env_ids] = 0.0

    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

    range_list = [
      self.cfg.pose_range.get(k, (0.0, 0.0))
      for k in ("x", "y", "z", "roll", "pitch", "yaw")
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), self.device)
    root_pos += rand[:, 0:3]
    root_ori = quat_mul(
      quat_from_euler_xyz(rand[:, 3], rand[:, 4], rand[:, 5]), root_ori
    )

    range_list = [
      self.cfg.velocity_range.get(k, (0.0, 0.0))
      for k in ("x", "y", "z", "roll", "pitch", "yaw")
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), self.device)
    root_lin_vel += rand[:, :3]
    root_ang_vel += rand[:, 3:]

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]
    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=self.device,
    )

    self._write_reference_state_to_sim(
      env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel
    )

  ##
  # Per-step update.
  ##

  def update_relative_body_poses(self) -> None:
    """Express the reference bodies relative to where the robot actually is.

    Horizontal position and heading are taken from the robot, height and everything
    else from the reference, so the body-tracking terms measure posture rather than
    accumulated drift. Global position is a separate term, on the anchor.
    """
    num_bodies = len(self.cfg.body_names)
    anchor_pos = self.anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
    anchor_quat = self.anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
    robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
    robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)

    delta_pos_w = robot_anchor_pos.clone()
    delta_pos_w[..., 2] = anchor_pos[..., 2]
    delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos
    )

  def _update_command(self) -> None:
    self.time_steps += 1
    lengths = self.motion.time_step_total_per_motion[self.motion_ids]
    self.motion_done = self.time_steps >= lengths
    self.time_steps = torch.minimum(self.time_steps, lengths - 1)

    self.update_relative_body_poses()

    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def _update_metrics(self) -> None:
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)
    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

    # Only meaningful once the jump is over, so hold the last value until then
    # rather than averaging in the distance to a landing that has not happened.
    landed = self.has_landed
    error = self.goal_pos_error
    self.metrics["error_goal_pos"] = torch.where(
      landed, error, self.metrics["error_goal_pos"]
    )
    self.metrics["goal_reached"] = torch.where(
      landed,
      (error < self.cfg.goal_success_threshold).float(),
      self.metrics["goal_reached"],
    )
    self.metrics["goal_distance"] = self.goal_distance

  ##
  # Viewer.
  ##

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        for gi in range(self._ghost_model.ngeom):
          if (
            self._ghost_model.geom_contype[gi] != 0
            or self._ghost_model.geom_conaffinity[gi] != 0
          ):
            self._ghost_model.geom_rgba[gi, 3] = 0
          else:
            self._ghost_model.geom_rgba[gi] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      body_pos_w = self.body_pos_w
      body_quat_w = self.body_quat_w
      joint_pos = self.joint_pos
      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = joint_pos[batch].cpu().numpy()
        visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")
    else:
      body_pos_w = self.body_pos_w
      body_quat_w = self.body_quat_w
      for batch in env_indices:
        desired_pos = body_pos_w[batch].cpu().numpy()
        desired_rotm = matrix_from_quat(body_quat_w[batch]).cpu().numpy()
        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_pos[i],
            rotation_matrix=desired_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """A jump-distance dial, which is the whole user-facing interface."""
    distances = self.motion.distances.tolist()
    lo, hi = self.cfg.scale_range

    with server.gui.add_folder(name.capitalize()):
      slider = server.gui.add_slider(
        "Jump distance (m)",
        min=round(min(distances) * lo, 2),
        max=round(max(distances) * hi, 2),
        step=0.05,
        initial_value=round(float(np.median(distances)), 2),
      )
      readout = server.gui.add_text("Clip / stretch", initial_value="", disabled=True)
      all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
      go_btn = server.gui.add_button("Jump this far")

      def _refresh() -> None:
        motion_id, scale = self.solve_goal(float(slider.value))
        readout.value = f"{self.motion.metadata[motion_id].name}  x{scale:.2f}"

      @slider.on_update
      def _(_) -> None:
        _refresh()

      @go_btn.on_click
      def _(_) -> None:
        self.request_goal(float(slider.value))
        if request_action is not None:
          request_action("CUSTOM", {"type": "gui_reset", "all_envs": all_envs_cb.value})

      _refresh()

    self._scrubber_handles = (slider, readout, all_envs_cb, go_btn)

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    if not self._scrubber_handles:
      return False
    self.request_goal(float(self._scrubber_handles[0].value))
    self._resample_command(env_ids)
    self._requested_goal = None
    self.update_relative_body_poses()
    return True


@dataclass(kw_only=True)
class JumpCommandCfg(CommandTermCfg):
  motion_files: tuple[str, ...]
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str

  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.1, 0.1)

  scale_range: tuple[float, float] = (0.85, 1.15)
  """How much a clip may be stretched horizontally. Wider is more goal coverage and
  a less human-like reference; past roughly ±20% the takeoff stops being something
  the recorded pose can plausibly produce."""

  goal_success_threshold: float = 0.25
  """Landing within this many metres of the target counts as reaching the goal."""

  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  adaptive_bins: int = 8
  adaptive_kernel_size: int = 3
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> JumpCommand:
    return JumpCommand(self, env)
