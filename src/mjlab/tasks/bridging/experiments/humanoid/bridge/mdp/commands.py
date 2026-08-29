"""The window: where the bridge starts, where it has to be, and how long it has.

One episode is one window. A start state from one skill's rollout, a target state from
another's, a deadline in control steps, nothing in between. The robot is teleported onto
the start and the clock runs. The middle is never scored.

##
# What the policy reads
##

    d_pos    3   where the target is, from here, in the heading frame
    d_rot    6   which way it faces, relative to this heading
    t_lin    3   how fast it is going, in the heading frame
    t_ang    3   and turning
    t_q      J   the joint angles it holds, relative to the default pose
    t_qd     J   and their rates
    clock    2   seconds left, and the fraction of the window left

Position and orientation are differences, since a difference is what remains to be closed.
Velocities and joint angles are absolute, since proprioception already carries the robot's
own in the same units and the policy can subtract them itself.

The clock is not optional. Without time-to-go the task is not Markov: the same state a
quarter of the way through a window and a step from its end call for opposite actions.

##
# Why a pair is built rather than drawn
##

Both endpoints come out of the dataset individually valid, and that is not enough. Two
reachable states can be an unreachable pair, and an unsolvable episode is worse than none:
it contributes a gradient pointing nowhere, and enough of them teach the policy to hedge.

Example of an impossible pair, and the acceleration it implies:

    3 m/s -> standing still in 0.2 s   =   15 m/s^2

Three things make a pair feasible:

  Placement is feasible by construction. Only the target's content comes from the dataset:
  height, tilt, joint angles, every velocity. Where it sits and which way it faces are
  chosen here, inside a set the robot can reach. The centre of that set is where a body
  carrying its current momentum would arrive; the offset around it is bounded by what a
  walk covers in the time available.

  The velocity change is checked, against `max_accel`. This is the one thing drawn from the
  dataset that placement cannot fix.

  The joint travel is checked, against `joint_speed`. A sustained rate, not the actuator
  limit, which is far above anything a loaded leg reaches and would never bind.

A candidate failing either check is redrawn a few times. If nothing admissible turns up,
the deadline is stretched to what the best candidate needs rather than the pair being
thrown away.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.dataset import (
  DEFAULT_DATASET,
  ROOT_STATE_DIM,
  Dataset,
  load_dataset,
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

CHANNELS = (
  "root_pos",
  "root_ori",
  "root_lin_vel",
  "root_ang_vel",
  "joint_pos",
  "joint_vel",
)
"""The six ways an arrival can be wrong. Kept apart because the units differ: a tenth of a
radian and a tenth of a metre per second are not the same mistake."""

TARGET_COLOR = (1.0, 0.72, 0.2, 0.45)
"""The target ghost."""


##
# Measuring an arrival.
##


@dataclass(kw_only=True)
class Tolerances:
  """How wrong each channel may be and still count as arrived.

  These are the whole metric. A tolerance wider than the gap that channel has to close
  makes the channel free: it scores near one from the start, contributes no gradient, and
  carries the total high enough that a robot standing still looks like it is doing the
  task. That happened here once, a statue at 0.459 against a trained policy's 0.454.

  Do not take these numbers on faith. Run this whenever the dataset changes, and paste back
  what it prints:

      uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.evaluate \
        --calibrate True

  It measures the median gap per channel and prints half of it, which is where each of
  these belongs.
  """

  root_pos: float = 0.29
  """Metres."""
  root_ori: float = 0.27
  """Radians."""
  root_lin_vel: float = 0.38
  """Metres per second. The channel that matters most for a hand-over: a body in the right
  pose carrying the wrong momentum is about to be somewhere else."""
  root_ang_vel: float = 0.29
  """Radians per second."""
  joint_pos: float = 0.10
  """Radians, as a root mean square over the joints."""
  joint_vel: float = 0.87
  """Radians per second, likewise."""

  def as_tensor(self, device: str | torch.device) -> torch.Tensor:
    return torch.tensor(
      [getattr(self, name) for name in CHANNELS], device=device, dtype=torch.float32
    )


def channel_errors(
  actual: torch.Tensor, target: torch.Tensor, num_joints: int
) -> torch.Tensor:
  """How wrong one state is against another, per channel. (N, 6), in natural units.

  Both arguments are dataset rows, (N, 13 + 2J). A free function and not a method, because
  the reward, the metrics and an evaluation with no live environment all have to get the
  same number out of it. Two implementations of "did it arrive" would drift apart.
  """
  q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + num_joints)
  qd = slice(ROOT_STATE_DIM + num_joints, ROOT_STATE_DIM + 2 * num_joints)
  return torch.stack(
    [
      (actual[:, 0:3] - target[:, 0:3]).norm(dim=-1),
      quat_error_magnitude(actual[:, 3:7], target[:, 3:7]),
      (actual[:, 7:10] - target[:, 7:10]).norm(dim=-1),
      (actual[:, 10:ROOT_STATE_DIM] - target[:, 10:ROOT_STATE_DIM]).norm(dim=-1),
      (actual[:, q] - target[:, q]).square().mean(dim=-1).sqrt(),
      (actual[:, qd] - target[:, qd]).square().mean(dim=-1).sqrt(),
    ],
    dim=-1,
  )


def arrival_score(errors: torch.Tensor, tolerances: torch.Tensor) -> torch.Tensor:
  """One kernel per channel, then the mean. (N, 6) -> (N,), in (0, 1].

  The mean of exponentials, not the exponential of a mean. The two agree when every channel
  is equally wrong. When they are not, a product is dominated by the worst channel and
  kills every other channel's gradient with it, while this is dominated by the best. The
  tracking tasks here already use the mean. What stops a policy farming the easy channels
  is tight tolerances and the strayed termination, not the shape of the kernel.
  """
  return torch.exp(-(errors / tolerances).square()).mean(dim=-1)


def arrived(errors: torch.Tensor, tolerances: torch.Tensor) -> torch.Tensor:
  """Whether every channel is inside tolerance. (N,) bool.

  The success metric, deliberately not the reward. A reward has to be smooth to be
  learnable; this has to be honest to be reportable. Inside tolerance on five channels out
  of six is not an arrival.
  """
  return (errors <= tolerances).all(dim=-1)


def transit_time(
  start: torch.Tensor,
  target: torch.Tensor,
  num_joints: int,
  max_accel: float,
  joint_speed: float,
) -> torch.Tensor:
  """Seconds a body needs to get from one state to another, on the two things placement
  cannot fix. Broadcasts over any leading shape.

  Ground distance is free: a target is put wherever the start's velocity would carry a
  body, so it is never what makes a pair hard. What cannot be arranged away is the velocity
  the target moves at and the pose it holds. Each takes time at a rate a humanoid sustains,
  and they happen at once, so the estimate is the larger of the two:

      max(speed_change / max_accel, joint_travel / joint_speed)

  A free function, next to `channel_errors`, for the same reason: this number decides which
  pairs the bridge trains on, and a second way of measuring a pair would be asking for
  something other than what the policy learned.
  """
  q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + num_joints)
  speed_up = (target[..., 7:10] - start[..., 7:10]).norm(dim=-1)
  reach = (target[..., q] - start[..., q]).abs().amax(dim=-1)
  return torch.maximum(speed_up / max_accel, reach / joint_speed)


##
# The command.
##


class BridgeCommand(CommandTerm):
  """Draws a window per environment, teleports onto its start, holds its target."""

  cfg: BridgeCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: BridgeCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.num_joints = self.robot.data.joint_pos.shape[1]

    self.dataset: Dataset = load_dataset(cfg.dataset_path, str(self.device), cfg.split)
    if self.dataset.num_joints != self.num_joints:
      raise ValueError(
        f"The dataset holds {self.dataset.num_joints}-joint states and this robot has "
        f"{self.num_joints}. Rebuild the dataset against this robot."
      )
    self.fps = self.dataset.fps
    self.leaving = self.dataset.of(cfg.leaving)
    self.entering = self.dataset.of(cfg.entering)

    self.tolerances = cfg.tolerances.as_tensor(self.device)
    """The calibrated tolerances. Fixed for the whole run. This is what `arrived` means and
    what every reported number is measured against, so it must not move."""

    self.reward_tolerances = self.tolerances * cfg.tolerance_ceiling
    """What the reward actually uses. Starts wide and ratchets down toward the fixed ones.
    See `_tighten` for why, and why the two have to be different objects."""

    self._running_error = self.reward_tolerances / max(cfg.tolerance_slack, 1e-6)
    """Slow average of what each channel misses by at the deadline. Initialized so the
    first tightening asks for exactly the ceiling and measurements pull it down from
    there."""

    self._tightened_at = -1
    """Environment step the curriculum last moved on. `errors_now` is called from a reward
    term, and a second term reading it would advance the average twice in one step."""
    self.state_dim = self.dataset.states.shape[1]

    # The window. target is a dataset row placed in this environment's world, deadline is
    # how many control steps the policy has to reach it
    self.target = torch.zeros(self.num_envs, self.state_dim, device=self.device)
    self.deadline = torch.full(
      (self.num_envs,), cfg.deadline_range[1], dtype=torch.long, device=self.device
    )
    self.start_distance = torch.zeros(self.num_envs, device=self.device)
    self.stretched = torch.zeros(self.num_envs, device=self.device)

    # Latched at the deadline and held until the next window is drawn. The metrics read
    # these rather than the live state, because by then the env has already been reset and
    # the live state is a fresh robot standing at its default pose
    self.reached = torch.zeros(self.num_envs, device=self.device)
    self.score = torch.zeros(self.num_envs, device=self.device)
    self.arrived = torch.zeros(self.num_envs, device=self.device)
    self.final = torch.zeros(self.num_envs, len(CHANNELS), device=self.device)

    self.metrics["arrived"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["score"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["reached_deadline"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["deadline_s"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["spread"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["stretched"] = torch.zeros(self.num_envs, device=self.device)
    for name in CHANNELS:
      self.metrics[f"err_{name}"] = torch.zeros(self.num_envs, device=self.device)
    for name in CHANNELS:
      self.metrics[f"tol_{name}"] = torch.zeros(self.num_envs, device=self.device)

    self._ghost: mujoco.MjModel | None = None

  ##
  # Where the episode is.
  ##

  @property
  def step(self) -> torch.Tensor:
    """How many control steps this episode has taken. (num_envs,).

    The environment's own counter, not one kept here. It is zeroed on reset and incremented
    at the top of every step, before terminations and rewards run, so during those it names
    the step that just happened, which is what a deadline wants. A counter maintained by
    this term would have to advance in `_update_command`, which runs after the auto-reset,
    and would be a step out for every environment that just finished.
    """
    return self._env.episode_length_buf

  @property
  def remaining(self) -> torch.Tensor:
    """Control steps left before the deadline, floored at zero."""
    return (self.deadline - self.step).clamp(min=0)

  @property
  def progress(self) -> torch.Tensor:
    """How far through the window, in [0, 1]."""
    return (self.step.float() / self.deadline.float()).clamp(max=1.0)

  def spread(self) -> float:
    """How far off the natural spot a target may be put, as a share of the budget.

    Ramps from zero over `curriculum_steps`, then holds. At zero every target sits exactly
    where the robot's current momentum would take it, which is what a skill running on
    undisturbed would produce and the easiest thing to ask for. At one it is anywhere in
    the disc a walk could reach in the time available.

    A curriculum on distance rather than on the deadline. They are the same dial from two
    ends and one is enough.
    """
    if self.cfg.curriculum_steps <= 0:
      return self.cfg.spread
    alpha = min(self._env.common_step_counter / self.cfg.curriculum_steps, 1.0)
    return self.cfg.spread * alpha

  ##
  # What the policy reads.
  ##

  @property
  def command(self) -> torch.Tensor:
    """(num_envs, 17 + 2J). Layout is in this module's header."""
    data = self.robot.data
    yaw = yaw_quat(data.root_link_quat_w)
    q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + self.num_joints)
    qd = slice(ROOT_STATE_DIM + self.num_joints, ROOT_STATE_DIM + 2 * self.num_joints)

    return torch.cat(
      [
        quat_apply_inverse(yaw, self.target[:, 0:3] - data.root_link_pos_w),
        _rot6d(quat_mul(quat_conjugate(yaw), self.target[:, 3:7])),
        quat_apply_inverse(yaw, self.target[:, 7:10]),
        quat_apply_inverse(yaw, self.target[:, 10:ROOT_STATE_DIM]),
        self.target[:, q] - data.default_joint_pos,
        self.target[:, qd],
        (self.remaining.float() / self.fps).unsqueeze(-1),
        (self.remaining.float() / self.deadline.float()).unsqueeze(-1),
      ],
      dim=-1,
    )

  ##
  # What the reward reads.
  ##

  def state_now(self) -> torch.Tensor:
    """The robot as a dataset row. (num_envs, 13 + 2J)."""
    data = self.robot.data
    return torch.cat(
      [
        data.root_link_pos_w,
        data.root_link_quat_w,
        data.root_link_lin_vel_w,
        data.root_link_ang_vel_w,
        data.joint_pos,
        data.joint_vel,
      ],
      dim=-1,
    )

  def errors_now(self) -> torch.Tensor:
    """The six channel errors against the target, this step, latching at the deadline.

    Called from the reward, which is the only place an arrival can be read at all. The
    reward manager runs before the auto-reset; the metrics run after it, by which point an
    environment that just finished holds a fresh robot at its default pose, and every
    number taken from it describes a different episode. This project once shipped a success
    metric that read 100% for exactly that reason.

    So the deadline snapshot is taken here and written straight into `metrics`, where
    `CommandTerm.reset` picks it up. `_update_metrics` never touches those entries.
    """
    errors = channel_errors(self.state_now(), self.target, self.num_joints)

    at_deadline = self.step == self.deadline
    if bool(at_deadline.any()):
      score = arrival_score(errors, self.tolerances)
      hit = arrived(errors, self.tolerances).float()
      self.final = torch.where(at_deadline.unsqueeze(-1), errors, self.final)
      self.score = torch.where(at_deadline, score, self.score)
      self.arrived = torch.where(at_deadline, hit, self.arrived)
      self.reached = torch.where(
        at_deadline, torch.ones_like(self.reached), self.reached
      )
      self._tighten(errors[at_deadline])

      self.metrics["score"] = self.score.clone()
      self.metrics["arrived"] = self.arrived.clone()
      self.metrics["reached_deadline"] = self.reached.clone()
      for index, name in enumerate(CHANNELS):
        self.metrics[f"err_{name}"] = self.final[:, index].clone()

    return errors

  def _tighten(self, arrived_errors: torch.Tensor) -> None:
    """Move each channel's reward tolerance toward what the policy is currently missing by.

    An exponential kernel only teaches over a narrow band. Past about three tolerances the
    kernel and its gradient are numerically zero, the channel drops out of the objective,
    and since it then costs nothing the policy spends it on the channels that still pay.

    Measured on an 8400-iteration run:

        joint_pos error     1.075          tolerance 0.10   ->   exp(-115)
        statue on the same channel  0.159

    The bridge was not neglecting the joints, it was flailing them to drive the root,
    because the root was the only thing it could feel.

    So the tolerance is kept near the error instead of near the goal. `tolerance_slack`
    times the running error puts the policy about 1.4 tolerances out, the steepest part of
    the kernel, and keeps it there as the error falls. The floor is the calibrated
    tolerance, so this makes the task reachable and never easier than what is asked for.

    Ratchets down only. A tolerance that could widen again would meet a policy that got
    worse with a wider kernel and the same score, which is the original disease in a new
    hat. Descending only means a regression costs reward.

    The average covers errors latched at a deadline and nothing else: an episode that ended
    on the floor has no arrival to be wrong about.

    Expect one side effect. The arrival reward falls as the tolerances tighten, even while
    the policy improves, so the reward curve is not a progress bar. Read
    Metrics/bridge/err_*, which are raw and carry no tolerance at all.
    """
    if arrived_errors.numel() == 0:
      return
    if self._env.common_step_counter == self._tightened_at:
      return
    self._tightened_at = self._env.common_step_counter

    rate = self.cfg.tolerance_rate
    self._running_error = (
      1.0 - rate
    ) * self._running_error + rate * arrived_errors.mean(dim=0)
    wanted = torch.clamp(
      self.cfg.tolerance_slack * self._running_error,
      min=self.tolerances,
      max=self.tolerances * self.cfg.tolerance_ceiling,
    )
    self.reward_tolerances = torch.minimum(self.reward_tolerances, wanted)

  ##
  # Drawing a window.
  ##

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    count = env_ids.numel()
    low, high = self.cfg.deadline_range

    start = self.dataset.states[
      self.leaving[torch.randint(0, self.leaving.numel(), (count,), device=self.device)]
    ].clone()
    deadline = torch.randint(low, high + 1, (count,), device=self.device)

    # A random heading for the start. Everything the policy reads is in its own heading
    # frame and every target is placed relative to it, so this changes no question that is
    # ever asked. It is here so nothing can come to depend on the world frame by accident
    facing = quat_from_angle_axis(
      torch.rand(count, device=self.device) * (2.0 * math.pi), _up(count, self.device)
    )
    start = reyaw(start, quat_mul(facing, quat_conjugate(yaw_quat(start[:, 3:7]))))

    target, deadline, stretched = self._draw_target(start, deadline)
    self.place(env_ids, start, target, deadline, stretched)

  def _draw_target(
    self, start: torch.Tensor, deadline: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A target the robot can reach from `start` in `deadline` steps.

    Candidates are drawn all at once and the first admissible one per environment is kept,
    rather than looping until something fits. Where nothing fits, the least demanding
    candidate is kept and the deadline is stretched to what it needs.
    """
    count = start.shape[0]
    attempts = self.cfg.attempts

    rows = self.entering[
      torch.randint(0, self.entering.numel(), (count, attempts), device=self.device)
    ]
    candidates = self.dataset.states[rows]

    # Turned to face the way this window asks for, before anything is measured. The turn
    # moves the velocity vector, and it is the turned velocity the robot has to arrive with
    heading = yaw_quat(start[:, 3:7]).unsqueeze(1).expand(count, attempts, 4)
    budget = self.cfg.turn_speed * deadline.float() / self.fps * self.spread()
    turn = (
      2.0 * torch.rand(count, attempts, device=self.device) - 1.0
    ) * budget.unsqueeze(1)
    facing = quat_mul(
      heading,
      quat_from_angle_axis(
        turn.reshape(-1), _up(count * attempts, self.device)
      ).reshape(count, attempts, 4),
    )
    candidates = reyaw(
      candidates.reshape(-1, self.state_dim),
      quat_mul(
        facing.reshape(-1, 4),
        quat_conjugate(yaw_quat(candidates.reshape(-1, self.state_dim)[:, 3:7])),
      ),
    ).reshape(count, attempts, self.state_dim)

    # What each candidate costs, in seconds, on the two things placement cannot fix
    horizon = (deadline.float() / self.fps).unsqueeze(1)
    need = transit_time(
      start[:, None],
      candidates,
      self.num_joints,
      self.cfg.max_accel,
      self.cfg.joint_speed,
    )

    fits = need <= horizon
    # First admissible candidate, or the cheapest when there is none. argmax on a bool picks
    # the first True, and on an all-False row picks index zero, so the fallback is selected
    # explicitly rather than relied on
    pick = torch.where(fits.any(dim=1), fits.float().argmax(dim=1), need.argmin(dim=1))
    index = pick.view(-1, 1, 1).expand(-1, 1, self.state_dim)
    target = candidates.gather(1, index).squeeze(1)
    chosen = need.gather(1, pick.view(-1, 1)).squeeze(1)

    stretched = chosen > horizon.squeeze(1)
    deadline = torch.where(
      stretched, (chosen * self.fps).ceil().long().clamp(min=1), deadline
    )

    # Where it goes. The centre of the disc is where a body would be if it changed
    # velocity steadily from the one it has to the one asked for, which is the placement a
    # window produces when nothing has gone wrong. The offset around it is the curriculum:
    # at spread zero every target sits on that spot
    horizon = deadline.float() / self.fps
    bearing = torch.rand(target.shape[0], device=self.device) * (2.0 * math.pi)
    radius = (
      torch.rand(target.shape[0], device=self.device)
      * self.cfg.travel_speed
      * horizon
      * self.spread()
    )
    drift = torch.stack([radius * bearing.cos(), radius * bearing.sin()], dim=-1)
    target[:, 0:2] = (
      start[:, 0:2]
      + 0.5 * (start[:, 7:9] + target[:, 7:9]) * horizon.unsqueeze(-1)
      + quat_apply(
        yaw_quat(start[:, 3:7]),
        torch.cat([drift, torch.zeros_like(radius).unsqueeze(-1)], dim=-1),
      )[:, 0:2]
    )
    return target, deadline, stretched.float()

  def place(
    self,
    env_ids: torch.Tensor,
    start: torch.Tensor,
    target: torch.Tensor,
    deadline: torch.Tensor,
    stretched: torch.Tensor | None = None,
  ) -> None:
    """Open a window on these environments and put the robot on its start.

    `start` and `target` are (N, 13 + 2J) in one shared frame. Both slide horizontally so
    the start lands on the environment's origin, which keeps them in one coordinate system
    without the caller knowing where that is. Heights, headings and velocities are
    untouched by the slide.

    Public because a window does not have to come from the dataset. Nothing else calls it
    today.
    """
    if env_ids.numel() == 0:
      return

    origin = self._env.scene.env_origins[env_ids]
    shift = origin[:, :2] - start[:, :2]
    target = target.clone()
    target[:, 0:2] += shift

    root_pos = start[:, 0:3].clone()
    root_pos[:, 0:2] = origin[:, :2]

    self.target[env_ids] = target
    self.deadline[env_ids] = deadline
    self.start_distance[env_ids] = (root_pos - target[:, 0:3]).norm(dim=-1)
    self.stretched[env_ids] = (
      torch.zeros_like(deadline, dtype=torch.float32)
      if stretched is None
      else stretched
    )
    self.reached[env_ids] = 0.0
    self.score[env_ids] = 0.0
    self.arrived[env_ids] = 0.0
    self.final[env_ids] = 0.0

    joint_pos = start[:, ROOT_STATE_DIM : ROOT_STATE_DIM + self.num_joints]
    joint_vel = start[:, ROOT_STATE_DIM + self.num_joints :]
    if self.cfg.start_noise > 0.0:
      scale = self.cfg.start_noise
      root_pos = root_pos + torch.randn_like(root_pos) * scale * 0.02
      joint_pos = joint_pos + torch.randn_like(joint_pos) * scale * 0.03
      joint_vel = joint_vel + torch.randn_like(joint_vel) * scale * 0.3

    limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(limits[:, :, 0], limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    self.robot.write_root_state_to_sim(
      torch.cat(
        [root_pos, start[:, 3:7], start[:, 7:10], start[:, 10:ROOT_STATE_DIM]], dim=-1
      ),
      env_ids=env_ids,
    )
    # Not optional. qpos and qvel are not the whole state: the action term holds the last
    # action it applied and the observation terms hold their history. A robot teleported
    # without clearing them starts its window carrying a step of somebody else's episode
    self.robot.reset(env_ids=env_ids)

  def _update_command(self) -> None:
    pass

  def _update_metrics(self) -> None:
    """Only the live numbers.

    `errors_now` writes the latched ones, before the reset that destroys the state they are
    read from. Writing them again here would overwrite them with whatever the freshly reset
    robot happens to look like.
    """
    self.metrics["deadline_s"] = self.deadline.float() / self.fps
    self.metrics["spread"] = torch.full_like(self.metrics["spread"], self.spread())
    self.metrics["stretched"] = self.stretched.clone()
    for index, name in enumerate(CHANNELS):
      self.metrics[f"tol_{name}"] = torch.full_like(
        self.metrics[f"tol_{name}"], float(self.reward_tolerances[index])
      )

  ##
  # Drawing it.
  ##

  def _debug_vis_impl(self, visualizer) -> None:
    """A translucent robot standing in the target.

    A pose and nothing else, because a pose is all that can be drawn: half of a target is
    velocity and a still body says nothing about that. It does show where the window is
    sending the robot, and after the deadline it stays where it was put, so the gap to the
    real robot is the arrival error left standing to be looked at.
    """
    if self._ghost is None:
      self._ghost = self._tinted(TARGET_COLOR)

    indexing = self.robot.indexing
    free = indexing.free_joint_q_adr.cpu().numpy()
    joints = indexing.joint_q_adr.cpu().numpy()
    for batch in visualizer.get_env_indices(self.num_envs):
      # From qpos0 rather than zeros: a zero quaternion is not a rotation, and anything
      # else in the scene keeps its own default
      qpos = np.array(self._env.sim.mj_model.qpos0, dtype=np.float64)
      row = self.target[batch].detach().cpu().numpy()
      qpos[free[0:3]] = row[0:3]
      qpos[free[3:7]] = row[3:7]
      qpos[joints] = row[ROOT_STATE_DIM : ROOT_STATE_DIM + self.num_joints]
      visualizer.add_ghost_mesh(
        qpos, model=self._ghost, alpha=TARGET_COLOR[3], label=f"target_{batch}"
      )

  def _tinted(self, color: tuple[float, float, float, float]) -> mujoco.MjModel:
    """This scene's model with the robot painted `color` and everything else hidden."""
    ghost = copy.deepcopy(self._env.sim.mj_model)
    mine = set(self.robot.indexing.geom_ids.tolist())
    for geom in range(ghost.ngeom):
      solid = ghost.geom_contype[geom] or ghost.geom_conaffinity[geom]
      if geom in mine and not solid:
        ghost.geom_rgba[geom] = color
      else:
        # Collision geoms are the crude convex stand-ins the solver uses and draw a robot
        # made of boxes. Everything else in the scene would be a second copy hanging in
        # the air beside the target
        ghost.geom_rgba[geom, 3] = 0.0
    return ghost


##
# Helpers.
##


def _up(count: int, device: torch.device | str) -> torch.Tensor:
  axis = torch.zeros(count, 3, device=device)
  axis[:, 2] = 1.0
  return axis


def reyaw(states: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
  """The same state facing a different way. (N, 13 + 2J). Position is the caller's.

  Every skill here is egocentric: it reads its own body frame, and a tracker reads poses
  relative to an anchor it carries. So a state a skill produced facing one way is a state it
  can be in facing another. The rotation has to reach the orientation and both velocity
  vectors, or the pose and the momentum disagree about which way the body is going.
  """
  out = states.clone()
  out[:, 3:7] = quat_mul(rotation, states[:, 3:7])
  out[:, 7:10] = quat_apply(rotation, states[:, 7:10])
  out[:, 10:ROOT_STATE_DIM] = quat_apply(rotation, states[:, 10:ROOT_STATE_DIM])
  return out


def _rot6d(quat: torch.Tensor) -> torch.Tensor:
  """The first two columns of a rotation matrix, flattened. (..., 4) -> (..., 6).

  Six numbers rather than a quaternion's four: a quaternion has two representations for
  every rotation, and this form is unique and continuous.
  """
  matrix = matrix_from_quat(quat)
  return matrix[..., :, :2].transpose(-1, -2).reshape(*quat.shape[:-1], 6)


@dataclass(kw_only=True)
class BridgeCommandCfg(CommandTermCfg):
  entity_name: str = "robot"

  dataset_path: Path = DEFAULT_DATASET
  split: str = "train"
  leaving: tuple[str, ...] | None = None
  """Which skills a start state may come from. None means any in the dataset."""
  entering: tuple[str, ...] | None = None
  """Which skills a target state may come from. None means any in the dataset."""

  deadline_range: tuple[int, int] = (15, 60)
  """Control steps the bridge gets, before any stretching. At 50 Hz that is 0.3 to 1.2 s:
  long enough to take a step or two, short enough that momentum still decides the answer."""

  tolerances: Tolerances = field(default_factory=Tolerances)
  """What arriving means. Fixed, and the floor the curriculum below descends to."""

  ##
  # The tolerance curriculum. See `BridgeCommand._tighten`.
  ##

  tolerance_ceiling: float = 10.0
  """How many times the calibrated tolerance a channel may start at.

  Ten is what the worst channel needs. joint_pos is calibrated at 0.10 and an untrained
  bridge misses by about 1.0, so anything tighter starts it on the flat part of its own
  kernel, which is the situation this exists to prevent."""

  tolerance_slack: float = 0.7
  """Where the tolerance sits relative to the error being made, as a fraction.

  At 0.7 the policy is about 1.4 tolerances out, near the steepest part of the kernel. Above
  1.0 the channel saturates and stops teaching; below 0.5 it flattens out at the other
  end."""

  tolerance_rate: float = 1.0e-3
  """How fast the running error follows the measured one, per environment step.

  Slow on purpose. The tolerance only descends, so a rate that reacts to a lucky batch
  writes that batch into the task permanently."""

  ##
  # What makes a pair feasible.
  ##

  max_accel: float = 3.0
  """What a humanoid root sustains, in m/s^2. A candidate needing more than this to match
  the target's velocity in the time available is redrawn."""

  joint_speed: float = 4.0
  """Sustained joint rate, in rad/s. Same use, for the joints.

  Far below the actuator velocity limit on purpose. That limit is what a joint reaches
  unloaded, high enough that a check against it never binds, which is how the previous
  bridge came to be trained on pairs whose joints could not get there."""

  attempts: int = 8
  """Candidate targets drawn per environment before the deadline is stretched instead."""

  spread: float = 1.0
  """How far off the natural spot a target may be put, as a share of the reachable disc."""

  travel_speed: float = 1.2
  """Ground a body is assumed able to cover, in m/s, when sizing that disc."""

  turn_speed: float = 1.5
  """Turn a body is assumed able to make, in rad/s, when sizing the heading offset."""

  curriculum_steps: int = 60_000
  """Environment steps over which `spread` ramps from zero. Zero starts at full spread,
  which is what a run resuming from a checkpoint that already crosses easy windows wants."""

  start_noise: float = 1.0
  """Scale on the perturbation applied to the start state. At inference the robot arrives
  from whatever the outgoing skill left behind, never from a dataset row, and a policy that
  has only started exactly on one has never had to steer."""

  def build(self, env: ManagerBasedRlEnv) -> BridgeCommand:
    return BridgeCommand(self, env)
