"""The window: where the bridge starts, where it has to be, and how long it has.

One episode is one window. A start state drawn from a skill's rollout, a target state
drawn from another's, a deadline in control steps, and nothing in between. The robot is
teleported onto the start and the clock runs; what happens in the middle is the policy's
business and is never scored against anything.

##
# Why a pair has to be built rather than drawn
##

Both endpoints come out of the bank individually valid: the robot was in each of them,
under physics, being held up by a policy. That is not enough. Two reachable states can be
an unreachable *pair* -- a body at 3 m/s asked to be standing still two tenths of a second
later is being asked for 15 m/s^2 -- and an episode that cannot be solved is worse than no
episode at all. It contributes a gradient that points nowhere, and enough of them teach a
policy to hedge: go roughly the right way, stay upright, ignore the target.

Three things make a pair feasible, and they work in different ways.

**Placement is feasible by construction.** Only the target's *content* comes from the bank
-- its height, tilt, joint angles, and every velocity. Where it sits and which way it faces
are chosen here, and they are chosen inside a set the robot can reach: the center is where
a body carrying the momentum it already has would arrive if nothing changed, and the offset
around that center is bounded by what a walk could cover in the time available.

**The velocity change is checked.** The one thing drawn from the bank that placement cannot
fix is how fast the target is going. `max_accel` says what a humanoid root sustains, and a
candidate that needs more than that in the time available is rejected.

**The joints are checked too.** The check is a sustained joint rate rather than the actuator
limit, which is far higher than anything a loaded leg achieves and would never bind.

A candidate failing either check is redrawn, a few times. If nothing admissible turns up,
the deadline is stretched to whatever the best candidate needs rather than the pair being
thrown away.

##
# What the policy is told
##

    d_pos    3   where the target is, from here, in the heading frame
    d_rot    6   which way it faces, relative to this heading
    t_lin    3   how fast it is going, in the heading frame
    t_ang    3   and turning
    t_q      J   the joint angles it holds, relative to the default pose
    t_qd     J   and their rates
    clock    2   seconds left, and the fraction of the window left

Position and orientation are differences because a difference is what remains to be
closed; velocities and joint angles are absolute because proprioception already carries
the robot's own, in the same units, and the policy can take the difference itself.

The clock is not optional. A deadline task without time-to-go in the observation is not
Markov -- the same state a quarter of the way through a window and a step from its end
call for opposite actions, and a policy that cannot tell them apart is being asked to
average over the two.
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
from mjlab.tasks.bridging.experiments.humanoid.bridge.dataset.dataset import (
  DEFAULT_BANK,
  ROOT_STATE_DIM,
  Bank,
  load_bank,
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
"""The six ways an arrival can be wrong, kept apart because they have different units and
a tenth of a radian and a tenth of a metre per second are not the same mistake."""

TARGET_COLOR = (1.0, 0.72, 0.2, 0.45)
"""The target ghost."""


##
# Measuring an arrival.
##


@dataclass(kw_only=True)
class Tolerances:
  """How wrong each channel may be and still count as having arrived.

  These are the whole metric, and getting them wrong is not a small error. A tolerance
  wider than the gap that channel has to close makes the channel free: it scores near one
  from the start, contributes nothing to the gradient, and carries the total high enough
  that a robot standing perfectly still looks like it is doing the task. That has already
  happened once on this project -- a statue scored 0.459 against a fully trained policy's
  0.454 -- and the metric could not tell them apart, so neither could PPO.

  So do not take these numbers on faith. `evaluate.py --calibrate` measures the median gap
  per channel over the bank and prints half of it, which is where each of these should sit.
  Run it once when the bank changes.
  """

  root_pos: float = 0.29
  """Metres."""
  root_ori: float = 0.27
  """Radians."""
  root_lin_vel: float = 0.38
  """Metres per second. The channel that matters most for a hand-over and the one that is
  easiest to get wrong: a body in the right pose carrying the wrong momentum is a body
  that is about to be somewhere else."""
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

  Both arguments are bank rows, (N, 13 + 2J). A free function rather than a method,
  because the number has to mean the same thing in the reward, in the metrics and in an
  evaluation that has no live environment. Two implementations of "did it arrive" would
  drift apart, and everything here rests on them agreeing.
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

  The mean of exponentials and not the exponential of a mean. They agree when every
  channel is equally wrong and differ only when they are not: a product is dominated by
  the worst channel and kills every other channel's gradient along with it, while this is
  dominated by the best. With tolerances calibrated tightly the sum is the right choice and
  it is what the tracking tasks in this repository already use. What stops a policy farming
  the easy channels is the tolerances being tight and the episode ending if it strays, not
  the shape of the kernel.
  """
  return torch.exp(-(errors / tolerances).square()).mean(dim=-1)


def arrived(errors: torch.Tensor, tolerances: torch.Tensor) -> torch.Tensor:
  """Whether every channel is inside tolerance. (N,) bool.

  The success metric, and deliberately not the reward. The reward has to be smooth to be
  learnable; this has to be honest to be reportable, and a policy that is inside tolerance
  on five channels out of six has not arrived.
  """
  return (errors <= tolerances).all(dim=-1)


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

    self.bank: Bank = load_bank(cfg.bank_path, str(self.device), cfg.split)
    if self.bank.num_joints != self.num_joints:
      raise ValueError(
        f"The bank holds {self.bank.num_joints}-joint states and this robot has "
        f"{self.num_joints}. Rebuild the bank against this robot."
      )
    self.fps = self.bank.fps
    self.leaving = self.bank.of(cfg.leaving)
    self.entering = self.bank.of(cfg.entering)

    self.tolerances = cfg.tolerances.as_tensor(self.device)
    self.state_dim = self.bank.states.shape[1]

    # The window. `target` is a bank row placed in this environment's world; `deadline` is
    # how many control steps the policy has to reach it.
    self.target = torch.zeros(self.num_envs, self.state_dim, device=self.device)
    self.deadline = torch.full(
      (self.num_envs,), cfg.deadline_range[1], dtype=torch.long, device=self.device
    )
    self.start_distance = torch.zeros(self.num_envs, device=self.device)
    self.stretched = torch.zeros(self.num_envs, device=self.device)

    # Latched at the deadline and held until the next window is drawn. Read by the metrics
    # rather than the live state, because by the time the metrics run the environment has
    # already been reset and the live state is a fresh robot standing at its default pose.
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

    self._ghost: mujoco.MjModel | None = None

  ##
  # Where the episode is.
  ##

  @property
  def step(self) -> torch.Tensor:
    """How many control steps this episode has taken. (num_envs,).

    The environment's own counter rather than one kept here. It is zeroed on reset and
    incremented at the top of every step, before terminations and rewards run, so during
    those it names the step that has just happened, which is exactly what a deadline
    wants. A second counter maintained by this term would have to be advanced in
    `_update_command`, which runs *after* the auto-reset, and would be a step out for
    every environment that just finished.
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

    Ramps from zero over `curriculum_steps` and then holds. At zero every target sits
    exactly where the momentum the robot already has would take it, which is the window a
    skill running on undisturbed would produce and the easiest thing to ask for. At one it
    is anywhere in the disc a walk could reach in the time available.

    A curriculum on distance rather than on the deadline, because they are the same dial
    seen from two ends and one of them is enough.
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
    """(num_envs, 17 + 2J). See this module's header for the layout."""
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
    """The robot as a bank row. (num_envs, 13 + 2J)."""
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

    Called from the reward, which is the only place the arrival can be read at all. The
    reward manager runs before the auto-reset; the metrics run after it, by which point
    an environment that just finished holds a fresh robot at its default pose and every
    number taken from it is a number about a different episode. This project has already
    shipped a success metric that read 100% for exactly that reason.

    So the deadline snapshot is taken here and written straight into `metrics`, where
    `CommandTerm.reset` picks it up on its way past. `_update_metrics` never touches those
    entries.
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

      self.metrics["score"] = self.score.clone()
      self.metrics["arrived"] = self.arrived.clone()
      self.metrics["reached_deadline"] = self.reached.clone()
      for index, name in enumerate(CHANNELS):
        self.metrics[f"err_{name}"] = self.final[:, index].clone()

    return errors

  ##
  # Drawing a window.
  ##

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    count = env_ids.numel()
    low, high = self.cfg.deadline_range

    start = self.bank.states[
      self.leaving[torch.randint(0, self.leaving.numel(), (count,), device=self.device)]
    ].clone()
    deadline = torch.randint(low, high + 1, (count,), device=self.device)

    # A random heading for the start. Everything the policy reads is expressed in its own
    # heading frame and every target is placed relative to it, so this changes no question
    # that is ever asked; it is here so that no part of the task can come to depend on the
    # world frame by accident.
    facing = quat_from_angle_axis(
      torch.rand(count, device=self.device) * (2.0 * math.pi), _up(count, self.device)
    )
    start = _reyaw(start, quat_mul(facing, quat_conjugate(yaw_quat(start[:, 3:7]))))

    target, deadline, stretched = self._draw_target(start, deadline)
    self.place(env_ids, start, target, deadline, stretched)

  def _draw_target(
    self, start: torch.Tensor, deadline: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A target the robot can actually reach from `start` in `deadline` steps.

    Candidates are drawn all at once and the first admissible one per environment is kept,
    rather than looping until something fits. Where nothing fits, the least demanding
    candidate is kept and the deadline is stretched to what it needs.
    """
    count = start.shape[0]
    attempts = self.cfg.attempts
    q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + self.num_joints)

    rows = self.entering[
      torch.randint(0, self.entering.numel(), (count, attempts), device=self.device)
    ]
    candidates = self.bank.states[rows]

    # Turned to face the way this window asks for, before anything is measured: the turn
    # moves the velocity vector, and it is the turned velocity the robot has to arrive
    # carrying.
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
    candidates = _reyaw(
      candidates.reshape(-1, self.state_dim),
      quat_mul(
        facing.reshape(-1, 4),
        quat_conjugate(yaw_quat(candidates.reshape(-1, self.state_dim)[:, 3:7])),
      ),
    ).reshape(count, attempts, self.state_dim)

    # What each candidate would cost, in seconds, on the two things placement cannot fix.
    horizon = (deadline.float() / self.fps).unsqueeze(1)
    speed_up = (candidates[:, :, 7:10] - start[:, None, 7:10]).norm(dim=-1)
    reach = (candidates[:, :, q] - start[:, None, q]).abs().amax(dim=-1)
    need = torch.maximum(speed_up / self.cfg.max_accel, reach / self.cfg.joint_speed)

    fits = need <= horizon
    # The first admissible candidate, or the cheapest one when there is none. `argmax` on a
    # bool picks the first True, and on an all-False row it picks index zero, which is why
    # the fallback is selected explicitly rather than relied upon.
    pick = torch.where(fits.any(dim=1), fits.float().argmax(dim=1), need.argmin(dim=1))
    index = pick.view(-1, 1, 1).expand(-1, 1, self.state_dim)
    target = candidates.gather(1, index).squeeze(1)
    chosen = need.gather(1, pick.view(-1, 1)).squeeze(1)

    stretched = chosen > horizon.squeeze(1)
    deadline = torch.where(
      stretched, (chosen * self.fps).ceil().long().clamp(min=1), deadline
    )

    # Where it goes. The center of the disc is where a body would be if it changed its
    # velocity steadily from the one it has to the one it is being asked for, which is the
    # placement a window produces when nothing has gone wrong. The offset around it is the
    # curriculum: at spread zero every target sits on that spot.
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

    `start` and `target` are (N, 13 + 2J) in one shared frame; both are slid horizontally
    so the start lands on the environment's origin, which keeps the two in one coordinate
    system without the caller having to know where that is. Heights, headings and every
    velocity survive the slide untouched.

    Public because a window does not have to come from the bank. Nothing else calls it
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
    # action it applied and the observation terms hold their history, and a robot
    # teleported without clearing them starts its window carrying a step of somebody
    # else's episode.
    self.robot.reset(env_ids=env_ids)

  def _update_command(self) -> None:
    pass

  def _update_metrics(self) -> None:
    """Only the live numbers.

    The latched ones are written in `errors_now`, before the reset that would destroy the
    state they are read from. Writing them again here would overwrite them with whatever
    the freshly reset robot happens to look like.
    """
    self.metrics["deadline_s"] = self.deadline.float() / self.fps
    self.metrics["spread"] = torch.full_like(self.metrics["spread"], self.spread())
    self.metrics["stretched"] = self.stretched.clone()

  ##
  # Drawing it.
  ##

  def _debug_vis_impl(self, visualizer) -> None:
    """A translucent robot standing in the target.

    A pose and nothing else, because a pose is all that can be drawn: half of what a
    target says is velocity and a still body says nothing about that. What it does show is
    where the window is sending the robot, and after the deadline it stays where it was
    put, so the gap between it and the real robot is the arrival error left standing to be
    looked at.
    """
    if self._ghost is None:
      self._ghost = self._tinted(TARGET_COLOR)

    indexing = self.robot.indexing
    free = indexing.free_joint_q_adr.cpu().numpy()
    joints = indexing.joint_q_adr.cpu().numpy()
    for batch in visualizer.get_env_indices(self.num_envs):
      # From qpos0 rather than from zeros: a zero quaternion is not a rotation, and
      # anything else in the scene keeps its own default.
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
        # made of boxes; everything else in the scene would be a second copy of it hanging
        # in the air beside the target.
        ghost.geom_rgba[geom, 3] = 0.0
    return ghost


##
# Helpers.
##


def _up(count: int, device: torch.device | str) -> torch.Tensor:
  axis = torch.zeros(count, 3, device=device)
  axis[:, 2] = 1.0
  return axis


def _reyaw(states: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
  """The same state facing a different way. (N, 13 + 2J), position left to the caller.

  Every skill here is egocentric: it reads its own body frame, and a tracker reads poses
  relative to an anchor it carries. So a state a skill produced facing one way is a state
  it can be in facing another, and the rotation has to reach the orientation and both
  velocity vectors or the pose and the momentum end up disagreeing about which way the
  body is going.
  """
  out = states.clone()
  out[:, 3:7] = quat_mul(rotation, states[:, 3:7])
  out[:, 7:10] = quat_apply(rotation, states[:, 7:10])
  out[:, 10:ROOT_STATE_DIM] = quat_apply(rotation, states[:, 10:ROOT_STATE_DIM])
  return out


def _rot6d(quat: torch.Tensor) -> torch.Tensor:
  """The first two columns of a rotation matrix, flattened. (..., 4) -> (..., 6).

  Six numbers rather than a quaternion's four because a quaternion has two
  representations for every rotation and the six-number form is unique and continuous.
  """
  matrix = matrix_from_quat(quat)
  return matrix[..., :, :2].transpose(-1, -2).reshape(*quat.shape[:-1], 6)


@dataclass(kw_only=True)
class BridgeCommandCfg(CommandTermCfg):
  entity_name: str = "robot"

  bank_path: Path = DEFAULT_BANK
  split: str = "train"
  leaving: tuple[str, ...] | None = None
  """Which skills a start state may come from. None means any in the bank."""
  entering: tuple[str, ...] | None = None
  """Which skills a target state may come from. None means any in the bank."""

  deadline_range: tuple[int, int] = (15, 60)
  """How many control steps the bridge gets, before any stretching. At 50 Hz that is 0.3
  to 1.2 seconds: long enough to take a step or two, short enough that momentum is still
  the thing that decides the answer."""

  tolerances: Tolerances = field(default_factory=Tolerances)

  ##
  # What makes a pair feasible.
  ##

  max_accel: float = 3.0
  """What a humanoid root sustains, in m/s^2. A candidate needing more than this to match
  the target's velocity in the time available is redrawn."""

  joint_speed: float = 4.0
  """A sustained joint rate, in rad/s, used the same way for the joints.

  Deliberately far below the actuator velocity limit. The limit is what a joint reaches
  unloaded and is high enough that a check against it never binds, which is exactly how
  the previous bridge came to be trained on pairs whose joints could not get there."""

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
  from whatever the outgoing skill left behind, never from a bank row, and a policy that
  has only ever started exactly on one has never had to steer."""

  def build(self, env: ManagerBasedRlEnv) -> BridgeCommand:
    return BridgeCommand(self, env)
