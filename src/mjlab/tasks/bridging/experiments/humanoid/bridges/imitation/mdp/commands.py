"""The bridge targets a physical state and is rewarded for improving its best arrival.

Command layout, 26 + 2J values:
    root position gap       3, heading frame
    root orientation gap    6
    root velocity gaps      6, body frame
    joint gaps              2J
    clock                   2, seconds left of the crossing and the fraction spent
    reward best             1
    requested tolerances    8, divided by the configured baseline

Each window holds one tolerance profile, shared by the reward, observation and success
check. Training draws it from a per channel band, puts one channel strictly and relaxes
the other seven, and tightens a channel's band position only when the policy is meeting it.
See _sample_tolerances and _advance_levels. Fixed baseline score, errors and best_step
remain comparable across the curriculum and track a separate best moment. arrived and
arrived_now use the requested profile.

The previous action is proprioception, not a target. Target actions are recorded for
resumption experiments but are not part of the bridge objective.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DEFAULT_DATASET,
  ROOT_STATE_DIM,
  Dataset,
  Segments,
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
  "leg_joint_pos",
  "leg_joint_vel",
  "arm_joint_pos",
  "arm_joint_vel",
)
"""The 8 ways an arrival can be wrong. See the module header for why they stay apart."""

UNITS: dict[str, str] = {
  "root_pos": "m",
  "root_ori": "rad",
  "root_lin_vel": "m/s",
  "root_ang_vel": "rad/s",
  "leg_joint_pos": "rad",
  "leg_joint_vel": "rad/s",
  "arm_joint_pos": "rad",
  "arm_joint_vel": "rad/s",
}
"""What each channel is measured in. Lives beside CHANNELS because everything that reports
a channel needs it, and two copies would be two chances to mislabel a number."""

ARM_JOINT = re.compile(r"(shoulder|elbow|wrist)")
"""What counts as an arm. Everything else, legs and waist alike, is the supporting chain."""

SUPPORT = ("arm_joint_pos", "arm_joint_vel")
"""Channels that are only worth anything once the other six are right.

An arm has no say in whether the next skill can stand up, so an arm delivered inside
tolerance onto a robot whose root and legs are wrong buys nothing. These two are held to a
looser band than the other six at every moment of training, which is what core_band and
support_band do, and the measurements agree: the kick accepts 0.8 rad on its worst arm
joint, 16 times the baseline, while it wants the root within 4 cm.
"""

CORE = tuple(name for name in CHANNELS if name not in SUPPORT)
"""Root and legs. What a hand-over is actually made of."""


TARGET_COLOR = (1.0, 0.72, 0.2, 0.45)
"""Target ghost: amber, standing still where the window ends."""


REFERENCE_COLOR = (0.35, 0.6, 1.0, 0.35)
"""Reference ghost: blue, walking the recorded crossing as the clock runs.

Fainter than the target on purpose. The target is what the robot is scored on, the
reference is only what it is shaped toward, and by the end of training the shaping is off
and the blue ghost is a demonstration nobody is paid for. It is drawn so the two can be
watched coming apart, which is the whole diagnosis of whether the shaping helps.
"""


##
# Measuring an arrival.
##


def arm_mask(joint_names: tuple[str, ...], device: str | torch.device) -> torch.Tensor:
  """Which joints belong to an arm. (J,) bool.

  By name, because the joint order is the model order and nothing guarantees the arms are
  contiguous. A robot whose names do not match leaves one group empty, which _worst scores
  as perfect rather than raising: a quadruped has no arms to abandon.
  """
  return torch.tensor(
    [bool(ARM_JOINT.search(name)) for name in joint_names],
    device=device,
    dtype=torch.bool,
  )


@dataclass(kw_only=True)
class Tolerances:
  """Arrival limits in physical units, in CHANNELS order.

  The config holds the baseline used for observation scaling and fixed evaluation.
  Each request can supply its own limits through open_window or place.

  Not the requirement any more, only the unit it is quoted in. What training asks for is
  core_band and support_band, in multiples of these numbers, and two of these values no
  longer cover the one skill that has been measured: skills.tolerance at 8 directions puts the
  kick at 0.041 m and 0.101 m/s, under the 0.05 and 0.15 here. Left alone deliberately.
  Moving the baseline moves the observation scaling and the fixed_ metrics with it, so no
  run would be comparable to any earlier one; the band floor of 0.6x reaches 0.030 m and
  0.090 m/s, which covers both with room.
  """

  root_pos: float = 0.05
  """Metres. About a fifth of the G1 foot length, so the support polygon the next skill
  inherits is the one it expects. Tighter is not measurable: a state estimator on hardware
  does not know the pelvis to a centimetre."""
  root_ori: float = 0.05
  """Radians, about 3 degrees of torso tilt or yaw. Small enough that the balance
  controller of the next skill sees a disturbance rather than a different task."""
  root_lin_vel: float = 0.15
  """Metres per second. Over the half second a skill takes to establish itself this is 7 cm
  of drift. The channel that matters most for a hand-over: a body in the right pose
  carrying the wrong momentum is about to be somewhere else."""
  root_ang_vel: float = 0.30
  """Radians per second, about 9 degrees of unwanted turn over that same half second."""
  leg_joint_pos: float = 0.08
  """Radians on the worst leg or waist joint, about 5 degrees. At the G1's thigh length that
  is roughly 2 cm of foot placement, which is the scale the root position bound is set at.

  The two leg channels are the only ones measured against a skill rather than argued, and
  both carry a fifth off what was measured. See leg_joint_vel."""
  leg_joint_vel: float = 0.80
  """Radians per second on the worst leg or waist joint.

  This was the one channel declared looser than a skill turned out to accept: at 1.50 a
  bridge could meet the requirement and still hand over a robot that does not track.

  Both leg bounds sit a fifth below what skills.tolerance measured on the kick,
  which held its clip at 0.10 rad and 1.00 rad/s and left it by 0.15 and 1.50. So the
  measured values were the last rung that passed rather than the edge of anything, and
  they were measured one channel at a time. The margin is for the eight moving together,
  which nothing has measured. The other six channels are already far tighter than the kick
  needs and are argued from what a hand-over costs, not from this."""
  arm_joint_pos: float = 0.05
  """Radians on the worst arm joint, about 3 degrees.

  A unit, not the requirement. What the arms are actually asked for is support_band times
  this, 0.2 to 0.8 rad, which is looser than the legs and is meant to be: an arm has little
  say in whether the next skill can stand up. This value once was the requirement, and
  three degrees on an arm joint is roughly what a bridge that has not solved the root is
  being asked to spend its capacity on.

  The arm channels still exist for the reason they always did, which is that nothing else
  stops a bridge parking its arms wherever it likes. A bound of 0.8 rad still rules that
  out. It just does not rule it out at four times the precision the legs get.
  """
  arm_joint_vel: float = 0.75
  """Radians per second on the worst arm joint."""

  def as_tensor(self, device: str | torch.device) -> torch.Tensor:
    if any(
      not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0
      for name in CHANNELS
    ):
      raise ValueError("Tolerances must be finite and positive")
    return torch.tensor(
      [getattr(self, name) for name in CHANNELS], device=device, dtype=torch.float32
    )


def _worst(errors: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
  """Largest error over a group of joints. (N, J), (J,) -> (N,).

  Worst joint, not mean or RMS. A humanoid has 29, and an average over that many hides a
  handful consistently missed by a lot, which is the failure this channel exists to catch.
  """
  if not bool(group.any()):
    return torch.zeros(errors.shape[0], device=errors.device)
  return errors[:, group].amax(dim=-1)


def channel_errors(
  actual: torch.Tensor, target: torch.Tensor, arms: torch.Tensor
) -> torch.Tensor:
  """How wrong one state is against another, per channel. (N, 8), in natural units.

  Both states are dataset rows, (N, 13 + 2J). arms is the mask from arm_mask and its
  length is where J comes from.

  A free function, not a method: the reward, the metrics and an offline evaluation with no
  live environment all have to get the same number. Two implementations of "did it arrive"
  would drift apart.
  """
  num_joints = int(arms.numel())
  q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + num_joints)
  qd = slice(ROOT_STATE_DIM + num_joints, ROOT_STATE_DIM + 2 * num_joints)
  joint_pos = (actual[:, q] - target[:, q]).abs()
  joint_vel = (actual[:, qd] - target[:, qd]).abs()
  legs = ~arms
  return torch.stack(
    [
      (actual[:, 0:3] - target[:, 0:3]).norm(dim=-1),
      quat_error_magnitude(actual[:, 3:7], target[:, 3:7]),
      (actual[:, 7:10] - target[:, 7:10]).norm(dim=-1),
      (actual[:, 10:ROOT_STATE_DIM] - target[:, 10:ROOT_STATE_DIM]).norm(dim=-1),
      _worst(joint_pos, legs),
      _worst(joint_vel, legs),
      _worst(joint_pos, arms),
      _worst(joint_vel, arms),
    ],
    dim=-1,
  )


def arrival_score(
  errors: torch.Tensor, tolerances: torch.Tensor, bottleneck_weight: float = 0.7
) -> torch.Tensor:
  """How near an arrival is, in one number. (N, 8) -> (N,), in (0, 1].

  Distance first, squashed once at the end. Each channel becomes log(1 + e / tolerance),
  those are aggregated into one distance, and the score is 1 / (1 + distance). See
  benchmarks/objective-proposal.md.

  The distance has two parts:

      bottleneck   the worst channel alone. Makes "arrive on all of them" the objective
                   rather than "arrive on the cheap ones"
      average      every channel at once, so a policy that is bad everywhere still knows
                   which way to move

  The average weights the eight by what a hand-over actually needs, root before legs before
  arms, 6 to 3 to 1. Read the ordering off skills/tolerance: the kick at entry 3 accepts
  0.8 rad on its worst arm joint without leaving its clip and 0.10 m/s on its root linear
  velocity, a difference of two orders of magnitude in how much each channel is worth. A
  hand-over is a body arriving somewhere carrying something; where the hands are is the
  part the next skill can fix for itself.

  This replaced the reverse ordering, which gave the joint channels 2.0 and 1.5 against the
  root's 1.0 on the argument that the root is reachable from many postures and would
  otherwise be solved first. That argument is about which channel is easy, not about which
  one matters, and the measurement settled it: against the kick's measured envelope, root
  linear velocity is the channel furthest outside on 74% of crossings and the two arm
  channels on none of them, while the weights had the arms at four times the root.

  Scores are not comparable across this change, the same way they were not across the
  logarithm below. A crossing with every channel exactly on its limit still reads 0.591,
  since the weights are normalised by their own sum, but any crossing that is uneven across
  the channels moves.

  The logarithm is the whole point and it replaced a per channel exp(-z^2). That kernel is
  flat to machine zero a few tolerances out, and a trained bridge was leaving six
  tolerances of leg joint position error: it read 2e-18 there with a derivative of 2e-17,
  against 0.19 on a root channel already inside its limit. So the worst channel carried
  none of the gradient, the bottleneck term was a constant, and the reward asked for the
  channels that were already met. At that same arrival this form puts 46% of the total
  gradient on the leg joints, which is the channel that is actually wrong.

  Scores are not comparable across that change. A crossing with every channel exactly on
  its limit used to read 0.368 and now reads 0.591.
  """
  # Root, legs, arms at 6 to 3 to 1. In CHANNELS order
  weights = torch.tensor([3.0, 3.0, 3.0, 3.0, 1.5, 1.5, 0.5, 0.5], device=errors.device)
  reach = torch.log1p(errors / tolerances)
  distance = (
    bottleneck_weight * reach.amax(dim=-1)
    + (1.0 - bottleneck_weight) * (reach * weights).sum(dim=-1) / weights.sum()
  )
  return 1.0 / (1.0 + distance)


def arrived(errors: torch.Tensor, tolerances: torch.Tensor) -> torch.Tensor:
  """Every channel inside tolerance. (N,) bool. The success metric, never the reward.

  A reward has to be smooth to be learnable, this has to be honest to be reportable. Seven
  channels out of eight is not an arrival, and with the arms split out that is not a
  technicality: seven out of eight is what a robot that never brought its arms back looks
  like.
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

    self.arms = arm_mask(tuple(self.robot.joint_names), self.device)
    """Which joints the two arm channels are measured over."""

    self.dataset: Dataset | None = None
    self.windows: Segments | None = None
    self._span = 0
    """Longest window in control ticks, and so the width of the reference table."""
    if cfg.dataset_path is None:
      # No corpus. A window then comes from outside through open_window, which is what
      # the transition arena does: it never draws one, so making it load and index the
      # whole training corpus was sixty megabytes read to learn a frame rate
      self.fps = 1.0 / (env.cfg.sim.mujoco.timestep * env.cfg.decimation)
    else:
      self.dataset = load_dataset(cfg.dataset_path, str(self.device), cfg.split)
      if self.dataset.num_joints != self.num_joints:
        raise ValueError(
          f"The dataset holds {self.dataset.num_joints}-joint states and this robot has "
          f"{self.num_joints}. Rebuild the dataset against this robot."
        )
      self.fps = self.dataset.fps
      # The duration range is configured in seconds and becomes control ticks only here,
      # at the simulator boundary. Everything above this line, and the whole external
      # interface, is in seconds
      min_steps = max(1, math.ceil(cfg.duration_s_range[0] * self.fps))
      max_steps = max(min_steps, math.floor(cfg.duration_s_range[1] * self.fps))
      self.windows = self.dataset.segments(
        min_steps, max_steps, self.dataset.of(cfg.sources)
      )
      self._span = max_steps

    self.tolerances = cfg.tolerances.as_tensor(self.device)
    """Fixed baseline for observation scaling, guidance and evaluation."""

    self._curriculum = cfg.adaptive_tolerances and cfg.dataset_path is not None
    """Whether corpus windows sample a tolerance profile."""
    self.window_tolerances = self.tolerances.expand(self.num_envs, -1).clone()
    """Per request limits, frozen until that environment opens another window."""

    ##
    # The curriculum. See _sample_tolerances for the three things it guarantees.
    ##

    self.bands = _bands(cfg, self.device)
    """(8, 2). The widest and the tightest multiple of the baseline each channel is ever
    asked for. A draw never leaves its own row, which is what makes the ordering between
    the arms and the rest hold by construction rather than on average."""

    self.level = torch.zeros(len(CHANNELS), device=self.device)
    """Where along its band each channel currently sits, 0 wide and 1 tight.

    Moved by _advance_levels on evidence, never by the step counter. Not part of any
    checkpoint: a resumed run restarts at cfg.level_init, so a resume that matters should
    read level_<channel> off the last run and pass it."""
    if cfg.level_init is not None:
      if len(cfg.level_init) != len(CHANNELS) or not all(
        0.0 <= v <= 1.0 for v in cfg.level_init
      ):
        raise ValueError(f"level_init needs {len(CHANNELS)} values in [0, 1]")
      self.level = self.level.new_tensor(cfg.level_init)

    if not 0.0 <= cfg.focus_relax < 1.0:
      raise ValueError("focus_relax must be in [0, 1)")
    if cfg.focus_jitter < 1.0:
      raise ValueError("focus_jitter is a multiplicative spread and must be at least 1")
    low, high = cfg.success_band
    if not 0.0 <= low <= high <= 1.0:
      raise ValueError("success_band must be two ordered rates in [0, 1]")
    if not 0.0 < cfg.focus_step <= 1.0 or cfg.focus_samples < 1:
      raise ValueError("focus_step must be in (0, 1] and focus_samples at least 1")

    self._focus = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
    """Which channel this window is the strict one, or -1 for a window aimed from outside."""
    self._hits = torch.zeros(len(CHANNELS), device=self.device)
    self._tries = torch.zeros(len(CHANNELS), device=self.device)
    """Windows closed since this channel's level last moved, and how many met it."""
    self._rate = torch.zeros(len(CHANNELS), device=self.device)
    """Last measured success rate per channel, kept only so it can be logged."""

    self._advanced_at = -1
    self._improvement = torch.zeros(self.num_envs, device=self.device)
    """The step advance last ran on, and what it returned. Everything in it is a write:
    the best of the window moves and the metrics are rewritten. Two
    callers in one environment step would pay the same improvement twice."""

    self._checked = False
    self._opening_gaps: list[torch.Tensor] = []
    """Gaps caught at the instant windows open, held until there are enough for a median.
    Only _check_tolerances touches either."""

    self.state_dim = ROOT_STATE_DIM + 2 * self.num_joints

    ##
    # The recorded crossing. Read by the guidance reward and by the viewer, by nothing
    # else: it is in neither observation group and no metric is measured against it.
    ##

    self.has_reference = torch.zeros(self.num_envs, device=self.device)
    """1 where this window carries a recorded crossing. Zero for a window aimed from
    outside through open_window, which is what a live hand-over is: the robot is already
    somewhere, and nothing recorded the motion from there."""

    self._ref_rows = torch.zeros(
      self.num_envs, self._span + 1, dtype=torch.long, device=self.device
    )
    """Dataset row per control tick. Column k is the state k ticks after the window opened.

    Rows and not states. The states are (13 + 2J) wide and a table of them would be tens
    of megabytes at 4096 environments, for something only one column of which is read per
    step.
    """
    self._ref_rotation = torch.zeros(self.num_envs, 4, device=self.device)
    self._ref_rotation[:, 0] = 1.0
    self._ref_from = torch.zeros(self.num_envs, 3, device=self.device)
    self._ref_to = torch.zeros(self.num_envs, 3, device=self.device)
    """The yaw and the translation place applied to this window, kept so a reference row
    can be moved into the environment the same way its endpoints were. Storing the recipe
    rather than the moved states keeps the table at one row index per tick."""

    # The window. target is a dataset row placed in the world of this environment,
    # patience is how many control steps it is given before the window is abandoned
    self.target = torch.zeros(self.num_envs, self.state_dim, device=self.device)
    self.patience = torch.full(
      (self.num_envs,),
      self.steps_for(torch.tensor(cfg.duration_s_range[1] * cfg.patience_scale)).item(),
      dtype=torch.long,
      device=self.device,
    )
    self.window_steps = torch.full(
      (self.num_envs,),
      self.steps_for(torch.tensor(cfg.duration_s_range[1])).item(),
      dtype=torch.long,
      device=self.device,
    )
    """How long the crossing was asked to take, in control ticks. The clock the policy
    reads, and not the same thing as patience: this is the instruction, patience is how
    long an unsuccessful attempt is left running before it is abandoned.

    Initialized to the longest window rather than to zero. A command is read before any
    window has been opened, and the fraction elapsed divides by this."""
    self.start_distance = torch.zeros(self.num_envs, device=self.device)

    ##
    # The best moment of the window, updated live and held until the next window is drawn.
    #
    # Not a snapshot at some chosen instant. There is no instant to choose any more: the
    # policy may cross the target at any point and a target carrying momentum is one it
    # cannot stay on, so the question "how did this window go" is answered by its best
    # moment and by nothing else. advance keeps these current, and the metrics read them
    # rather than the live state, because the metrics run after the auto-reset and the
    # live state by then is a fresh robot at its default pose
    ##

    self.best = torch.zeros(self.num_envs, device=self.device)
    """Reward baseline observed by the policy, using this window's fixed widths."""

    self.fixed_best = torch.zeros(self.num_envs, device=self.device)
    self._has_scored = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.arrived = torch.zeros(self.num_envs, device=self.device)
    """1 where every channel was inside tolerance at some point this window."""
    self.fixed_arrived = torch.zeros(self.num_envs, device=self.device)

    self.best_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    """Control step the best moment happened on. How long the crossing really took, which
    with no deadline is an output rather than something that was asked for."""

    self.final = torch.zeros(self.num_envs, len(CHANNELS), device=self.device)
    self.final_joint_pos = torch.zeros(
      self.num_envs, self.num_joints, device=self.device
    )
    self.final_joint_vel = torch.zeros(
      self.num_envs, self.num_joints, device=self.device
    )
    """The channel errors, and the per joint errors, at the best moment."""

    self.metrics["arrived"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["fixed_arrived"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["requested_score"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["score"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["arrival_s"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["patience_s"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["start_noise"] = torch.zeros(self.num_envs, device=self.device)
    for name in CHANNELS:
      self.metrics[f"err_{name}"] = torch.zeros(self.num_envs, device=self.device)
    for name in CHANNELS:
      self.metrics[f"tol_{name}"] = torch.zeros(self.num_envs, device=self.device)

    self._ghost: mujoco.MjModel | None = None
    self._reference_ghost: mujoco.MjModel | None = None

  ##
  # Where the episode is.
  ##

  @property
  def step(self) -> torch.Tensor:
    """Control steps this episode has taken. (num_envs,).

    The environment counter, not one kept here. It is zeroed on reset and incremented at
    the top of every step, before terminations and rewards, so during those it names the
    step that just happened. A counter maintained by this term would have to advance in
    _update_command, which runs after the auto-reset, and would be a step out for every
    environment that just finished.
    """
    return self._env.episode_length_buf

  @property
  def patience_s(self) -> torch.Tensor:
    """How long this window is allowed to run before it is abandoned, in seconds.

    Still not in the observation, and still not a deadline. The clock the policy reads is
    window_steps, the crossing it was asked for; this is the slack around it, and knowing
    exactly when an attempt stops being scored is of no use to a crossing.
    """
    return self.patience.float() / self.fps

  @property
  def out_of_patience(self) -> torch.Tensor:
    """Whether the window has run as long as it is allowed to. (num_envs,) bool."""
    return self.step >= self.patience

  @property
  def in_landing(self) -> torch.Tensor:
    """Whether this step is one the arrival may be scored on. (num_envs,) bool.

    Everything, when landing_s is None, which is the behaviour this grew out of: the best
    moment of the whole window counts, whenever it happens.

    With a band, only the steps within landing_s of the duration the window asked for. The
    policy already reads that duration on its clock, so a band is what makes the clock mean
    something: before it, the reward agreed that arriving at any moment was equally good,
    while the observation carried a deadline nothing enforced.

    Symmetric, not one-sided. A target carries momentum, so it is a state the robot passes
    through and not one it can sit in, and a crossing that went through it perfectly three
    ticks early is a good crossing. What the band rules out is the other three tenths of the
    patience overrun, where an arrival is not early or late but untimed.
    """
    if self.cfg.landing_s is None:
      return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    slack = self.steps_for(torch.tensor(self.cfg.landing_s, device=self.device))
    return (self.step >= self.window_steps - slack) & (
      self.step <= self.window_steps + slack
    )

  @property
  def score(self) -> torch.Tensor:
    """Best fixed-tolerance score reached this window, zero before scoring."""
    return self.fixed_best

  @property
  def arrived_now(self) -> torch.Tensor:
    """Whether the current state satisfies every requested arrival requirement."""
    return arrived(self.errors_now(), self.window_tolerances)

  @property
  def arrival_s(self) -> torch.Tensor:
    """When the best moment happened, in seconds since the window opened.

    How long the crossing actually took, against the window_steps it was asked for. Read
    the two together: a best moment landing well before the clock runs out is a policy
    arriving early and drifting, and one landing after it is a policy using the patience
    slack it was not promised.
    """
    return self.best_step.float() / self.fps

  ##
  # What the policy reads.
  ##

  @property
  def clock(self) -> torch.Tensor:
    """Seconds left of the crossing, and the fraction of it already spent. (num_envs, 2).

    The target is a dynamic state, a pose and a momentum at one moment, and reaching one
    is a matter of when to stop closing and start arriving. A policy that cannot tell a
    0.3 second window from a 1.2 second one cannot make that decision and can only learn
    one average approach for every window it is ever given, which is what it did: four
    separate interventions left the arrival error where it was.

    Both go past their end rather than being clamped there. patience_scale leaves a window
    running half as long again as it was given, and a crossing inside that overrun is late,
    which is a thing worth being able to see. Seconds left goes negative and the fraction
    goes past one.
    """
    spent = self.step.float()
    length = self.window_steps.float()
    return torch.stack(
      [(length - spent) / self.fps, spent / length],
      dim=-1,
    )

  @property
  def command(self) -> torch.Tensor:
    """(num_envs, 26 + 2J), including the clock, reward baseline and tolerance scales."""
    data = self.robot.data
    yaw = yaw_quat(data.root_link_quat_w)
    q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + self.num_joints)
    qd = slice(ROOT_STATE_DIM + self.num_joints, ROOT_STATE_DIM + 2 * self.num_joints)

    return torch.cat(
      [
        quat_apply_inverse(yaw, self.target[:, 0:3] - data.root_link_pos_w),
        _rot6d(quat_mul(quat_conjugate(data.root_link_quat_w), self.target[:, 3:7])),
        quat_apply_inverse(
          data.root_link_quat_w, self.target[:, 7:10] - data.root_link_lin_vel_w
        ),
        quat_apply_inverse(
          data.root_link_quat_w,
          self.target[:, 10:ROOT_STATE_DIM] - data.root_link_ang_vel_w,
        ),
        self.target[:, q] - data.joint_pos,
        self.target[:, qd] - data.joint_vel,
        self.clock,
        self.best.unsqueeze(-1),
        self.window_tolerances / self.tolerances,
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

  def reference_now(self) -> torch.Tensor:
    """The recorded state for this control tick. (num_envs, 13 + 2J).

    One column of the row table, moved into the environment by the same yaw and
    translation place applied to the window's endpoints. Undefined where has_reference is
    zero, and the guidance reward multiplies by that rather than branching.

    A yaw and a horizontal slide are the whole transform, which is why this is cheap:
    nothing about the crossing is stretched, retimed or bent to fit. It is either the
    motion that happened, moved, or it is not used.
    """
    tick = self.step.clamp(min=0, max=self._span)
    rows = self._ref_rows.gather(1, tick.unsqueeze(-1)).squeeze(-1)
    state = self.dataset.states[rows] if self.dataset is not None else self.target
    moved = reyaw(state, self._ref_rotation)
    moved[:, 0:3] = self._ref_to + quat_apply(
      self._ref_rotation, state[:, 0:3] - self._ref_from
    )
    return moved

  @property
  def guide_scale(self) -> float:
    """How much the recorded crossing is worth right now, from one down to zero.

    Linear, reaching zero at guide_steps. After that this task is the ordinary bridge
    exactly, which is the point of shaping rather than of adding an objective: the policy
    finally optimized is the one that was always wanted, and the demonstration only said
    where to look first.

    It runs down while noise_scale runs up, which is not a coincidence. The crossing
    starts from the recorded start, so the further the perturbation moves the robot off
    it, the less that particular motion answers the question being asked.
    """
    if self.cfg.guide_steps <= 0:
      return 0.0
    alpha = min(self._env.common_step_counter / self.cfg.guide_steps, 1.0)
    return 1.0 - alpha

  def advance(self) -> torch.Tensor:
    """Score this step, fold it into the best of the window, and return the improvement.

    The whole objective. mdp.arrival pays what this returns, so summed over an episode the
    policy is paid exactly the best arrival score it reached, once, whenever it reached it.

    Why the best moment and not the last one. A target carries momentum, so it is a state
    the robot passes through rather than one it can sit in: scored at a fixed instant, a
    crossing that went through the target perfectly three ticks early reads as a miss. And
    scored every step without the maximum, a policy aiming at a moving target is paid to
    turn round and come back to it, which is the opposite of a hand-over.

    Reward and arrival use the window's requested profile. score and fixed_arrived keep
    the configured baseline so progress remains measurable as the requests get tighter.

    Called from the reward, which is the only place the pre-reset state can be read. The
    reward manager runs before the auto-reset; the metrics run after it, by which point an
    environment that just finished holds a fresh robot at its default pose, so every number
    taken from it describes a different episode. This project once shipped a success metric
    that read 100% for exactly that reason.

    So the metrics are written here, every step, and CommandTerm.reset picks up whatever
    stands when an environment ends. Every step and not only at the end, because there is
    no end to wait for any more: an environment that falls at step nine logs the best it
    reached in those nine steps rather than a zero.
    """
    if self._advanced_at == self._env.common_step_counter:
      return self._improvement
    self._advanced_at = self._env.common_step_counter

    errors = channel_errors(self.state_now(), self.target, self.arms)
    self._check_tolerances(errors)

    # Which environments may be scored this step. Everything, unless a landing band is
    # configured. See in_landing
    landing = self.in_landing
    first = ~self._has_scored

    score = arrival_score(errors, self.window_tolerances)
    improvement = (score - self.best).clamp(min=0.0) * landing
    fixed_score = arrival_score(errors, self.tolerances)
    better = landing & (first | (fixed_score > self.fixed_best))
    self._has_scored |= landing

    self.best = torch.where(landing, torch.maximum(self.best, score), self.best)
    self.fixed_best = torch.where(
      landing, torch.maximum(self.fixed_best, fixed_score), self.fixed_best
    )
    self.arrived = torch.where(
      landing,
      torch.maximum(self.arrived, arrived(errors, self.window_tolerances).float()),
      self.arrived,
    )
    self.fixed_arrived = torch.where(
      landing,
      torch.maximum(self.fixed_arrived, arrived(errors, self.tolerances).float()),
      self.fixed_arrived,
    )

    # The errors and the step of the best moment, so every reported number describes one
    # instant of the crossing rather than a mix of the closest the root ever came and the
    # closest the arms ever came
    now = self.state_now()
    q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + self.num_joints)
    qd = slice(ROOT_STATE_DIM + self.num_joints, ROOT_STATE_DIM + 2 * self.num_joints)
    # Latched at the best scored moment once there is one, and following the live errors
    # until then. Without the second half an episode that fell before its band opened would
    # report the zeros _open left behind, which read as a perfect arrival
    show = better | first
    wide = show.unsqueeze(-1)
    self.final = torch.where(wide, errors, self.final)
    self.final_joint_pos = torch.where(
      wide, (now[:, q] - self.target[:, q]).abs(), self.final_joint_pos
    )
    self.final_joint_vel = torch.where(
      wide, (now[:, qd] - self.target[:, qd]).abs(), self.final_joint_vel
    )
    self.best_step = torch.where(better, self.step, self.best_step)

    # Every channel over its own requirement. 1.0 is the limit whatever the units were, so
    # the eight are comparable to each other and the binding one is the largest. Against
    # the fixed baseline, like score and unlike requested_score, or a curve could fall
    # because the crossing improved or because the curriculum let go
    reach = self.final / self.tolerances

    self.metrics["score"] = self.fixed_best.clone()
    self.metrics["arrived"] = self.arrived.clone()
    self.metrics["fixed_arrived"] = self.fixed_arrived.clone()
    self.metrics["requested_score"] = self.best.clone()
    self.metrics["arrival_s"] = self.arrival_s
    self.metrics["channels_met"] = (reach <= 1.0).sum(dim=-1).float()
    self.metrics["worst_channel"] = reach.amax(dim=-1)
    for index, name in enumerate(CHANNELS):
      self.metrics[f"err_{name}"] = self.final[:, index].clone()
      self.metrics[f"reach_{name}"] = reach[:, index].clone()

    self._improvement = improvement
    return improvement

  def errors_now(self) -> torch.Tensor:
    """The 8 channel errors against the target, this step. Read only.

    advance is what moves the window's state on. This is for a term or a caller that wants
    the live gap without paying anything for it.
    """
    return channel_errors(self.state_now(), self.target, self.arms)

  def _check_tolerances(self, errors: torch.Tensor) -> None:
    """Print once per run: which requirements a motionless robot already meets, and which
    none will reach soon.

    Measures the gap a robot that does nothing still has, and compares each requirement
    against it:

        satisfied by doing nothing   requirement is above the gap, so this channel is not
                                     what makes the task hard. Usually means the quantity
                                     barely changes over a window this long
        out of reach                 requirement is more than 10x below the gap, so
                                     arrived reads zero on it for a long time

    Neither is a reason to edit Tolerances. The requirement is what a hand-over needs,
    this only reports how the task sits against it. An earlier version printed a value to
    paste in, which made the definition of success a function of current difficulty.

    Read at the step a window opens, which is the error a statue would still have when the
    window was abandoned. Windows open a few environments at a time, so gaps are collected
    until there are enough for a median. The gap does not move during a run, so one
    measurement is it.
    """
    if self._checked:
      return

    at_open = self.step == 1
    if bool(at_open.any()):
      self._opening_gaps.append(errors[at_open].clone())
    if sum(g.shape[0] for g in self._opening_gaps) < 256:
      return

    gaps = torch.cat(self._opening_gaps).median(dim=0).values
    self._opening_gaps = []
    self._checked = True

    free = self.tolerances >= gaps
    unreachable = self.tolerances * 10.0 < gaps
    if not bool((free | unreachable).any()):
      return
    print("[bridge] how the requirements sit against a robot that does nothing:")
    for index, name in enumerate(CHANNELS):
      if not bool(free[index] or unreachable[index]):
        continue
      verdict = "satisfied by doing nothing" if free[index] else "out of reach for now"
      print(
        f"  {name:<14} requires {float(self.tolerances[index]):.2f}"
        f"   statue misses by {float(gaps[index]):.2f}   {verdict}"
      )

  def _sample_tolerances(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One window's limits and the channel it puts the question to. (count, 8), (count,).

    Three things hold for every draw this returns.

    The arms are never asked for more precision than the rest. Each channel is confined to
    its own band and the core bands sit entirely below the arm bands, so the ordering is a
    property of the bands, checked once in _bands, rather than something that happens to
    come out right on average. It did not before: the eight channels were drawn from one
    shared range, so half of all windows asked the arms to be tighter than the legs, which
    is the case where a perfect arm is worth nothing.

    One channel is the question and the other seven are the background. The focused
    channel sits as far along its band as its curriculum has earned; the rest are pulled
    back toward wide by focus_relax. The policy is told which one it is, for free: the
    observation already carries window_tolerances / tolerances, so the strict channel is
    visible as the small number among eight.

    Everything tightens over the run, but only on evidence. The band positions only move
    in _advance_levels, which needs the policy to be keeping up first.

    The jitter is what keeps the tolerance input in the observation meaningful. Without it
    a channel's requirement is one number per training phase and the policy can learn it
    instead of reading it.
    """
    if not self._curriculum:
      return self.tolerances.expand(count, -1), torch.full(
        (count,), -1, dtype=torch.long, device=self.device
      )
    focus = torch.randint(len(CHANNELS), (count,), device=self.device)
    position = self.level.expand(count, -1) * self.cfg.focus_relax
    position = position.scatter(1, focus.unsqueeze(-1), self.level[focus].unsqueeze(-1))

    wide = self.bands[:, 0].log().expand(count, -1)
    strict = self.bands[:, 1].log().expand(count, -1)
    spread = math.log(self.cfg.focus_jitter)
    jitter = (torch.rand(count, len(CHANNELS), device=self.device) * 2 - 1) * spread
    multiplier = (torch.lerp(wide, strict, position) + jitter).exp()
    # Back into the band. The jitter is the one thing here that could cross the ordering
    multiplier = multiplier.clamp(self.bands[:, 1], self.bands[:, 0])
    return self.tolerances * multiplier, focus

  def _record_outcomes(self, env_ids: torch.Tensor) -> None:
    """Fold the windows that are closing into the counters their focus channel is gated on.

    Read at the best moment of the window and not at any moment. A channel that dipped
    inside its limit at some point while the rest of the robot was elsewhere has not
    arrived, and a curriculum gated on that would tighten away from a policy that is not
    actually there yet.
    """
    if not self._curriculum:
      return
    live = (self._focus[env_ids] >= 0) & self._has_scored[env_ids]
    ids = env_ids[live]
    if ids.numel() == 0:
      return
    focus = self._focus[ids]
    column = focus.unsqueeze(-1)
    met = self.final[ids].gather(1, column) <= self.window_tolerances[ids].gather(
      1, column
    )
    self._tries.index_add_(0, focus, torch.ones_like(focus, dtype=self._tries.dtype))
    self._hits.index_add_(0, focus, met.squeeze(-1).to(self._hits.dtype))
    self._advance_levels()

  def _advance_levels(self) -> None:
    """Move each channel's band position, on evidence. Florensa's band, per channel.

    A channel whose focused windows are met more often than success_band's upper rate is
    solved at the precision it is being asked for, so it is asked for more. One met less
    often than the lower rate is past what the policy can do, so it is asked for less. In
    between, the requirement is sitting exactly where the samples are worth something and
    nothing moves.

    This is the whole difference from the step ramp it replaces. That one tightened on the
    environment step counter whether or not anything was being learned, and did: across
    four separate runs the worst channel sat at 5.5 times its limit for thousands of
    iterations while the schedule went on demanding more of it. A requirement the policy
    has no chance of meeting produces failures it cannot learn anything from.

    Per channel and not global, so a channel that is stuck cannot hold back the seven that
    are not. Each one has its own counters and moves on its own evidence.
    """
    ready = self._tries >= self.cfg.focus_samples
    if not bool(ready.any()):
      return
    rate = self._hits / self._tries.clamp(min=1.0)
    low, high = self.cfg.success_band
    step = torch.zeros_like(rate)
    step = torch.where(rate >= high, torch.full_like(rate, self.cfg.focus_step), step)
    step = torch.where(rate <= low, torch.full_like(rate, -self.cfg.focus_step), step)
    self.level = torch.where(ready, (self.level + step).clamp(0.0, 1.0), self.level)
    self._rate = torch.where(ready, rate, self._rate)
    self._hits = torch.where(ready, torch.zeros_like(self._hits), self._hits)
    self._tries = torch.where(ready, torch.zeros_like(self._tries), self._tries)

  def _request_tolerances(
    self, count: int, tolerances: Tolerances | torch.Tensor | None
  ) -> torch.Tensor:
    """Validate an explicit profile, or use the baseline for an external request."""
    if tolerances is None:
      return self.tolerances.expand(count, -1)
    values = (
      tolerances.as_tensor(self.device)
      if isinstance(tolerances, Tolerances)
      else tolerances.to(device=self.device, dtype=self.tolerances.dtype)
    )
    if values.shape not in ((len(CHANNELS),), (count, len(CHANNELS))):
      raise ValueError(f"Expected tolerances with shape (8,) or ({count}, 8)")
    if not bool(torch.isfinite(values).all() and (values > 0).all()):
      raise ValueError("Tolerances must be finite and positive")
    return values.expand(count, -1)

  ##
  # Drawing a window.
  ##

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    if self.windows is None or self.dataset is None:
      raise RuntimeError(
        "This command has no corpus to draw a window from. Either give it a dataset_path "
        "or drive it from outside with `open_window`, which is what the transition arena "
        "does."
      )
    # Before place, which is where the closing window's numbers are cleared
    self._record_outcomes(env_ids)
    count = env_ids.numel()
    start_rows, target_rows, steps, position = self.windows.draw(count)
    start = self.dataset.states[start_rows].clone()
    target = self.dataset.states[target_rows].clone()

    # A random heading for the start. Everything the policy reads is in its own heading
    # frame and the target is carried around with it, so this changes no question that is
    # ever asked. It is here so nothing can come to depend on the world frame by accident
    facing = quat_from_angle_axis(
      torch.rand(count, device=self.device) * (2.0 * math.pi), _up(count, self.device)
    )
    rotation = quat_mul(facing, quat_conjugate(yaw_quat(start[:, 3:7])))
    origin = start[:, 0:3].clone()
    start, target = reframe_pair(start, target, rotation)
    tolerances, focus = self._sample_tolerances(count)
    self.place(env_ids, start, target, steps.float() / self.fps, tolerances=tolerances)
    # After place, which clears the focus along with every other latch
    self._focus[env_ids] = focus
    if self.dataset.previous_action is not None:
      self._env.action_manager.action[env_ids] = self.dataset.previous_action[
        start_rows
      ]

    # After place, which clears the reference along with every other latch. origin is the
    # start position before the reframe, which is also after it: a yaw turns a state
    # without moving it. landed is where the nominal start went, recomputed rather than
    # read back off the robot, because the robot is perturbed after being put there and
    # the crossing is anchored to the window, not to the noise
    landed = start[:, 0:3].clone()
    landed[:, 0:2] = self._env.scene.env_origins[env_ids][:, :2]
    self._ref_rows[env_ids] = self.windows.path(position, steps, self._span)
    self._ref_rotation[env_ids] = rotation
    self._ref_from[env_ids] = origin
    self._ref_to[env_ids] = landed
    self.has_reference[env_ids] = 1.0

  ##
  # The interface: a state to leave from, a state to arrive in, and a duration in seconds.
  ##

  def steps_for(self, duration_s: torch.Tensor) -> torch.Tensor:
    """Seconds to control ticks. The only place the conversion happens.

    A tick count is an implementation detail: it changes with the decimation and means
    nothing to a caller choosing between a 1.0 s target and a 0.5 s one. Seconds are the
    currency everywhere above this line.
    """
    return (duration_s * self.fps).round().long().clamp(min=1)

  def open_window(
    self,
    env_ids: torch.Tensor,
    duration_s: torch.Tensor,
    *,
    tolerances: Tolerances | torch.Tensor | None = None,
  ) -> None:
    """Open a window on the target currently held, allowing this many seconds. Teleports
    nobody.

    place is this plus a start state to teleport onto. A live hand-over already has the
    robot where it wants it, so it calls this instead.

    duration_s is how long the crossing is expected to take, not a deadline. It buys
    patience_scale times that much patience and never reaches the policy. A caller with no
    opinion should pass the middle of duration_s_range: asking for too little only abandons
    a crossing that was going to work.

    tolerances accepts physical limits as Tolerances, an (8,) tensor or an (N, 8) tensor
    aligned with env_ids. None requests the configured baseline, without sampling.
    """
    self._open(
      env_ids,
      duration_s,
      self.robot.data.root_link_pos_w[env_ids],
      self._request_tolerances(env_ids.numel(), tolerances),
    )

  def _open(
    self,
    env_ids: torch.Tensor,
    duration_s: torch.Tensor,
    root_pos: torch.Tensor,
    tolerances: torch.Tensor,
  ) -> None:
    """Set the patience and clear every latch from the window before.

    root_pos is where the robot starts, passed in rather than read, because place calls
    this before it has written the teleport to the simulator and the live buffers still
    hold the previous episode.
    """
    self.patience[env_ids] = self.steps_for(duration_s * self.cfg.patience_scale)
    self.window_steps[env_ids] = self.steps_for(duration_s)
    self.start_distance[env_ids] = (root_pos - self.target[env_ids, 0:3]).norm(dim=-1)
    # Cleared here rather than in place, so a window aimed from outside cannot inherit the
    # crossing of the window before it
    self.has_reference[env_ids] = 0.0
    # A window aimed from outside has no focused channel and is not evidence about one
    self._focus[env_ids] = -1
    # best has to go back to zero or the next window opens already paid for the last one,
    # and since it is in the observation the policy would read a crossing that never began
    # as one nearly finished
    self.best[env_ids] = 0.0
    self.fixed_best[env_ids] = 0.0
    self._has_scored[env_ids] = False
    self._improvement[env_ids] = 0.0
    self.window_tolerances[env_ids] = tolerances
    self.arrived[env_ids] = 0.0
    self.fixed_arrived[env_ids] = 0.0
    self.best_step[env_ids] = 0
    self.final[env_ids] = 0.0
    self.final_joint_pos[env_ids] = 0.0
    self.final_joint_vel[env_ids] = 0.0

  def place(
    self,
    env_ids: torch.Tensor,
    start: torch.Tensor,
    target: torch.Tensor,
    duration_s: torch.Tensor,
    *,
    tolerances: Tolerances | torch.Tensor | None = None,
  ) -> None:
    """Open a window on these environments and teleport the robot onto its start.

    Args:
      env_ids: which environments.
      start, target: (N, 13 + 2J) dataset rows in one shared frame.
      duration_s: how far apart in time the two ends were drawn. See open_window.
      tolerances: requested physical limits. See open_window.

    Both states slide horizontally so the start lands on the environment origin, which
    keeps them in one coordinate system without the caller knowing where that is. Heights,
    headings and velocities are untouched by the slide.

    The target is written once, here, in world coordinates, and nothing moves it until the
    next window. That is the difference between a goal and a carrot.
    """
    if env_ids.numel() == 0:
      return
    profile = self._request_tolerances(env_ids.numel(), tolerances)

    origin = self._env.scene.env_origins[env_ids]
    shift = origin[:, :2] - start[:, :2]
    target = target.clone()
    target[:, 0:2] += shift

    root_pos = start[:, 0:3].clone()
    root_pos[:, 0:2] = origin[:, :2]

    self.target[env_ids] = target
    self._open(env_ids, duration_s, root_pos, profile)

    root_quat = start[:, 3:7].clone()
    root_lin_vel = start[:, 7:10].clone()
    root_ang_vel = start[:, 10:ROOT_STATE_DIM].clone()
    joint_pos = start[:, ROOT_STATE_DIM : ROOT_STATE_DIM + self.num_joints]
    joint_vel = start[:, ROOT_STATE_DIM + self.num_joints :]
    scale = self.noise_scale
    if scale > 0.0:
      root_pos = root_pos + torch.randn_like(root_pos) * scale * 0.02
      axis = torch.randn_like(root_pos)
      axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
      root_quat = quat_mul(
        quat_from_angle_axis(
          torch.randn(env_ids.numel(), device=self.device) * scale * 0.05, axis
        ),
        root_quat,
      )
      root_lin_vel = root_lin_vel + torch.randn_like(root_lin_vel) * scale * 0.15
      root_ang_vel = root_ang_vel + torch.randn_like(root_ang_vel) * scale * 0.15
      joint_pos = joint_pos + torch.randn_like(joint_pos) * scale * 0.03
      joint_vel = joint_vel + torch.randn_like(joint_vel) * scale * 0.3

    limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(limits[:, :, 0], limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    self.robot.write_root_state_to_sim(
      torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1),
      env_ids=env_ids,
    )
    # Not optional. qpos and qvel are not the whole state: the action term holds the last
    # action it applied and the observation terms hold their history. A robot teleported
    # without clearing them starts its window carrying a step of somebody else's episode
    self.robot.reset(env_ids=env_ids)

  @property
  def noise_scale(self) -> float:
    """How hard the start state is perturbed right now. Ramps from 0 to start_noise.

    At inference the bridge takes over from whatever the outgoing skill left behind, never
    from a dataset row, so a policy that has only ever started exactly on one has never
    had to steer.

    The perturbation is also the one thing here that can make a window unreachable: the
    endpoints are demonstrated, and pushing the start off its demonstrated value breaks
    that. Ramping means the task is solvable while the policy is learning what it is.
    """
    if self.cfg.start_noise <= 0.0:
      return 0.0
    alpha = min(self._env.common_step_counter / max(self.cfg.start_noise_steps, 1), 1.0)
    return self.cfg.start_noise * alpha

  def _update_command(self) -> None:
    """Keep the window's best moment current even where no reward runs.

    advance is normally called by mdp.arrival, which is where it has to be during training:
    the reward manager runs before the auto-reset, and everything advance latches is read
    off a state the reset is about to destroy.

    The transition arena and the parkour demo have no reward manager at all, so without this
    nothing would ever call it there. That is not a missing log line: `best` is half of what
    the policy reads, so a bridge whose best never moved would spend every inference step on
    an observation training never produced.

    Idempotent, so in training this is the second call of the step and does nothing. The two
    callers see the state one physics substep apart, which is the same lag the reward and
    the metrics already differ by.
    """
    self.advance()

  def _update_metrics(self) -> None:
    """Only the live numbers.

    advance writes the rest before the reset that destroys the state they are read from.
    Writing them again here would overwrite them with a freshly reset robot.
    """
    self.metrics["patience_s"] = self.patience_s
    self.metrics["start_noise"] = torch.full_like(
      self.metrics["start_noise"], self.noise_scale
    )
    ones = torch.ones_like(self.metrics["start_noise"])
    for index, name in enumerate(CHANNELS):
      self.metrics[f"tol_{name}"] = self.window_tolerances[:, index].clone()
      # Where the curriculum has got to, and the evidence that moved it there. Per channel
      # scalars broadcast over the environments, because this is what the metric dict is
      self.metrics[f"level_{name}"] = ones * self.level[index]
      self.metrics[f"focus_rate_{name}"] = ones * self._rate[index]
    self.metrics["level"] = ones * self.level.mean()

  ##
  # Drawing it.
  ##

  def _debug_vis_impl(self, visualizer) -> None:
    """Two translucent robots: the target, and the crossing that leads to it.

    The amber one stands in the target. A pose and nothing else, because a pose is all
    that can be drawn: half of a target is velocity and a still body says nothing about
    that. It does show where the window is sending the robot, and it stays where it was put
    for the whole window, so the gap to the real robot is the arrival error left standing to
    be looked at.

    The blue one walks the recorded crossing, a frame per control tick, and ends up inside
    the amber one because the last frame of the crossing is the target. It
    is what guidance is paying for, so watching the robot fall behind it is watching the
    shaping fail to take. Watching the robot follow it and still miss the target would
    mean the crossing is being tracked and the arrival is not.

    Drawn only where there is a crossing to draw. A window aimed from outside through
    open_window has none, and a blue robot standing at the last window's pose would be a
    picture of something that is not happening.
    """
    if self._ghost is None:
      self._ghost = self._tinted(TARGET_COLOR)
    if self._reference_ghost is None:
      self._reference_ghost = self._tinted(REFERENCE_COLOR)

    indexing = self.robot.indexing
    free = indexing.free_joint_q_adr.cpu().numpy()
    joints = indexing.joint_q_adr.cpu().numpy()
    reference = self.reference_now().detach().cpu().numpy()
    has_reference = self.has_reference.detach().cpu().numpy()
    target = self.target.detach().cpu().numpy()

    def pose(row: np.ndarray) -> np.ndarray:
      # From qpos0 rather than zeros: a zero quaternion is not a rotation, and anything
      # else in the scene keeps its own default
      qpos = np.array(self._env.sim.mj_model.qpos0, dtype=np.float64)
      qpos[free[0:3]] = row[0:3]
      qpos[free[3:7]] = row[3:7]
      qpos[joints] = row[ROOT_STATE_DIM : ROOT_STATE_DIM + self.num_joints]
      return qpos

    for batch in visualizer.get_env_indices(self.num_envs):
      visualizer.add_ghost_mesh(
        pose(target[batch]),
        model=self._ghost,
        alpha=TARGET_COLOR[3],
        label=f"target_{batch}",
      )
      if has_reference[batch] > 0.0:
        visualizer.add_ghost_mesh(
          pose(reference[batch]),
          model=self._reference_ghost,
          alpha=REFERENCE_COLOR[3],
          label=f"reference_{batch}",
        )

  def _tinted(self, color: tuple[float, float, float, float]) -> mujoco.MjModel:
    """This scene's model with the robot painted color and everything else hidden."""
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


def _bands(cfg: BridgeCommandCfg, device: str | torch.device) -> torch.Tensor:
  """Each channel's widest and tightest multiple of the baseline. (8, 2).

  Where the ordering guarantee is enforced, once, rather than per draw: the widest any
  core channel is ever asked for has to be at least as tight as the tightest any arm
  channel is ever asked for. With that true of the bands it is true of every window, and
  _sample_tolerances only has to keep its draws inside their row.
  """
  rows = []
  for name in CHANNELS:
    band = cfg.band_overrides.get(
      name, cfg.support_band if name in SUPPORT else cfg.core_band
    )
    if (
      len(band) != 2
      or not all(math.isfinite(v) for v in band)
      or not 0 < band[1] <= band[0]
    ):
      raise ValueError(f"{name}: band must be a finite, positive (wide, strict) pair")
    rows.append(list(band))
  table = torch.tensor(rows, device=device, dtype=torch.float32)

  is_core = [name in CORE for name in CHANNELS]
  core = [i for i, yes in enumerate(is_core) if yes]
  support = [i for i, yes in enumerate(is_core) if not yes]
  if core and support:
    widest = float(table[core, 0].max())
    tightest = float(table[support, 1].min())
    if widest > tightest:
      raise ValueError(
        f"Root and legs reach {widest:.2f}x the baseline at their widest and the arms "
        f"reach {tightest:.2f}x at their tightest, so a window could ask the arms for "
        "more precision than the legs. Lower the wide end of core_band, or raise the "
        "strict end of support_band."
      )
  return table


def _up(count: int, device: torch.device | str) -> torch.Tensor:
  axis = torch.zeros(count, 3, device=device)
  axis[:, 2] = 1.0
  return axis


def reyaw(states: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
  """The same state facing a different way. (N, 13 + 2J). Position is left to the caller.

  Every policy here is egocentric, so a state produced facing one way is a state the robot
  can be in facing another. The rotation has to reach the orientation and both velocity
  vectors, or the pose and the momentum disagree about which way the body is going.
  """
  out = states.clone()
  out[:, 3:7] = quat_mul(rotation, states[:, 3:7])
  out[:, 7:10] = quat_apply(rotation, states[:, 7:10])
  out[:, 10:ROOT_STATE_DIM] = quat_apply(rotation, states[:, 10:ROOT_STATE_DIM])
  return out


def reframe_pair(
  start: torch.Tensor, target: torch.Tensor, rotation: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Apply one yaw change to a demonstrated transition without changing its displacement."""
  displacement = target[:, 0:3] - start[:, 0:3]
  start = reyaw(start, rotation)
  target = reyaw(target, rotation)
  target[:, 0:3] = start[:, 0:3] + quat_apply(rotation, displacement)
  return start, target


def _rot6d(quat: torch.Tensor) -> torch.Tensor:
  """First two columns of a rotation matrix, flattened. (..., 4) -> (..., 6).

  Six numbers rather than the four of a quaternion: a quaternion has two representations
  for every rotation, and this form is unique and continuous.
  """
  matrix = matrix_from_quat(quat)
  return matrix[..., :, :2].transpose(-1, -2).reshape(*quat.shape[:-1], 6)


@dataclass(kw_only=True)
class BridgeCommandCfg(CommandTermCfg):
  entity_name: str = "robot"

  dataset_path: Path | None = DEFAULT_DATASET
  """The corpus windows are drawn from, or None for a command aimed entirely from outside.

  None is what the transition arena wants: it never draws a window, so loading a corpus
  there reads a large file to learn a frame rate the environment already knows."""
  split: str = "train"
  sources: tuple[str, ...] | None = None
  """Which skills or clips a window may come from. None means any in the dataset.

  One filter, not one per end. A window is a contiguous stretch of a single rollout, so
  both ends are always the same source, and asking for a start from one skill and a target
  from another describes nothing this dataset contains. Covering a posture family is a
  matter of what went into the corpus, which is the job of dataset/tracker.py.
  """

  duration_s_range: tuple[float, float] = (0.3, 1.2)
  """How far apart in time the two ends of a window are drawn, in seconds.

  A feasibility device, not a deadline. Cutting both ends out of one rollout this far apart
  is what proves a crossing between them exists; the policy is never told the number and is
  not scored on matching it.

  The simulator still advances in discrete control ticks, but the dataset, this config and
  the external bridge interface are all in seconds. BridgeCommand.steps_for is the only
  conversion.
  """

  landing_s: float | None = None
  """How near the asked duration an arrival has to be to count, in seconds, or None for
  anywhere in the window.

  None is what this task has always done and what imitation still does: the best moment of
  the whole window is the score, whenever it happened. That was argued from the target
  carrying momentum, which makes it a state the robot passes through rather than one it can
  sit in, so a fixed instant would read a perfect crossing three ticks early as a miss.

  The argument is sound and the conclusion overshot. It also handed the policy the whole
  patience overrun, 1.5 times the duration, with no reason to hit the mark at the time it
  was asked for, while the observation carried a clock counting down to a deadline nothing
  enforced. A band keeps the protection and removes the rest.

  Measured on the first full distillation run before any band existed: the best moment
  landed at 1.02 times the asked duration in the median, p10 0.94 and p90 1.19, and scoring
  at exactly the asked duration instead of at the free best moment moved the aggregate from
  0.406 to 0.394. So the freedom was not being exploited by that policy, which is not the
  same as saying a policy trained without it would be no better: the same run put each
  channel's own minimum a median of 4 to 8 control steps away from the scored instant, and a
  band is what forces the eight to coincide.

  Interacts with the shortest windows. 0.15 s is 8 control steps, which is half of a 0.3 s
  crossing and an eighth of a 1.2 s one, so the band is loose for short windows by
  construction. That is a statement about how short those crossings are, not a defect: a
  hand-over's timing slack is physical and does not scale with how long the approach took.
  """

  patience_scale: float = 1.5
  """How much longer than the drawn duration a window is allowed to run.

  The pair is a duration apart because the recorded policy covered it in that time, and the
  bridge is a different policy solving a harder version of the problem: it starts perturbed
  off the recorded state and has no reference to follow. Held to exactly the recorded
  duration it would be scored on being as quick as the demonstration, which is not the
  requirement.

  Slack rather than a large fixed budget, so a short window stays a short episode. Above
  about 2 the tail of every episode is a robot that has already done its best sitting out
  the clock, which is sample time spent on nothing.
  """

  tolerances: Tolerances = field(default_factory=Tolerances)
  """Fixed baseline for scaling and evaluation, also the default external request.

  Keep this identical between training and inference. Supply a different request through
  open_window or place instead of changing the observation's normalization baseline.
  """

  adaptive_tolerances: bool = True
  """Sample tolerance profiles for corpus windows. Disabled for play and live requests."""

  core_band: tuple[float, float] = (4.0, 0.6)
  """Widest and tightest multiple of the baseline asked of root and legs, in that order.

  0.6 is the floor because it is what the one measured consumer needs. skills.tolerance at 8
  directions puts the kick at entry 3 at 0.67 times the baseline on root_lin_vel, which is
  the binding channel; 0.6 clears it, and clears root_pos at 0.82 with room to spare.

  4.0 at the wide end is the ordering constraint and not much else: it is the tightest the
  arms are ever asked for, so it is the loosest the legs can be without inverting the two.
  """

  support_band: tuple[float, float] = (16.0, 4.0)
  """The same for the arm channels. Looser than core_band at both ends, by construction.

  16 is the kick's measured arm_joint_pos, 0.8 rad, and that was a floor: the ladder ran
  out before the policy did. The old schedule asked for 0.025 to 0.2 rad here, up to 32
  times tighter than any consumer has ever wanted, while arm_joint_pos carries the joint
  highest reward weight. That is a large share of the gradient spent on a requirement
  nobody asked for.
  """

  band_overrides: dict[str, tuple[float, float]] = field(default_factory=dict)
  """Per channel bands, for a channel the group default is wrong for.

  root_ang_vel is the known one: the kick accepts 1.3 rad/s, 4.33 times the baseline, and
  the core band tops out at 4. Left at the default for now so the first run of this
  curriculum changes one thing, but it is a candidate.
  """

  focus_relax: float = 0.3
  """How far back toward wide the seven unfocused channels sit, as a fraction of level.

  Zero puts them at the wide end of their band whatever the focused one is doing, which is
  the purest form of one question at a time and also throws away everything the other
  seven have learned. 0.3 keeps them meaningfully easier than the focused channel without
  letting them rot.
  """

  focus_jitter: float = 1.3
  """Multiplicative spread around a channel's band position, clamped back into the band.

  Without it a requirement is one number per phase and the policy can memorize it rather
  than read it off the observation, which would make the tolerance input decoration.
  """

  success_band: tuple[float, float] = (0.3, 0.7)
  """Florensa's R_min and R_max. Tighten above the upper rate, back off below the lower.

  Wider than the paper's (0.1, 0.9) on purpose. That band is measured over starts sampled
  around one goal; this is measured over a channel of a mixed corpus, so the rate is
  noisier and a narrow band would move the level on noise.
  """

  focus_step: float = 0.05
  """How far a channel's band position moves when it does move. 20 moves end to end."""

  focus_samples: int = 512
  """Closed windows focusing a channel before its rate is trusted enough to act on."""

  level_init: tuple[float, ...] | None = None
  """Band positions to start from, one per channel, or None for wide.

  The curriculum is evidence and not a step counter, so unlike the schedule it replaces it
  is not recovered from the restored step counter on a resume. Read level_<channel> off
  the run being resumed and pass it here.
  """

  start_noise: float = 1.0
  """Full scale of the perturbation applied to the start state.

  Every component of the state is perturbed, not just position and joint angles: an
  interruption or a state estimator can be wrong about a root orientation or a velocity
  too, and those are the ones a bridge has to steer out of.
  """

  guide_steps: int = 60_000
  """Environment steps the guidance shaping takes to fade from full weight to nothing.

  Zero switches it off, which turns this task into the bridge without shaping.

  Matched to start_noise_steps. The crossing starts from the recorded start, so the
  further the perturbation moves the robot off it, the less that motion answers the
  question.

  Must be over long before the tolerance curriculum tightens far, or the arrival kernel
  reaches the accuracy it is actually asking for while the policy is still being paid to
  imitate. The curriculum moves on evidence now, so this is no longer two step counters
  to line up: the levels simply will not advance while the shaping is competing.
  """

  start_noise_steps: int = 60_000
  """Environment steps the perturbation takes to widen from nothing to start_noise.

  See BridgeCommand.noise_scale. Finished early on purpose: this one
  makes the task harder and should be finished well before the tolerances arrive at what
  they are actually asking for."""

  def build(self, env: ManagerBasedRlEnv) -> BridgeCommand:
    return BridgeCommand(self, env)
