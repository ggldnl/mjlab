"""The bridge targets a physical state and is rewarded for improving its best arrival.

Command layout, 24 + 2J values:
    root position gap       3, heading frame
    root orientation gap    6
    root velocity gaps      6, body frame
    joint gaps              2J
    reward best             1
    requested tolerances    8, divided by the configured baseline

Each window holds one tolerance profile, shared by the reward, observation and success
check. Training samples each channel independently and tightens the sampling range over
time. Fixed baseline score, errors and best_step remain comparable across the curriculum
and track a separate best moment. arrived and arrived_now use the requested profile.

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
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.dataset import (
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

ARM_JOINT = re.compile(r"(shoulder|elbow|wrist)")
"""What counts as an arm. Everything else, legs and waist alike, is the supporting chain."""

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
  leg_joint_pos: float = 0.10
  """Radians on the worst leg or waist joint, about 6 degrees. At the G1's thigh length that
  is roughly 3 cm of foot placement, which is the scale the root position bound is set at."""
  leg_joint_vel: float = 1.50
  """Radians per second on the worst leg or waist joint."""
  arm_joint_pos: float = 0.05
  """Radians on the worst arm joint, about 3 degrees.

  Tighter than the legs, which is the opposite of what the dynamics suggest, and
  deliberate. An arm has little say in whether the next skill can stand up, so a
  requirement argued from balance alone would be loose, and loose here is exactly the hole
  that let a bridge park its arms wherever it liked. Three degrees is the bar for "same
  posture", and holding the arms to it is what the arm channels exist for.
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
  """One kernel per channel, blended into one number. (N, 8) -> (N,), in (0, 1].

  Two parts:

      average      gradient on every channel at once, so a policy that is bad everywhere
                   still knows which way to move
      bottleneck   the worst channel's kernel alone. Makes "arrive on all of them" the
                   objective rather than "arrive on the cheap ones"

  At bottleneck_weight 0.7 the bottleneck dominates: one channel at zero caps the whole
  score at 0.3 whatever the other seven do.

  The four joint channels outweigh the four root ones in the average. One root state can
  be reached by many postures, so the root channels are the easy half, and left equal they
  are where the policy spends its capacity.
  """
  kernel = torch.exp(-(errors / tolerances).square())
  weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 1.5, 2.0, 1.5], device=errors.device)
  average = (kernel * weights).sum(dim=-1) / weights.sum()
  return (1.0 - bottleneck_weight) * average + bottleneck_weight * kernel.min(
    dim=-1
  ).values


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
    for bounds in (cfg.tolerance_initial_range, cfg.tolerance_final_range):
      if not all(math.isfinite(v) for v in bounds) or not 0 < bounds[0] <= bounds[1]:
        raise ValueError("Tolerance ranges must be finite, positive and ordered")
    if any(
      end > start
      for start, end in zip(
        cfg.tolerance_initial_range, cfg.tolerance_final_range, strict=True
      )
    ):
      raise ValueError("Final tolerance bounds must not exceed initial bounds")
    if cfg.tolerance_steps < 0:
      raise ValueError("tolerance_steps must be nonnegative")
    self.window_tolerances = self.tolerances.expand(self.num_envs, -1).clone()
    """Per request limits, frozen until that environment opens another window."""

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
    """How long this window is allowed to run, in physical seconds.

    Not a deadline and not in the observation. It says when an unsuccessful crossing is
    abandoned, which the policy has no use for: told how long it had, a policy trades
    accuracy for punctuality, and punctuality is not what a hand-over needs.
    """
    return self.patience.float() / self.fps

  @property
  def out_of_patience(self) -> torch.Tensor:
    """Whether the window has run as long as it is allowed to. (num_envs,) bool."""
    return self.step >= self.patience

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

    The duration the crossing actually took. With no deadline this is measured rather than
    commanded, which is the whole point of dropping one: how long a hand-over needs is a
    property of the two states and the body, not a number a caller should have to guess.
    """
    return self.best_step.float() / self.fps

  ##
  # What the policy reads.
  ##

  @property
  def command(self) -> torch.Tensor:
    """(num_envs, 24 + 2J), including reward baseline and requested tolerance scales."""
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

    score = arrival_score(errors, self.window_tolerances)
    improvement = (score - self.best).clamp(min=0.0)
    fixed_score = arrival_score(errors, self.tolerances)
    better = (fixed_score > self.fixed_best) | ~self._has_scored
    self._has_scored[:] = True

    self.best = torch.maximum(self.best, score)
    self.fixed_best = torch.maximum(self.fixed_best, fixed_score)
    self.arrived = torch.maximum(
      self.arrived, arrived(errors, self.window_tolerances).float()
    )
    self.fixed_arrived = torch.maximum(
      self.fixed_arrived, arrived(errors, self.tolerances).float()
    )

    # The errors and the step of the best moment, so every reported number describes one
    # instant of the crossing rather than a mix of the closest the root ever came and the
    # closest the arms ever came
    now = self.state_now()
    q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + self.num_joints)
    qd = slice(ROOT_STATE_DIM + self.num_joints, ROOT_STATE_DIM + 2 * self.num_joints)
    wide = better.unsqueeze(-1)
    self.final = torch.where(wide, errors, self.final)
    self.final_joint_pos = torch.where(
      wide, (now[:, q] - self.target[:, q]).abs(), self.final_joint_pos
    )
    self.final_joint_vel = torch.where(
      wide, (now[:, qd] - self.target[:, qd]).abs(), self.final_joint_vel
    )
    self.best_step = torch.where(better, self.step, self.best_step)

    self.metrics["score"] = self.fixed_best.clone()
    self.metrics["arrived"] = self.arrived.clone()
    self.metrics["fixed_arrived"] = self.fixed_arrived.clone()
    self.metrics["requested_score"] = self.best.clone()
    self.metrics["arrival_s"] = self.arrival_s
    for index, name in enumerate(CHANNELS):
      self.metrics[f"err_{name}"] = self.final[:, index].clone()

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

  def _sample_tolerances(self, count: int) -> torch.Tensor:
    """Sample each request and channel independently in log space."""
    if not self._curriculum:
      return self.tolerances.expand(count, -1)
    steps = self.cfg.tolerance_steps
    alpha = min(max(self._env.common_step_counter / steps, 0.0), 1.0) if steps else 1.0
    initial = self.tolerances.new_tensor(self.cfg.tolerance_initial_range).log()
    final = self.tolerances.new_tensor(self.cfg.tolerance_final_range).log()
    low, high = torch.lerp(initial, final, alpha).unbind()
    draws = torch.rand(count, len(CHANNELS), device=self.device)
    return self.tolerances * (low + draws * (high - low)).exp()

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
    self.place(
      env_ids,
      start,
      target,
      steps.float() / self.fps,
      tolerances=self._sample_tolerances(count),
    )
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
    self.start_distance[env_ids] = (root_pos - self.target[env_ids, 0:3]).norm(dim=-1)
    # Cleared here rather than in place, so a window aimed from outside cannot inherit the
    # crossing of the window before it
    self.has_reference[env_ids] = 0.0
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
    for index, name in enumerate(CHANNELS):
      self.metrics[f"tol_{name}"] = self.window_tolerances[:, index].clone()

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
  matter of what went into the corpus, which is the job of datasets/tracker.py.
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

  tolerance_initial_range: tuple[float, float] = (5.0, 10.0)
  """Initial multipliers of the baseline, sampled independently per channel in log space."""

  tolerance_final_range: tuple[float, float] = (0.5, 4.0)
  """Final multiplier range. Retains mixed precision profiles around the runtime default.

  Tune these bounds to cover measured runtime requests. They are training defaults,
  not measured skill robustness limits. Equal bounds request a single precision scale.
  """

  tolerance_steps: int = 120_000
  """Environment steps to interpolate both bounds in log space. Zero uses final bounds.

  Only newly drawn windows change. The runner restores the step counter on resume.
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

  Must finish well before tolerance_steps, or the arrival kernel reaches the accuracy it
  is actually asking for while the policy is still being paid to imitate.
  """

  start_noise_steps: int = 60_000
  """Environment steps the perturbation takes to widen from nothing to start_noise.

  See BridgeCommand.noise_scale. Shorter than the tolerance schedule on purpose: this one
  makes the task harder and should be finished well before the tolerances arrive at what
  they are actually asking for."""

  def build(self, env: ManagerBasedRlEnv) -> BridgeCommand:
    return BridgeCommand(self, env)
