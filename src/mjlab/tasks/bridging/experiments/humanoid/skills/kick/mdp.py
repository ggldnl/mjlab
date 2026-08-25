"""The kicking skill: what the robot sees, what it is paid for, and how the ball is set up.

The task in one sentence: a G1 stands, a size 5 football sits within reach of its right
foot, and a command says how fast and in which direction that ball should be travelling
once it has been struck.

Three ideas run through everything below.

The command is a launch velocity, not a target point. A ground target would only pay out
once the ball had finished rolling, several seconds after the action that decided where
it went. A launch velocity is measurable at the instant of contact, which is the instant
the policy controls, so the goal signal and the action that produces it sit in the same
place. The command reaches the observation as a vector in the robot's own heading frame,
and it reaches the reward as the thing a struck ball's velocity is compared against.
Both, or it is not conditioning.

Standing comes first, and is enforced rather than hoped for. Every episode opens with a
stance window during which the kick terms pay nothing and disturbing the ball costs. The
window's progress is an observation, so the gate is visible to the policy rather than
being an unexplained change in the reward. On top of that the training curriculum holds
the kick weights at zero for the first stretch of training, so the first thing ever
learned is how to stand still. See kick_env_cfg.py.

The outcome is latched, and paid per step afterwards. A one shot bonus at the moment of
contact is worth a rounding error against a few hundred steps of standing reward, so it
cannot compete with simply standing. Instead, touching the ball and launching it well
each raise the floor for the rest of the episode. That gives a three rung ladder the
policy can climb one rung at a time: stand, touch the ball, send it where it was asked.

The latch tracks the fastest the ball has gone so far rather than the first time it
moved. A policy that nudges the ball before it can really kick it would otherwise lock
that nudge in as its launch velocity for the whole episode, with no way back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

# The velocity task's terms come in first and the kick specific ones are layered on top,
# so a config can reach everything through this one namespace. This also brings in
# mjlab's generic terms, which the velocity task re-exports.
from mjlab.tasks.velocity.mdp import *  # noqa: F401, F403
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  sample_uniform,
  wrap_to_pi,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


##
# Names and geometry.
##

ROBOT = "robot"
BALL = "ball"

COMMAND_NAME = "kick"

# Contact sensors, defined in kick_env_cfg.py.
FOOT_BALL_SENSOR = "foot_ball_contact"
FEET_GROUND_SENSOR = "feet_ground_contact"

# The foot that does the kicking, and the one that has to hold the robot up while it
# happens. The ball spawns in front of the kicking foot only, so these are not
# interchangeable; moving the skill to the other leg means moving both.
KICK_SITE = "right_foot"
SUPPORT_SITE = "left_foot"

# Size 5 football, matching the defaults in asset_zoo/objects/ball.
BALL_RADIUS = 0.11

# How long the robot must stand before the kick terms turn on, in seconds.
STANCE_TIME_S = 1.0

# Horizontal ball speed above which the ball counts as launched rather than merely
# jostled. Below this the kick reward stays at zero, so a policy that only leans on the
# ball earns nothing for it.
LAUNCH_SPEED = 0.5

# Ball speed that counts as the ball having been disturbed during the stance window.
# Well under LAUNCH_SPEED: the point is to catch a robot creeping into the ball early,
# not to wait until it has actually kicked it.
DISTURB_SPEED = 0.15

_ROBOT_CFG = SceneEntityCfg(ROBOT)
_BALL_CFG = SceneEntityCfg(BALL)


##
# Scene readings.
##


def _site_index(robot: Entity, name: str) -> int:
  """Index of a site among the robot's sites.

  Looked up by name rather than through a SceneEntityCfg, whose site_ids are only
  resolved for terms that take the cfg in their params. The helpers here are called from
  the phase tracker and the command term, neither of which has params.
  """
  return robot.site_names.index(name)


def ball_pos_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball position in world coordinates, shaped (num_envs, 3)."""
  ball: Entity = env.scene[BALL]
  return ball.data.root_link_pos_w


def ball_vel_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball linear velocity in world axes, shaped (num_envs, 3)."""
  ball: Entity = env.scene[BALL]
  return ball.data.root_link_lin_vel_w


def ball_speed_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Horizontal ball speed, shaped (num_envs,).

  Horizontal rather than full 3D throughout. A kick that lofts the ball is fine, but the
  command is a ground plane heading and speed, and vertical motion should neither help
  nor hurt it.
  """
  return torch.norm(ball_vel_w(env)[:, :2], dim=-1)


def root_pos_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Robot root position in world coordinates, shaped (num_envs, 3)."""
  robot: Entity = env.scene[ROBOT]
  return robot.data.root_link_pos_w


def heading_quat(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The robot's yaw only orientation, shaped (num_envs, 4).

  Yaw only, never the full root orientation. The torso rolls and pitches hard through a
  kick, and a vector expressed in the full base frame swings around with it, so the ball
  would appear to move when only the robot had leaned. The heading frame turns with the
  robot and stays level, which is the frame the goal actually lives in.
  """
  robot: Entity = env.scene[ROBOT]
  return yaw_quat(robot.data.root_link_quat_w)


def to_heading_frame(env: ManagerBasedRlEnv, vec_w: torch.Tensor) -> torch.Tensor:
  """Rotate a world frame vector into the robot's heading frame."""
  return quat_apply_inverse(heading_quat(env), vec_w)


def kick_foot_pos_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """World position of the kicking foot's site, shaped (num_envs, 3)."""
  robot: Entity = env.scene[ROBOT]
  return robot.data.site_pos_w[:, _site_index(robot, KICK_SITE)]


def ball_surface_gap(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Distance from the kicking foot to the ball's surface, shaped (num_envs,).

  The foot site to ball centre distance minus the ball's radius, floored at zero. The raw
  centre distance can never reach zero, so a kernel built on it peaks at a value the foot
  can never attain and spends most of its range on distances that all mean touching.
  Subtracting the radius puts the peak where contact actually happens.
  """
  gap = torch.norm(ball_pos_w(env) - kick_foot_pos_w(env), dim=-1) - BALL_RADIUS
  return torch.clamp(gap, min=0.0)


def touching_ball(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Whether the kicking foot is in contact with the ball, shaped (num_envs,) bool."""
  sensor: ContactSensor = env.scene[FOOT_BALL_SENSOR]
  found = sensor.data.found
  assert found is not None
  return (found > 0).any(dim=-1)


##
# Episode phase.
##

# The env has no slot for this, so the tracker is attached under a namespaced attribute
# and reached with getattr and setattr, so it does not read as something the env class is
# expected to declare.
_PHASE_ATTR = "_kick_phase"


class KickPhase:
  """Per env episode state, refreshed at most once per environment step.

  Terminations and rewards are computed before the command manager runs, so this cannot
  live inside the command term without every reward reading a one step stale phase. It
  refreshes lazily on first access within a step, guarded by the env's own step counter,
  which makes it correct whichever manager touches it first.
  """

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    n, device = env.num_envs, env.device
    self._step = -1

    # Live, recomputed every step.
    self.touching = torch.zeros(n, dtype=torch.bool, device=device)

    # Latched for the rest of the episode once they happen.
    self.touched = torch.zeros(n, dtype=torch.bool, device=device)
    self.launched = torch.zeros(n, dtype=torch.bool, device=device)

    # The ball's horizontal velocity at the fastest instant so far, and that speed. This
    # is what the kick is scored on. Taking the fastest instant rather than the first one
    # means an early nudge is not locked in as the answer: a real kick later in the
    # episode simply replaces it.
    self.launch_vel_w = torch.zeros(n, 2, device=device)
    self.max_speed = torch.zeros(n, device=device)

    # Where the robot stood when the episode began. The stay put reward is measured
    # against this rather than against the env origin, because the reset scatters the
    # robot a little and the task is to stay where it started, not to return to a fixed
    # point.
    self.home_pos_w = torch.zeros(n, 2, device=device)

  def reset(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    """Clear the state for envs that just restarted."""
    for flag in (self.touching, self.touched, self.launched):
      flag[env_ids] = False
    self.launch_vel_w[env_ids] = 0.0
    self.max_speed[env_ids] = 0.0
    self.home_pos_w[env_ids] = root_pos_w(env)[env_ids, :2]
    # A refresh is owed: the scene moved under us, so the cached step is stale.
    self._step = -1

  def refresh(self, env: ManagerBasedRlEnv) -> None:
    if self._step == env.common_step_counter:
      return
    self._step = env.common_step_counter

    self.touching = touching_ball(env)
    self.touched = self.touched | self.touching

    # Gated on the kicking foot having touched the ball, and updated after that flag, so
    # a strike registers on the same step it happens. Without the gate anything that
    # moved the ball would be recorded as the launch: the other foot, or the robot
    # falling onto it. A later touch by the kicking foot would then find max_speed
    # already high and be paid in full for a kick it did not make.
    speed = ball_speed_xy(env)
    faster = self.touched & (speed > self.max_speed)
    if bool(faster.any()):
      self.launch_vel_w[faster] = ball_vel_w(env)[faster][:, :2]
      self.max_speed = torch.where(faster, speed, self.max_speed)

    # max_speed can only be nonzero once touched, so this needs no further gating.
    self.launched = self.launched | (self.max_speed >= LAUNCH_SPEED)


def phase(env: ManagerBasedRlEnv) -> KickPhase:
  """The env's phase tracker, created on first use and refreshed once per step."""
  tracker = getattr(env, _PHASE_ATTR, None)
  if tracker is None:
    tracker = KickPhase(env)
    setattr(env, _PHASE_ATTR, tracker)
  tracker.refresh(env)
  return tracker


def stance_progress(env: ManagerBasedRlEnv) -> torch.Tensor:
  """How far through the stance window the episode is, shaped (num_envs,).

  Zero at reset, one when the kick window opens and thereafter. This is what makes the
  gate legible to the policy. Without it the kick terms would switch on at a moment the
  observation gives no way to anticipate, which is a non stationary reward rather than a
  task with a phase.
  """
  elapsed = env.episode_length_buf.float() * env.step_dt
  return torch.clamp(elapsed / STANCE_TIME_S, max=1.0)


def kick_window_open(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Whether the robot is allowed to go for the ball, shaped (num_envs,) bool."""
  return stance_progress(env) >= 1.0


##
# Command.
##


class KickCommand(CommandTerm):
  """The velocity the ball should be travelling at once it has been struck.

  Sampled once per episode as a speed and a heading offset relative to whichever way the
  robot happens to be facing at reset, then frozen in world coordinates. World rather
  than robot relative is the whole point: the goal is a fact about the ball, and stored
  relative to the live heading the robot could satisfy a badly aimed kick by simply
  turning to face wherever the ball went.

  What the policy sees is that same world vector rotated back into its current heading
  frame, so the observation reads as "send it this way relative to where you are facing
  now" and stays correct even if the robot turns while kicking.
  """

  cfg: KickCommandCfg

  def __init__(self, cfg: KickCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self._target_vel_w = torch.zeros(self.num_envs, 2, device=self.device)
    self._command = torch.zeros(self.num_envs, 2, device=self.device)
    for name in ("vel_error", "speed_achieved", "heading_error", "launch_rate"):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """The goal in the robot's heading frame, shaped (num_envs, 2). What the policy sees."""
    return self._command

  @property
  def target_vel_w(self) -> torch.Tensor:
    """The goal in world coordinates, shaped (num_envs, 2). What the reward scores."""
    return self._target_vel_w

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Read after the reset events have run and forced a forward pass, so this is the
    # robot's new heading rather than the one it had last episode. See the ordering note
    # in reset_ball_near_foot.
    robot: Entity = self._env.scene[ROBOT]
    quat = robot.data.root_link_quat_w[env_ids]
    yaw = torch.atan2(
      2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
      1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
    )

    n = len(env_ids)
    speed = sample_uniform(*self.cfg.ranges.speed, n, self.device)
    offset = sample_uniform(*self.cfg.ranges.heading, n, self.device)
    direction = yaw + offset

    self._target_vel_w[env_ids, 0] = speed * torch.cos(direction)
    self._target_vel_w[env_ids, 1] = speed * torch.sin(direction)

  def _update_command(self) -> None:
    flat = torch.zeros(self.num_envs, 3, device=self.device)
    flat[:, :2] = self._target_vel_w
    self._command = to_heading_frame(self._env, flat)[:, :2]

  def _update_metrics(self) -> None:
    p = phase(self._env)

    # Honest whether or not the ball ever moved: an env that never launched scores an
    # error equal to the full commanded speed, because that is how far a stationary ball
    # is from the goal.
    self.metrics["vel_error"] = torch.norm(p.launch_vel_w - self._target_vel_w, dim=-1)
    self.metrics["speed_achieved"] = p.max_speed
    self.metrics["launch_rate"] = p.launched.float()

    # Aim, which only means anything for a ball that actually went somewhere. Zero for
    # the rest, so read it next to launch_rate rather than on its own.
    target_angle = torch.atan2(self._target_vel_w[:, 1], self._target_vel_w[:, 0])
    launch_angle = torch.atan2(p.launch_vel_w[:, 1], p.launch_vel_w[:, 0])
    error = torch.abs(wrap_to_pi(launch_angle - target_angle))
    self.metrics["heading_error"] = torch.where(
      p.launched, error, torch.zeros_like(error)
    )

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw the commanded launch velocity at the ball, and what the ball is doing."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    ball = ball_pos_w(self._env).cpu().numpy()
    goal = self._target_vel_w.cpu().numpy()
    actual = ball_vel_w(self._env).cpu().numpy()
    scale = self.cfg.arrow_scale

    for i in env_indices:
      start = ball[i]
      # The goal, in green, drawn flat along the ground plane.
      goal_w = np.array([goal[i, 0], goal[i, 1], 0.0])
      visualizer.add_arrow(
        start, start + goal_w * scale, (0.1, 0.8, 0.3, 0.9), width=0.012
      )
      # What the ball is actually doing, in blue.
      visualizer.add_arrow(
        start, start + actual[i] * scale, (0.2, 0.4, 0.95, 0.9), width=0.012
      )


@dataclass(kw_only=True)
class KickCommandCfg(CommandTermCfg):
  @dataclass
  class Ranges:
    speed: tuple[float, float]
    """How fast the ball should leave the foot, in m/s."""
    heading: tuple[float, float]
    """Where it should go, in radians relative to the robot's heading at reset."""

  ranges: Ranges
  arrow_scale: float = 0.25

  def build(self, env: ManagerBasedRlEnv) -> KickCommand:
    return KickCommand(self, env)


##
# Observations.
##


def ball_pos_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball position seen from the robot, in its heading frame, shaped (num_envs, 3).

  Relative, so it reads identically in every env and carries nothing about where in the
  world this particular robot happens to be standing.
  """
  return to_heading_frame(env, ball_pos_w(env) - root_pos_w(env))


def ball_vel_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball velocity in the robot's heading frame, shaped (num_envs, 3)."""
  return to_heading_frame(env, ball_vel_w(env))


def kick_foot_pos_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Kicking foot position seen from the robot, shaped (num_envs, 3).

  Derivable from the joint angles, so this is a convenience for the network rather than
  extra information, which is why it sits in the actor group rather than the critic's.
  """
  return to_heading_frame(env, kick_foot_pos_w(env) - root_pos_w(env))


def stance_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Stance window progress, shaped (num_envs, 1)."""
  return stance_progress(env).unsqueeze(-1)


def ball_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Whether the ball has been touched yet this episode, shaped (num_envs, 1).

  A foot mounted contact switch is something a real G1 can have, so this is an ordinary
  observation rather than a privileged one. It tells the policy which side of the kick it
  is on, which is what separates "go and get the ball" from "recover your balance".
  """
  return phase(env).touched.float().unsqueeze(-1)


def launch_velocity_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The latched launch velocity in the heading frame, shaped (num_envs, 3).

  Privileged. Every step after a kick is paid on this rather than on anything currently
  visible, so a critic without it is valuing a state it cannot see. Zero until something
  has been struck.
  """
  flat = torch.zeros(env.num_envs, 3, device=env.device)
  flat[:, :2] = phase(env).launch_vel_w
  return to_heading_frame(env, flat)


##
# Rewards.
##


def approach_ball(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
  """Dense pull of the kicking foot toward the ball, before contact and after the stance.

  A positive kernel, never a difference of distances. A difference telescopes over the
  episode to start minus end, so any rollout where the foot ends up away from the ball
  scores net negative however well it kicked, which teaches the policy to end episodes
  rather than to try.

  It stops paying the moment the ball has been touched. Left running it would reward
  keeping a foot parked against the ball, which is the cheapest way to hold a contact and
  the exact opposite of a kick.
  """
  score = torch.exp(-torch.square(ball_surface_gap(env)) / std**2)
  active = kick_window_open(env) & ~phase(env).touched
  return torch.where(active, score, torch.zeros_like(score))


def ball_touched(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Latched bonus for having made contact at all, shaped (num_envs,).

  The middle rung of the ladder. Contact is the discrete event standing between a policy
  that can balance and one that can score the kick terms, and a one shot bonus for it
  would be worth a rounding error against a few hundred steps of standing reward. Paying
  it for the rest of the episode is what makes it worth finding.
  """
  return phase(env).touched.float()


def kick_quality(
  env: ManagerBasedRlEnv, std: float, command_name: str = COMMAND_NAME
) -> torch.Tensor:
  """How close the launch velocity came to the commanded one, shaped (num_envs,).

  Zero until the ball has been struck hard enough to count as launched, then latched at
  the value earned and paid every step to the end of the episode. Latched rather than
  measured live because a ball's speed bleeds off as it rolls: a live kernel would peak
  some time after the kick, when the ball happened to decelerate through the commanded
  speed, and would end up scoring a struck ball on how the ground treated it afterwards.

  The kernel takes the mean over the two horizontal axes rather than the sum, so std
  reads as the per axis velocity error in m/s that the kernel is centred on.
  """
  p = phase(env)
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, KickCommand)
  error = torch.mean(torch.square(p.launch_vel_w - command.target_vel_w), dim=-1)
  score = torch.exp(-error / std**2)
  return torch.where(p.launched, score, torch.zeros_like(score))


def stay_put(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
  """Reward the robot for still being where it started, shaped (num_envs,).

  The ball is placed within reach, so walking to it is not part of the task, and a policy
  that shuffles into a comfortable range has solved a different problem from the one
  being posed. Measured on horizontal root displacement rather than on base velocity,
  which lets the robot sway and shift its weight through a kick without being charged for
  the transient.
  """
  displacement = root_pos_w(env)[:, :2] - phase(env).home_pos_w
  return torch.exp(-torch.sum(torch.square(displacement), dim=-1) / std**2)


def early_disturbance(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Cost for moving the ball before the stance window closes. Pair with a negative weight.

  The gate on the kick rewards makes an early kick worthless; this makes it costly. The
  two are not the same. Without a cost, a policy that lunges the moment the episode
  starts loses nothing by it and keeps rediscovering the lunge.
  """
  moving = ball_speed_xy(env) > DISTURB_SPEED
  return (moving & ~kick_window_open(env)).float()


def planted_foot_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str = FEET_GROUND_SENSOR,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Squared horizontal speed of whichever feet are on the ground. Negative weight.

  Named apart from the velocity task's feet_slip, which this module re-exports, because
  it is not the same term. That one gates on the commanded twist being large enough to
  justify moving; here there is no twist and a planted foot should never be sliding, so
  the gate is gone and the cost applies throughout.
  """
  robot: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  in_contact = (found > 0).float()
  foot_vel_xy = robot.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  return torch.sum(torch.sum(torch.square(foot_vel_xy), dim=-1) * in_contact, dim=-1)


##
# Events.
##


def reset_ball_near_foot(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  forward_range: tuple[float, float],
  lateral_range: tuple[float, float],
  foot_site: str = KICK_SITE,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
) -> None:
  """Place the ball in front of the kicking foot, in the robot's own heading frame.

  Relative to the foot, never at an absolute point. The robot resets with a random yaw
  and a scattered position, so a fixed world box would put the ball behind it as often as
  in front, and a ball the robot cannot reach is not a task but a guaranteed failure. The
  offsets are measured forward from the foot site, so they stay meaningful if the stance
  keyframe is ever retuned.

  The ranges have to clear the toe. The foot's forward collision geoms end about 0.09 m
  ahead of the site and the ball's rear surface sits one radius behind its centre, so a
  forward offset below roughly 0.20 m spawns the ball already in contact.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

  # The robot reset that ran before this wrote root pose and joint state into the entity
  # buffers; site poses come out of forward kinematics and are still whatever they were.
  # Flush and re-run, or the ball is placed relative to where the foot used to be, which
  # on the very first reset is the model's zero pose.
  env.scene.write_data_to_sim()
  env.sim.forward()

  robot: Entity = env.scene[ROBOT]
  ball: Entity = env.scene[ball_cfg.name]
  n = len(env_ids)

  foot_w = robot.data.site_pos_w[env_ids, _site_index(robot, foot_site)]
  offset_b = torch.zeros(n, 3, device=env.device)
  offset_b[:, 0] = sample_uniform(*forward_range, n, env.device)
  offset_b[:, 1] = sample_uniform(*lateral_range, n, env.device)
  offset_w = quat_apply(yaw_quat(robot.data.root_link_quat_w[env_ids]), offset_b)

  root_state = torch.zeros(n, 13, device=env.device)
  root_state[:, 0:3] = foot_w + offset_w
  root_state[:, 2] = BALL_RADIUS  # Resting on the ground, whatever the foot is doing.
  root_state[:, 3] = 1.0  # Identity quaternion.
  ball.write_root_state_to_sim(root_state, env_ids=env_ids)


def reset_kick_phase(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Clear the episode state for envs that just restarted.

  Registered last among the reset events, so the robot and the ball are already in their
  new places when the stay put anchor is read off them.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  tracker = getattr(env, _PHASE_ATTR, None)
  if tracker is None:
    tracker = KickPhase(env)
    setattr(env, _PHASE_ATTR, tracker)
  tracker.reset(env, env_ids)


##
# Terminations.
##


def ball_out_of_range(env: ManagerBasedRlEnv, distance: float = 8.0) -> torch.Tensor:
  """The ball has gone far enough that nothing more can happen this episode.

  A time out, not a failure. The kick is already scored and latched by the time this
  fires, and terminating it as a failure would bootstrap zero onto the end of the best
  rollouts the policy ever produces, teaching it that kicking the ball a long way is bad.
  """
  offset = ball_pos_w(env)[:, :2] - root_pos_w(env)[:, :2]
  return torch.norm(offset, dim=-1) > distance


##
# Curriculum helper.
##


def ramp(final_weight: float, stage_steps: int) -> list[dict]:
  """Stage list for the velocity task's reward_weight curriculum: off, half, full.

  Standing is learned first because for the first stage it is the only thing that pays.
  The half step in the middle is not decoration: turning a large kick weight on in one go
  hands the policy a reward it has no idea how to earn, and the cheapest way to discover
  it is to throw itself at the ball, which undoes the balance it just learned.
  """
  return [
    {"step": 0, "weight": 0.0},
    {"step": stage_steps, "weight": 0.5 * final_weight},
    {"step": 2 * stage_steps, "weight": final_weight},
  ]
