"""Analytical diffdrive experts: drive and turn.

These are the handwritten counterparts of the two RL skills trained from
diffdrive_env_cfg.py. They implement the same two behaviors as closed-form
feedback controllers so the bridging experiments can be run with experts whose
design is trivial and whose competence is not in question, leaving the bridge as
the only thing under test.

Prefer these analytical experts: RL skills are discouraged for this experiment,
they buy nothing here and only add a checkpoint to manage.

Skills:
- drive: tracks a commanded forward speed with zero yaw rate. It is a plain
  wheel-velocity servo and holds no state.
- turn: arcs to a given angle: it drives a tight, fixed-radius arc, rotating
  toward a target heading and easing off as it arrives. It is stateful, because the
  observation carries no absolute orientation, only a yaw rate; the skill integrates
  that yaw rate from the moment it takes over to know how far it has turned. The
  target is therefore relative to the heading the robot had when the turn began.

Why an arc (and why it fails at speed). The arc's lateral acceleration is
v**2 / R: at the low speed turn was trained at it is negligible, but handed the
robot at drive's cruise it is large enough to roll the tall, narrow chassis over its
inner wheels. turn does not brake the speed it inherits; it asks for its low arc
speed and lets the acceleration-limited wheels get there in their own time, so
a robot handed at cruise keeps barreling forward as it starts to rotate and
goes over. Slowing down first is the bridge's job, not the skill's.

Both experts command wheel velocities (the action space is joint velocity,
realized by torque-limited velocity servos driven by an acceleration-limited
command; see diffdrive_env_cfg.py). The acceleration limit is deliberate and
load-bearing: the wheels cannot change speed instantly, so a controller cannot
reject momentum it did not build up, which is exactly why a naive hand-off tips and
a bridge is worth having. The failure is emergent from the physics, not hand-coded
into the controller.

The controllers read the robot state straight out of the flattened observation
vector, so this file is coupled to the observation layout declared in
diffdrive_env_cfg.py. That layout is mirrored in the index constants below; if
the observation terms there change, these must change with them.
"""

from __future__ import annotations

import math

import torch

from mjlab.envs import VecEnvObs
from mjlab.tasks.skills.experiments.diffdrive import DRIVE_SPEED, TURN_ANGLE, TURN_SPEED
from mjlab.tasks.skills.skill import Skill

##
# Chassis geometry. Mirrors `diffdrive_env_cfg.py` / the diffdrive asset XML.
##

# Wheel radius [m]
WHEEL_RADIUS = 0.06

# Lateral offset of each wheel from the chassis centerline [m]. Must match the
# narrowed track this experiment builds in `diffdrive_env_cfg._tall_diffdrive_spec`.
HALF_TRACK = 0.075

"""
Control period [s]: decimation (4) * sim timestep (0.005) = 50 Hz.

Used by turn to integrate its yaw rate into an accumulated heading. Kept as a
module constant so it stays next to the geometry it belongs with; override per
skill via the factory if the env's timing changes.
"""
CONTROL_DT = 0.02

##
# Observation layout. Mirrors the `obs_terms` order in `diffdrive_env_cfg.py`:
#   base_lin_vel [0:3]  (root_link_lin_vel_b; x is forward)
#   base_ang_vel [3:6]  (root_link_ang_vel_b; z is yaw rate)
#   wheel_vel    [6:8]  (left, right; preserve_order)
#   last_action  [8:10]
#   command      [10:12]
##

# `drive` is open-loop (a constant wheel-velocity command). `turn` reads the forward
# speed to build its arc around the speed it is actually carrying, and integrates the
# yaw rate to track how far it has turned.
_LIN_VEL_X = 0
_ANG_VEL_Z = 5


def _state(obs: VecEnvObs) -> torch.Tensor:
  """The flattened robot state to control on.

  Prefer the uncorrupted critic group when present (cleaner for the turn skill's
  yaw integrator).
  """
  group = obs["actor"]
  assert isinstance(group, torch.Tensor)
  return group


def _twist_to_wheel_speeds(
  lin_vel_x: torch.Tensor | float, yaw_rate: torch.Tensor | float
) -> tuple[torch.Tensor | float, torch.Tensor | float]:
  """Chassis (forward speed, yaw rate) -> (left, right) wheel angular speeds.

  Inverse of the differential-drive kinematics the env's slip reward assumes:
  v = r/2 (w_l + w_r) and yaw = r/(2b) (w_r - w_l), so a positive yaw
  rate turns the robot left (right wheel faster than left).
  """
  wheel_l = (lin_vel_x - yaw_rate * HALF_TRACK) / WHEEL_RADIUS
  wheel_r = (lin_vel_x + yaw_rate * HALF_TRACK) / WHEEL_RADIUS
  return wheel_l, wheel_r


class DriveSkill(Skill):
  """Drive straight at a fixed forward speed, zero yaw rate. Stateless.

  Emits a constant pair of wheel velocity targets. The robot's actuation ramps to
  them under its acceleration limit, so nothing here has to shape the approach,
  the skill just names the cruise it wants.
  """

  name = "drive"

  def __init__(self, speed: float) -> None:
    self.speed = speed

  def act(self, obs: VecEnvObs, active: torch.Tensor) -> torch.Tensor:
    del active  # Stateless: same action for every env, caller selects rows.
    state = _state(obs)
    wheel_l, wheel_r = _twist_to_wheel_speeds(self.speed, 0.0)
    n = state.shape[0]
    left = torch.full((n,), float(wheel_l), device=state.device)
    right = torch.full((n,), float(wheel_r), device=state.device)
    return torch.stack((left, right), dim=-1)


class TurnSkill(Skill):
  """Turn to a target heading, the way a skill trained only from rest would.

  Stateful: the observation carries no absolute orientation, so the skill
  accumulates the measured yaw rate from the step it takes over (the rising edge
  of `active`) to know how far it has turned, and a saturating law aims for a
  target yaw rate that eases off as the heading is reached.

  Crucially it does not actively brake away the speed it inherits. It asks for a
  low arc speed and a yaw rate and lets the acceleration-limited wheels get there in
  their own time; from rest that is a gentle, tight arc, but handed drive's cruise
  the wheels can only spin down gradually, so the robot keeps barreling forward while
  it starts to rotate. The arc's lateral acceleration then scales with the inherited
  speed, and past a threshold it rolls the tall, narrow chassis over its inner wheels
  and the robot tips. The bounded acceleration is what carries the momentum through
  the hand-off; the skill itself just names the twist it wants.
  """

  name = "turn"

  def __init__(
    self,
    target_angle: float = math.pi / 2,
    turn_rate: float = 2.0,
    creep_speed: float = 0.3,
    heading_gain: float = 3.0,
    ease_accel: float = 0.6,
    dt: float = CONTROL_DT,
  ) -> None:
    self.target_angle = target_angle
    self.turn_rate = turn_rate
    self.creep_speed = creep_speed
    self.heading_gain = heading_gain
    # How fast the forward-speed target drifts toward creep_speed [m/s^2]. Kept
    # gentle on purpose: handed a fast robot, turn arcs around the speed it is
    # actually carrying for many steps before it has shed much, so the arc is
    # violent from the first step and the robot tips. This is what makes the naive
    # hand-off fail; a stronger ease would brake the momentum away before it could.
    self.ease_step = ease_accel * dt
    self.dt = dt
    # Accumulated heading and previous active mask, per env. Allocated lazily on
    # the first act/reset, once the batch size and device are known
    self._heading: torch.Tensor | None = None
    self._active_prev: torch.Tensor | None = None

  def _ensure(self, ref: torch.Tensor) -> None:
    if self._heading is None:
      self._heading = torch.zeros(ref.shape[0], device=ref.device)
    if self._active_prev is None:
      self._active_prev = torch.zeros(ref.shape[0], dtype=torch.bool, device=ref.device)

  def act(self, obs: VecEnvObs, active: torch.Tensor) -> torch.Tensor:
    state = _state(obs)
    self._ensure(state)
    assert self._heading is not None and self._active_prev is not None

    # Restart the heading count for envs that just took control this step, so the
    # target is measured relative to where the turn began
    newly_active = active & ~self._active_prev
    self._heading = torch.where(
      newly_active, torch.zeros_like(self._heading), self._heading
    )

    # Integrate the measured yaw rate only where this skill is driving
    yaw_rate = state[:, _ANG_VEL_Z]
    self._heading = torch.where(
      active, self._heading + yaw_rate * self.dt, self._heading
    )

    # Saturating proportional heading law -> desired yaw rate
    remaining = self.target_angle - self._heading
    yaw_cmd = torch.clamp(
      self.heading_gain * remaining, -self.turn_rate, self.turn_rate
    )

    # Build the arc around the forward speed the robot is actually carrying,
    # nudged only gently toward the low creep speed. Centering the wheel targets on
    # the current speed (rather than commanding the low creep speed outright) is what
    # lets the yaw differential take effect immediately: the downstream rate limiter
    # would otherwise ramp both wheels down together and swallow the differential, so
    # the robot would just brake in a straight line instead of tipping. Handed drive's
    # cruise, this arcs hard at that cruise -> lateral acceleration rolls it over.
    v_cur = state[:, _LIN_VEL_X]
    v_target = v_cur + torch.clamp(
      self.creep_speed - v_cur, -self.ease_step, self.ease_step
    )
    wheel_l = (v_target - yaw_cmd * HALF_TRACK) / WHEEL_RADIUS
    wheel_r = (v_target + yaw_cmd * HALF_TRACK) / WHEEL_RADIUS
    actions = torch.stack((wheel_l, wheel_r), dim=-1)

    self._active_prev = active.clone()
    return actions

  def reset(self, mask: torch.Tensor) -> None:
    self._ensure(mask)
    assert self._heading is not None and self._active_prev is not None
    self._heading = torch.where(mask, torch.zeros_like(self._heading), self._heading)
    self._active_prev = self._active_prev & ~mask


def analytical_drive(speed: float = DRIVE_SPEED) -> DriveSkill:
  """The drive expert, cruising at a given speed [m/s].

  The default is high enough that handing over to turn at this speed tips the tall
  chassis, which is the failure the bridge has to remove.
  """
  return DriveSkill(speed=speed)


def analytical_turn(
  target_angle: float = TURN_ANGLE, forward_speed: float = TURN_SPEED
) -> TurnSkill:
  """The turn expert, arcing to a target_angle [rad] turn (default 90 deg).

  It creeps forward at forward_speed [m/s] while it rotates, so the robot describes
  a tight arc rather than pivoting in place. Kept low so the arc is safe at this
  speed but violent if entered while still carrying drive's momentum.
  """
  return TurnSkill(target_angle=target_angle, creep_speed=forward_speed)
