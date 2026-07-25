"""Analytical diffdrive experts: `drive` and `turn`.

These are the hand-written counterparts of the two RL skills trained from
`carlike_env_cfg.py`. They implement the *same* two behaviors as closed-form
feedback controllers so the bridging experiments can be run with experts whose
design is trivial and whose competence is not in question, leaving the bridge as
the only thing under test.

Design, following the problem statement:

- ``drive`` tracks a commanded forward speed with zero yaw rate. It is a plain
  wheel-velocity servo and holds no state.
- ``turn`` *realizes a given angle*: it rotates toward a target heading and holds
  there. It is stateful, because the observation carries no absolute orientation,
  only a yaw rate; the skill integrates that yaw rate from the moment it takes
  over to know how far it has turned. The target is therefore relative to the
  heading the robot had when the turn began.

Why a target angle rather than a fixed open-loop 90 deg spin: the experiment
compares several bridge architectures, so it needs a *graded* transition-quality
signal, not a binary "did it spin". With a target heading, the momentum a naive
hand-off fails to shed shows up as a continuous heading error (the residual yaw
rate carried through the switch overshoots the target), while a genuine spin-out
is just the saturating tail of that same metric. That error is measured
externally, against ground-truth heading, around the switch; nothing here reports
it, in keeping with the `Skill` contract (a skill never self-reports progress).

Both experts command wheel *velocities* (the action space is joint velocity,
realized by torque-limited velocity servos driven by an acceleration-limited
command; see `carlike_env_cfg.py`). The acceleration limit is deliberate and
load-bearing: the wheels cannot change speed instantly, so a controller cannot
reject momentum it did not build up, which is exactly why a naive hand-off
degrades and a bridge is worth having. The failure is emergent from the physics,
not hand-coded into the controller.

The controllers read the robot state straight out of the flattened observation
vector, so this file is coupled to the observation layout declared in
`carlike_env_cfg.py`. That layout is mirrored in the index constants below; if
the observation terms there change, these must change with them.
"""

from __future__ import annotations

import math

import torch

from mjlab.envs import VecEnvObs
from mjlab.tasks.skills.skill import Skill

##
# Chassis geometry. Mirrors `carlike_env_cfg.py` / the diffdrive asset XML.
##

# Wheel radius [m]
WHEEL_RADIUS = 0.06

# Lateral offset of each wheel from the chassis centerline [m]
HALF_TRACK = 0.11

"""
Control period [s]: decimation (4) * sim timestep (0.005) = 50 Hz.

Used by `turn` to integrate its yaw rate into an accumulated heading. Kept as a
module constant so it stays next to the geometry it belongs with; override per
skill via the factory if the env's timing changes.
"""
CONTROL_DT = 0.02

##
# Observation layout. Mirrors the `obs_terms` order in `carlike_env_cfg.py`:
#   base_lin_vel [0:3]  (root_link_lin_vel_b; x is forward)
#   base_ang_vel [3:6]  (root_link_ang_vel_b; z is yaw rate)
#   wheel_vel    [6:8]  (left, right; preserve_order)
#   last_action  [8:10]
#   command      [10:12]
##

# Only the yaw rate is read now: `drive` is open-loop (a constant wheel-velocity
# command) and `turn` integrates the yaw rate to track how far it has turned.
_ANG_VEL_Z = 5


def _state(obs: VecEnvObs) -> torch.Tensor:
  """The flattened robot state to control on.

  Prefer the uncorrupted critic group when present (cleaner for the turn skill's
  yaw integrator), falling back to the actor group a deployed policy would see.
  """
  group = obs["actor"]
  assert isinstance(group, torch.Tensor)
  return group


def _twist_to_wheel_speeds(
  lin_vel_x: torch.Tensor | float, yaw_rate: torch.Tensor | float
) -> tuple[torch.Tensor | float, torch.Tensor | float]:
  """Chassis (forward speed, yaw rate) -> (left, right) wheel angular speeds.

  Inverse of the differential-drive kinematics the env's slip reward assumes:
  ``v = r/2 (w_l + w_r)`` and ``yaw = r/(2b) (w_r - w_l)``, so a positive yaw
  rate turns the robot left (right wheel faster than left).
  """
  wheel_l = (lin_vel_x - yaw_rate * HALF_TRACK) / WHEEL_RADIUS
  wheel_r = (lin_vel_x + yaw_rate * HALF_TRACK) / WHEEL_RADIUS
  return wheel_l, wheel_r


class DriveSkill(Skill):
  """Drive straight at a fixed forward speed, zero yaw rate. Stateless.

  Emits a constant pair of wheel velocity targets. The robot's actuation ramps to
  them under its acceleration limit, so nothing here has to shape the approach --
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

  Crucially it does *not* actively brake away the speed it inherits. It asks for a
  low forward speed and a yaw rate and lets the acceleration-limited wheels get
  there in their own time -- from rest that is a gentle arc, but handed drive's
  cruise the wheels can only spin down gradually, so the robot keeps barreling
  forward while it tries to rotate. Rotating a body that still carries forward
  momentum throws that momentum sideways across the wheels, which is the skid (and,
  once the reaction unloads a wheel onto the frictionless caster, the spin-out)
  that a momentum-naive turn is supposed to suffer. The bounded acceleration is
  what carries the momentum through the hand-off; the skill itself just names the
  twist it wants.
  """

  name = "turn"

  def __init__(
    self,
    target_angle: float = math.pi / 2,
    turn_rate: float = 1.5,
    creep_speed: float = 0.0,
    heading_gain: float = 3.0,
    dt: float = CONTROL_DT,
  ) -> None:
    self.target_angle = target_angle
    self.turn_rate = turn_rate
    self.creep_speed = creep_speed
    self.heading_gain = heading_gain
    self.dt = dt
    # Accumulated heading and previous active mask, per env. Allocated lazily on
    # the first act/reset, once the batch size and device are known.
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
    # target is measured relative to where the turn began.
    newly_active = active & ~self._active_prev
    self._heading = torch.where(
      newly_active, torch.zeros_like(self._heading), self._heading
    )

    # Integrate the measured yaw rate only where this skill is driving.
    yaw_rate = state[:, _ANG_VEL_Z]
    self._heading = torch.where(
      active, self._heading + yaw_rate * self.dt, self._heading
    )

    # Saturating proportional heading law -> desired yaw rate, which with the
    # creep speed makes a twist. The twist becomes wheel velocity targets; the
    # acceleration limit downstream means a robot handed at speed sheds its
    # forward momentum only gradually while it rotates -- the momentum-naive arc.
    remaining = self.target_angle - self._heading
    yaw_cmd = torch.clamp(
      self.heading_gain * remaining, -self.turn_rate, self.turn_rate
    )
    # Twist -> wheel velocity targets (the tensor form of _twist_to_wheel_speeds).
    wheel_l = (self.creep_speed - yaw_cmd * HALF_TRACK) / WHEEL_RADIUS
    wheel_r = (self.creep_speed + yaw_cmd * HALF_TRACK) / WHEEL_RADIUS
    actions = torch.stack((wheel_l, wheel_r), dim=-1)

    self._active_prev = active.clone()
    return actions

  def reset(self, mask: torch.Tensor) -> None:
    self._ensure(mask)
    assert self._heading is not None and self._active_prev is not None
    self._heading = torch.where(mask, torch.zeros_like(self._heading), self._heading)
    self._active_prev = self._active_prev & ~mask


def analytical_drive(speed: float = 1.0) -> DriveSkill:
  """The drive expert, cruising at a given speed [m/s] (default mid-command)."""
  return DriveSkill(speed=speed)


def analytical_turn(
  target_angle: float = math.pi / 2, forward_speed: float = 0.1
) -> TurnSkill:
  """The turn expert, arcing to a target_angle [rad] turn (default 90 deg).

  It creeps forward at forward_speed [m/s] while it rotates, so the robot
  describes an arc rather than pivoting in place; set it to 0 for a pivot.
  """
  return TurnSkill(target_angle=target_angle, creep_speed=forward_speed)
