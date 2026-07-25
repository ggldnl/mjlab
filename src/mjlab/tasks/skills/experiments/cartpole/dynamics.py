"""Analytical cart-pole experts: spin_up and balance.

These are the hand-written counterparts of the two RL skills, implemented as
closed-form controllers so the bridging experiments can run with experts whose
design is trivial and whose competence is not in question. Better to avoid RL
when possible.

- spin_up is an energy-shaping swing-up: it drives the cart to pump the
  pole's mechanical energy toward the value it has standing upright, in phase
  with the swing. It shapes energy only: it has no model of the unstable
  equilibrium, so it brings the pole up and then sails straight through the top.
  It cannot balance.
- balance is the LQR for the pole linearized about upright. It holds the pole
  when it is already near vertical and slow; handed a pole that is down or
  swinging, its linear law reads a huge angle and drives the cart the wrong way.
  It cannot bring the pole up.

Both read the robot state straight out of the flattened observation vector, so
this file is coupled to the observation layout of the cartpole env (mirrored in
the index constants below). Both command a scalar cart effort in normalized units
[-1, 1].

The LQR gain is precomputed offline from the linearization; it is hardcoded
so the skill needs no runtime solver.
"""

from __future__ import annotations

import torch

from mjlab.envs import VecEnvObs
from mjlab.tasks.skills.skill import Skill

##
# Physical constants. Mirror `asset_zoo/robots/cartpole/xmls/cartpole.xml`.
##

# Cart mass [kg], from <geom name="cart" ... mass="1"/>
CART_MASS = 1.0

# Pole mass [kg], from the pole default <geom ... mass=".1"/>
POLE_MASS = 0.1

# Pole length [m], from the pole capsule fromto="0 0 0 0 0 1"
POLE_LENGTH = 1.0

GRAVITY = 9.81

# Distance from the hinge to the pole centre of mass [m]
_POLE_COM = POLE_LENGTH / 2.0

# Pole moment of inertia about the hinge [kg m^2] (uniform rod)
_POLE_INERTIA = POLE_MASS * POLE_LENGTH**2 / 3.0

# m g l: the pole's potential energy scale, and its energy standing upright
_POLE_MGL = POLE_MASS * GRAVITY * _POLE_COM

# Cart force per unit action [N]. The env's motor has gear=10 and ctrlrange +-1,
# so effort target (= action) maps to +-10 N of cart force
FORCE_SCALE = 10.0

##
# Observation layout. Mirrors the cartpole env's actor obs term order:
#   cart_pos  [0]    (slider position, joint_pos_rel)
#   pole_cos  [1]    (cos of the hinge angle; +1 upright)
#   pole_sin  [2]    (sin of the hinge angle)
#   cart_vel  [3]    (slider velocity)
#   pole_vel  [4]    (hinge angular velocity)
# The hinge angle is measured from upright: 0 is up, +-pi is hanging down, and a
# positive angle leans the pole toward +x.
##

_CART_POS = 0
_POLE_COS = 1
_POLE_SIN = 2
_CART_VEL = 3
_POLE_VEL = 4

# LQR gain in action units, i.e. action = -(K / FORCE_SCALE) . [x, xdot, th, thdot].
# K was solved from the upright linearization with Q = diag(1, 1, 10, 1), R = 0.5;
# the closed loop is stable (all eigenvalues in the left-half plane).
_BALANCE_GAIN = (0.14142, 0.30196, 3.54788, 0.90904)


def _state(obs: VecEnvObs) -> torch.Tensor:
  """The flattened robot state to control on (the actor observation group)."""
  group = obs["actor"]
  assert isinstance(group, torch.Tensor)
  return group


class SpinUpSkill(Skill):
  """
  Energy-shaping swing-up. Stateless; brings the pole up but cannot balance.

  The pole's mechanical energy about the hinge is

    E = 1/2 J thetadot^2 + m g l cos(theta)

  (theta from upright), and standing upright it is

    E_up = m g l

  Driving the cart with

    a = -k (E_up - E) sign(thetadot cos(theta))

  feeds energy in when the pole is short of upright and in the phase where
  the cart's motion does positive work on the swing. It shapes energy only,
  so near the top it stops pushing and the pole coasts through: there is
  no term that stabilizes the unstable equilibrium.
  """

  name = "spin_up"

  def __init__(
    self,
    energy_gain: float = 2.0,
    center_gain: float = 0.1,
    center_damping: float = 0.2,
  ) -> None:
    self.energy_gain = energy_gain
    # A light cart-centering term keeps the swing from walking off the rail; it
    # is deliberately weak so it never overrides the energy pumping.
    self.center_gain = center_gain
    self.center_damping = center_damping

  def act(self, obs: VecEnvObs, active: torch.Tensor) -> torch.Tensor:
    del active  # Stateless.
    state = _state(obs)
    cart_pos = state[:, _CART_POS]
    pole_cos = state[:, _POLE_COS]
    cart_vel = state[:, _CART_VEL]
    pole_vel = state[:, _POLE_VEL]

    energy = 0.5 * _POLE_INERTIA * pole_vel**2 + _POLE_MGL * pole_cos
    pump = -self.energy_gain * (_POLE_MGL - energy) * torch.sign(pole_vel * pole_cos)
    center = -self.center_gain * cart_pos - self.center_damping * cart_vel
    action = torch.clamp(pump + center, -1.0, 1.0)
    return action.unsqueeze(-1)


class BalanceSkill(Skill):
  """
  LQR balance about upright. Stateless; holds the pole but cannot bring it up.

    action = gain . [x, xdot, theta, thetadot]

  with the precomputed LQR gain, where

    theta = atan2(sin, cos)

  is the wrapped hinge angle from upright. Near the top this is the stabilizing
  feedback; far from it (pole swinging or hanging) the wrapped angle is large
  and the same linear law drives the cart the wrong way, which is exactly why
  the balancer cannot catch a pole that has not been brought up for it.
  """

  name = "balance"

  def __init__(self, gain: tuple[float, float, float, float] = _BALANCE_GAIN) -> None:
    self.gain = gain

  def act(self, obs: VecEnvObs, active: torch.Tensor) -> torch.Tensor:
    del active  # Stateless.
    state = _state(obs)
    cart_pos = state[:, _CART_POS]
    cart_vel = state[:, _CART_VEL]
    theta = torch.atan2(state[:, _POLE_SIN], state[:, _POLE_COS])
    pole_vel = state[:, _POLE_VEL]

    g_x, g_xd, g_th, g_thd = self.gain
    action = g_x * cart_pos + g_xd * cart_vel + g_th * theta + g_thd * pole_vel
    return torch.clamp(action, -1.0, 1.0).unsqueeze(-1)


def analytical_spin_up() -> SpinUpSkill:
  """The spin_up expert: energy-shaping swing-up that cannot balance."""
  return SpinUpSkill()


def analytical_balance() -> BalanceSkill:
  """The balance expert: LQR that holds upright but cannot swing up."""
  return BalanceSkill()
