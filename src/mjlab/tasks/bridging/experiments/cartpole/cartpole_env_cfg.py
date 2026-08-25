"""The cart-pole this experiment runs in: the shared one, with a lossy hinge.

Everything here exists to make one change, and the change is one number. The shared
cart-pole asset gives its hinge a damping of 2e-6, which is a frictionless pendulum in
all but name. This experiment raises it.

## Why

The bridging experiment needs a hand-over that can go wrong, and on a frictionless
cart-pole it cannot. `spin_up` is an energy-shaping controller: it drives the pole's
energy to exactly the energy of standing upright. "Exactly enough energy to reach the
top" and "arrives at the top with no speed left" are the same statement, so a pole that
`spin_up` has finished with will present itself upright and slow, which is precisely
`balance`'s catchable condition. Miss it, and with no losses the pole keeps that energy
forever and comes back around to offer it again. So a hand-over at the worst possible
moment costs nothing but time, `balance` waits it out at zero force, and a bridge has
nothing left to do.

Damping breaks that, and it breaks it asymmetrically, which is the point. `spin_up` has
an energy source: it pumps, so damping is a bill it can pay, and it pays it by settling
at a slightly lower energy and taking a little longer. `balance` outside its basin
applies zero force, so it is coasting, and a coasting pole cannot pay anything. Hand over
while the pole is still swinging and the energy `spin_up` spent seconds building bleeds
away before the pole gets to the top. It falls short, swings back down, loses more, comes
up lower still, and dies. `balance` cannot swing up, so the failure is permanent, and no
termination term or episode deadline is needed to make it stick.

It is also graded, which is what the experiment wants: hand over near the top and there
is barely any distance left to lose energy over, so it is still caught. Hand over at the
bottom (which is exactly what `CartpoleController`'s position rule does) and there is a
full half-swing to bleed out.

Note this is not an engineered failure so much as a removed idealization. Real hinges
have friction. The claim being made is "we stopped pretending the pole is lossless", not
"we modified the balancer until it broke", and in particular neither skill is touched.

## How much

The pole leaves the bottom on the orbit `spin_up` put it on and dissipates `damping *
integral of thetadot d(theta)` climbing to the top. On that orbit the integral has a
closed form, `sqrt(2 mgl / J) * 2 sqrt(2)`, about 15.3 for this pole. To fall short of
`balance`'s 0.35 rad basin it has to shed `mgl (1 - cos 0.35)`, about 0.0297 J. That puts
break-even near 2e-3, a thousand times the asset's value.

That derivation treats the pole as a fixed-pivot pendulum while the cart is in fact free
to move and takes a share of the energy, so it is a bracket rather than a prediction. The
value below was chosen by measuring both halves of the requirement instead: that
`spin_up` on its own still brings the pole into the basin, and that a hand-over at the
bottom of the swing no longer gets there.

## Where

In this file rather than in the shared asset, because `cartpole.xml` is also what
`Mjlab-Cartpole-Swingup`, `Mjlab-Cartpole-Balance` and the documentation tutorial are
built from, and those are the tasks the skills themselves come from. Editing a copy of
the spec follows what the diffdrive experiment already does when it narrows its own wheel
track to make the robot tip (see `_tall_diffdrive_spec`).
"""

from __future__ import annotations

import numpy as np

from mjlab.asset_zoo.robots.cartpole.cartpole_constants import (
  CARTPOLE_ARTICULATION,
  CARTPOLE_BALANCE_INIT,
  CARTPOLE_SWINGUP_INIT,
  get_spec,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.cartpole.cartpole_env_cfg import (
  cartpole_balance_env_cfg,
  cartpole_swingup_env_cfg,
)

# Hinge damping [N m s / rad] for this experiment's cart-pole. The asset ships 2e-6.
# See the module docstring for where this number comes from and what it buys.
HINGE_DAMPING = 5e-3

# The hinge whose damping is being raised. The asset sets it through the `pole` default
# class rather than on the joint, but MjSpec resolves the class onto the element, so
# writing the joint's own value overrides it.
HINGE_JOINT = "hinge_1"


def damped_cartpole_spec(hinge_damping: float = HINGE_DAMPING):
  """The shared cart-pole spec with a lossy hinge, as a fresh copy each call.

  `get_spec` re-reads the XML every time, so nothing here can leak into the shared
  tasks. Damping is a three-vector on a joint (a ball joint has three axes); a hinge
  uses the first entry and ignores the rest.
  """
  spec = get_spec()
  spec.joint(HINGE_JOINT).damping = np.array([hinge_damping, 0.0, 0.0])
  return spec


def damped_cartpole_robot_cfg(
  swing_up: bool = False, hinge_damping: float = HINGE_DAMPING
) -> EntityCfg:
  """The lossy cart-pole, used only by this experiment."""
  return EntityCfg(
    spec_fn=lambda: damped_cartpole_spec(hinge_damping),
    articulation=CARTPOLE_ARTICULATION,
    init_state=CARTPOLE_SWINGUP_INIT if swing_up else CARTPOLE_BALANCE_INIT,
  )


def damped_cartpole_env_cfg(
  swing_up: bool = False,
  play: bool = False,
  hinge_damping: float = HINGE_DAMPING,
) -> ManagerBasedRlEnvCfg:
  """The shared cart-pole task's env cfg with this experiment's lossy pole swapped in.

  Deliberately built by taking the shared config and replacing one entity rather than
  by restating it. The observation layout, action term, rewards and reset events must
  stay identical to the task the skills were designed against -- the damping is the only
  thing this experiment is allowed to change, and a forked copy of the config would make
  that impossible to see and easy to break.
  """
  cfg = (
    cartpole_swingup_env_cfg(play=play)
    if swing_up
    else cartpole_balance_env_cfg(play=play)
  )
  cfg.scene.entities["cartpole"] = damped_cartpole_robot_cfg(
    swing_up=swing_up, hinge_damping=hinge_damping
  )
  return cfg
