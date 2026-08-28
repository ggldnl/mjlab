"""Box entity, static or free-floating.

Two things come out of the same parametric spec, and which one you get is decided by
``mass``.

Left at None the box has no freejoint, so mjlab auto-wraps it as a *mocap* body: it
still participates in collisions (a rigid, immovable barrier) but is never moved by
physics, and its pose can be set per-environment at reset via
:func:`reset_root_state_uniform`. That is the obstacle a robot has to clear.

Given a mass the box gets a freejoint and an inertial, and becomes an ordinary free
body the robot can shove around. That is the object a robot has to push.
"""

from functools import partial

import mujoco

from mjlab.entity import EntityCfg

# Half-extents (m) of the default obstacle: 0.10 m deep (x), 1.20 m wide (y),
# 0.15 m tall (z). Wide enough that a forward-walking robot cannot trivially
# sidestep it, and low enough to be a reasonable first hurdle.
BOX_HALF_SIZE: tuple[float, float, float] = (0.05, 0.6, 0.075)

# Distance (m) ahead of the environment origin where the obstacle is placed by
# default. The robot resets near the origin facing +x, so it walks into the box.
BOX_INIT_X: float = 1.5

BOX_COLOR: tuple[float, float, float, float] = (0.85, 0.35, 0.2, 1.0)

# Sliding friction of the box against whatever it rests on. MuJoCo takes the
# elementwise maximum of the two geoms' friction unless one of them declares a higher
# priority, and neither the terrain nor the G1's collision geoms do, so a value below
# the terrain's own (1.0 by default) has no effect at the default priority. Raise it to
# make a box harder to shift without making it heavier; to lower it, raise `priority`
# too.
BOX_FRICTION: float = 1.0

# Contact priority of the box's collision geom. At 0 the box loses every friction
# argument it has, since the terrain's 1.0 is the maximum either way. At 1 the box's own
# friction wins against the terrain (priority 0) and ties with the G1's feet (priority 1,
# 0.6), where the maximum rule applies again. Raise it when the box has to be more
# slippery than the ground it sits on, which is what stops a tall crate toppling instead
# of sliding when it is pushed high up.
BOX_PRIORITY: int = 0


def get_box_spec(
  half_size: tuple[float, float, float] = BOX_HALF_SIZE,
  mass: float | None = None,
  friction: float = BOX_FRICTION,
  priority: int = BOX_PRIORITY,
  color: tuple[float, float, float, float] = BOX_COLOR,
) -> mujoco.MjSpec:
  """Return MjSpec describing a box.

  Args:
    half_size: Half-extents along x, y, z, in metres.
    mass: Leave None for a static (fixed-base) obstacle. Give a mass in kg for a
      free-floating box that physics can move.
    friction: Sliding friction of the collision geom.
    priority: Contact priority of the collision geom. Leave at 0 to let the other geom's
      friction win where it is higher; raise it to make ``friction`` authoritative.
    color: RGBA.
  """
  sx, sy, sz = half_size

  if mass is None:
    body = ""
  else:
    # Solid box of uniform density, half-extents (a, b, c):
    # I = m/3 * (b^2 + c^2, a^2 + c^2, a^2 + b^2).
    ixx = mass / 3.0 * (sy**2 + sz**2)
    iyy = mass / 3.0 * (sx**2 + sz**2)
    izz = mass / 3.0 * (sx**2 + sy**2)
    body = f"""
      <freejoint/>
      <site name="root_site"/>
      <inertial pos="0 0 0" mass="{mass}" diaginertia="{ixx} {iyy} {izz}"/>
"""

  rgba = " ".join(str(c) for c in color)
  xml = f"""<mujoco>
  <worldbody>
    <body name="box" pos="0 0 {sz}">{body}
      <geom name="box_collision" type="box" size="{sx} {sy} {sz}"
            condim="3" friction="{friction} 0.005 0.0001" priority="{priority}"
            rgba="{rgba}"/>
    </body>
  </worldbody>
</mujoco>
"""
  return mujoco.MjSpec.from_string(xml)


def get_box_cfg(
  half_size: tuple[float, float, float] = BOX_HALF_SIZE,
  init_x: float = BOX_INIT_X,
  mass: float | None = None,
  friction: float = BOX_FRICTION,
  priority: int = BOX_PRIORITY,
  color: tuple[float, float, float, float] = BOX_COLOR,
) -> EntityCfg:
  """Return a fresh EntityCfg for the box.

  The box sits on the ground (center at ``z = half_height``) a fixed distance
  ``init_x`` ahead of the environment origin. Pass ``mass`` for a pushable box; see
  :func:`get_box_spec`.
  """
  sz = half_size[2]
  return EntityCfg(
    # EntityCfg.spec_fn takes no arguments, so the geometry is bound here.
    spec_fn=partial(
      get_box_spec,
      half_size=half_size,
      mass=mass,
      friction=friction,
      priority=priority,
      color=color,
    ),
    init_state=EntityCfg.InitialStateCfg(
      pos=(init_x, 0.0, sz),
      joint_pos={},  # No named joints: either mocap-wrapped, or a bare freejoint.
    ),
  )
