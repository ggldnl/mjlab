"""Box obstacle entity for the jumping task.

A static rectangular obstacle the robot must jump over. The box has no freejoint,
so mjlab auto-wraps it as a *mocap* body: it still participates in collisions (a
rigid, immovable barrier) but is never moved by physics, and its pose can be set
per-environment at reset via :func:`reset_root_state_uniform`.
"""

import mujoco

from mjlab.entity import EntityCfg

# Half-extents (m) of the default obstacle: 0.10 m deep (x), 1.20 m wide (y),
# 0.15 m tall (z). Wide enough that a forward-walking robot cannot trivially
# sidestep it, and low enough to be a reasonable first hurdle.
BOX_HALF_SIZE: tuple[float, float, float] = (0.05, 0.6, 0.075)

# Distance (m) ahead of the environment origin where the obstacle is placed by
# default. The robot resets near the origin facing +x, so it walks into the box.
BOX_INIT_X: float = 1.5


def get_box_spec(
  half_size: tuple[float, float, float] = BOX_HALF_SIZE,
) -> mujoco.MjSpec:
  """Return MjSpec describing a static (fixed-base) box obstacle."""
  sx, sy, sz = half_size
  xml = f"""<mujoco>
  <worldbody>
    <body name="obstacle" pos="0 0 {sz}">
      <geom name="obstacle_collision" type="box" size="{sx} {sy} {sz}"
            condim="3" friction="1.0 0.005 0.0001"
            rgba="0.85 0.35 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""
  return mujoco.MjSpec.from_string(xml)


def get_box_cfg(
  half_size: tuple[float, float, float] = BOX_HALF_SIZE,
  init_x: float = BOX_INIT_X,
) -> EntityCfg:
  """Return a fresh EntityCfg for the box obstacle.

  The box sits on the ground (center at ``z = half_height``) a fixed distance
  ``init_x`` ahead of the environment origin.
  """
  sz = half_size[2]
  return EntityCfg(
    spec_fn=lambda: get_box_spec(half_size),
    init_state=EntityCfg.InitialStateCfg(
      pos=(init_x, 0.0, sz),
      joint_pos={},  # No joints (fixed base, mocap-wrapped).
    ),
  )
