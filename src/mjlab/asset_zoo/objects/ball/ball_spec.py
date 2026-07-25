from functools import partial
from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

# Default physical ball properties (size-5 football). Every function below takes
# these as defaults, so a differently sized ball (a 40 mm table tennis ball, say) is
# a matter of passing radius/mass rather than a second asset.
BALL_RADIUS = 0.11  # m (22 cm diameter, size-5)
BALL_MASS = 0.425  # kg
BALL_FRICTION = 0.8  # dimensionless
BALL_COLOR = 0.8, 0.2, 0.2, 1

# Unit sphere .obj (radius = 1)
MESH_TEMPLATE_RADIUS = 1.0

_ASSETS_DIR = Path(__file__).parent


def get_ball_spec(
  radius: float = BALL_RADIUS,
  mass: float = BALL_MASS,
  friction: float = BALL_FRICTION,
  color: tuple[float, float, float, float] = BALL_COLOR,
) -> mujoco.MjSpec:
  """Return MjSpec describing a free-floating ball of the given size and mass."""

  mesh_path = str(_ASSETS_DIR / "unit_sphere.obj")

  # Uniform scale needed to transform the template sphere
  # into a sphere with the requested radius.
  mesh_scale = radius / MESH_TEMPLATE_RADIUS

  # Solid sphere inertia:
  # I = 2/5 m r^2
  inertia = (2.0 / 5.0) * mass * radius**2

  xml = f"""
<mujoco>
  <asset>
    <material
        name="ball_mat"
        texrepeat="1 1"
        specular="0.3"
        shininess="0.5"
        reflectance="0.05"
        rgba="{" ".join([str(e) for e in color])}"/>

    <mesh
        name="ball_mesh"
        file="{mesh_path}"
        scale="{mesh_scale} {mesh_scale} {mesh_scale}"/>
  </asset>

  <worldbody>
    <body name="ball" pos="0 0 {radius}">
      <freejoint/>

      <site name="root_site"/>

      <inertial
          pos="0 0 0"
          mass="{mass}"
          diaginertia="{inertia} {inertia} {inertia}"/>

      <!-- Exact analytical collision sphere -->
      <geom
          name="ball_collision"
          type="sphere"
          size="{radius}"
          condim="6"
          friction="{friction} 0.005 0.02"
          rgba="0 0 0 0"/>

      <!-- Scaled visual mesh -->
      <geom
          name="ball_visual"
          type="mesh"
          mesh="ball_mesh"
          material="ball_mat"
          contype="0"
          conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

  return mujoco.MjSpec.from_string(xml)


def get_ball_cfg(
  radius: float = BALL_RADIUS,
  mass: float = BALL_MASS,
  friction: float = BALL_FRICTION,
  color: tuple[float, float, float, float] = BALL_COLOR,
  pos: tuple[float, float, float] | None = None,
) -> EntityCfg:
  """Return a fresh EntityCfg for a ball of the given size and mass.

  `pos` defaults to resting on the ground plane (z = radius).
  """

  return EntityCfg(
    # EntityCfg.spec_fn takes no arguments, so the geometry is bound here.
    spec_fn=partial(
      get_ball_spec, radius=radius, mass=mass, friction=friction, color=color
    ),
    init_state=EntityCfg.InitialStateCfg(
      pos=pos if pos is not None else (0.0, 0.0, radius),
      joint_pos={},
    ),
  )
