from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

# Physical ball properties
BALL_RADIUS = 0.11  # m (22 cm diameter, size-5)
BALL_MASS = 0.425  # kg
BALL_FRICTION = 0.8  # dimensionless
BALL_COLOR = 0.8, 0.2, 0.2, 1

# Unit sphere .obj (radius = 1)
MESH_TEMPLATE_RADIUS = 1.0

_ASSETS_DIR = Path(__file__).parent


def get_ball_spec() -> mujoco.MjSpec:
  """Return MjSpec describing a free-floating ball."""

  mesh_path = str(_ASSETS_DIR / "unit_sphere.obj")

  # Uniform scale needed to transform the template sphere
  # into a sphere with radius BALL_RADIUS.
  mesh_scale = BALL_RADIUS / MESH_TEMPLATE_RADIUS

  # Solid sphere inertia:
  # I = 2/5 m r^2
  inertia = (2.0 / 5.0) * BALL_MASS * BALL_RADIUS**2

  xml = f"""
<mujoco>
  <asset>
    <material
        name="ball_mat"
        texrepeat="1 1"
        specular="0.3"
        shininess="0.5"
        reflectance="0.05"
        rgba="{" ".join([str(e) for e in BALL_COLOR])}"/>

    <mesh
        name="ball_mesh"
        file="{mesh_path}"
        scale="{mesh_scale} {mesh_scale} {mesh_scale}"/>
  </asset>

  <worldbody>
    <body name="ball" pos="0 0 {BALL_RADIUS}">
      <freejoint/>

      <site name="root_site"/>

      <inertial
          pos="0 0 0"
          mass="{BALL_MASS}"
          diaginertia="{inertia} {inertia} {inertia}"/>

      <!-- Exact analytical collision sphere -->
      <geom
          name="ball_collision"
          type="sphere"
          size="{BALL_RADIUS}"
          condim="6"
          friction="{BALL_FRICTION} 0.005 0.02"
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


def get_ball_cfg() -> EntityCfg:
  """Return a fresh EntityCfg for the ball."""

  return EntityCfg(
    spec_fn=get_ball_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, BALL_RADIUS),
      joint_pos={},
    ),
  )
