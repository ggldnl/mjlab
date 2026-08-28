"""Unitree G1 constants."""

from functools import partial
from pathlib import Path

import mujoco
import numpy as np

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

G1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
)
assert G1_XML.exists()

##
# Hands.
##

# Radius of the dome that stands in for a hand, in metres.
#
# 0.03 is the wrist housing's own radius where the hand bolts on, measured off the
# `*_wrist_yaw_link` mesh, so at this value the dome and the housing share a silhouette
# and the join is seamless. Larger and the dome overhangs the housing like a knob;
# smaller and it sinks into it. Either still meets the housing -- the flat face is what
# is anchored, not the centre -- so the radius is free to change without the hand coming
# adrift.
HAND_RADIUS = 0.03

# Colour of the dome's visual geom. The rubber hand it replaces is drawn in the XML's
# `black` material, and reusing that keeps a G1 looking like a G1 from a distance.
HAND_MATERIAL = "black"

# Facets of the dome, as latitude rings and longitude segments. Visual only, so this
# trades nothing but triangles: MuJoCo takes the convex hull of the points below and
# the collision geom is a primitive sphere regardless.
_HAND_DOME_RINGS = 6
_HAND_DOME_SEGMENTS = 24


def _dome_points(radius: float) -> np.ndarray:
  """Points whose convex hull is a dome of ``radius``, flat face on the x = 0 plane.

  Handed to MuJoCo as a mesh with vertices and no faces, which is the documented way of
  saying "take the convex hull of this". For a point set covering a hemispherical cap
  plus its rim, that hull is the dome and the flat disc that closes it, so there is no
  triangulation to write and no winding order to get wrong.

  The pole is on +x because that is the wrist's own axis, and the mount face the dome
  sits on is perpendicular to it. The rubber hand cants a little off that axis further
  out, where the fingers are, but the join does not.
  """
  points = [(radius, 0.0, 0.0)]
  for i in range(1, _HAND_DOME_RINGS + 1):
    polar = 0.5 * np.pi * i / _HAND_DOME_RINGS  # 0 at the pole, pi/2 at the rim
    axial = radius * np.cos(polar)
    ring = radius * np.sin(polar)
    for j in range(_HAND_DOME_SEGMENTS):
      azimuth = 2.0 * np.pi * j / _HAND_DOME_SEGMENTS
      points.append((axial, ring * np.cos(azimuth), ring * np.sin(azimuth)))
  return np.array(points, dtype=np.float64)


def _replace_hand_with_dome(spec: mujoco.MjSpec, side: str) -> None:
  """Swap one rubber hand for a dome, in place.

  The XML is left alone deliberately, so this is surgery on the loaded spec: find the
  hand's two geoms in the wrist yaw link, note where the hand bolts on, delete both, and
  put a dome there instead. The mesh asset goes too, since nothing else refers to it.

  Everything is anchored to the hand's own mount point, read off the rubber hand geom's
  position. That is the plane where the housing stops and the hand began, so a dome
  seated on it stays joined at any radius. Anchoring instead to the middle of the thing
  being replaced -- the collision capsule's midpoint, say, which is 0.069 m further out
  because the capsule spans the whole hand rather than its root -- leaves a stand-in that
  floats free of the arm as soon as it is made smaller than the gap.

  Visual and collision are different shapes on purpose. The visual is the dome, so what
  you see is a hemisphere growing out of the wrist. The collision geom is a whole sphere
  centred on the same point, because a primitive sphere is far cheaper to collide against
  than a mesh at four thousand environments. Its back half costs nothing: it sits inside
  the volume the forearm capsule already occupies -- that capsule's surface reaches
  0.049 m along this body's x, well past the 0.0415 m mount -- so nothing outside the
  robot can touch it without already being inside the arm, and MuJoCo's parent-child
  filter keeps it from touching the arm itself.

  Mass is untouched, and that is a property of the model rather than luck: the wrist yaw
  link carries an explicit `<inertial>`, so MuJoCo takes the body's mass and inertia from
  that and never from its geoms. Swapping geometry here changes what the robot collides
  with and what it looks like, not what it weighs.
  """
  body = spec.body(f"{side}_wrist_yaw_link")
  mesh_name = f"{side}_rubber_hand"
  hand = next(g for g in body.geoms if g.meshname == mesh_name)
  mount = np.array(hand.pos)

  spec.delete(hand)
  spec.delete(spec.mesh(mesh_name))
  spec.delete(spec.geom(f"{side}_hand_collision"))

  dome = spec.add_mesh()
  dome.name = f"{side}_hand_dome"
  dome.uservert = _dome_points(HAND_RADIUS).flatten()

  # Keeping the collision geom's name is what makes this a drop-in swap: every collision
  # policy in this file selects on `.*_collision`, and so does anything downstream that
  # names the hand.
  #
  # Attributes are set here rather than by pointing the geoms at the XML's `visual` and
  # `collision` default classes: assigning a class to a geom the spec API has already
  # created does not apply that class's attributes, so it would look right and behave
  # like a bare geom.
  collision = body.add_geom()
  collision.name = f"{side}_hand_collision"
  collision.type = mujoco.mjtGeom.mjGEOM_SPHERE
  collision.size[0] = HAND_RADIUS
  collision.pos = mount
  collision.group = 3
  collision.rgba[:] = (0.2, 0.6, 0.2, 0.3)

  visual = body.add_geom()
  visual.name = f"{side}_hand_visual"
  visual.type = mujoco.mjtGeom.mjGEOM_MESH
  visual.meshname = dome.name
  visual.pos = mount
  visual.group = 2
  visual.material = HAND_MATERIAL
  visual.contype = 0
  visual.conaffinity = 0
  visual.density = 0.0


def get_spec(hands: bool = False) -> mujoco.MjSpec:
  """Load the G1 spec.

  Args:
    hands: Keep the rubber hands the XML ships with. The default replaces each of them
      with a dome seated where the hand bolted on: a simpler thing to collide against and
      to reason about, being one radius and one centre rather than a five-fingered mesh
      and a capsule canted off the wrist axis. Nothing about the arm's joints, masses or
      actuators changes either way.
  """
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  if not hands:
    for side in ("left", "right"):
      _replace_hand_with_dome(spec, side)
  return spec


##
# Actuator config.
##

# Motor specs (from Unitree).
ROTOR_INERTIAS_5020 = (
  0.139e-4,
  0.017e-4,
  0.169e-4,
)
GEARS_5020 = (
  1,
  1 + (46 / 18),
  1 + (56 / 16),
)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

ROTOR_INERTIAS_7520_14 = (
  0.489e-4,
  0.098e-4,
  0.533e-4,
)
GEARS_7520_14 = (
  1,
  4.5,
  1 + (48 / 22),
)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (
  0.489e-4,
  0.109e-4,
  0.738e-4,
)
GEARS_7520_22 = (
  1,
  4.5,
  5,
)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

ROTOR_INERTIAS_4010 = (
  0.068e-4,
  0.0,
  0.0,
)
GEARS_4010 = (
  1,
  5,
  5,
)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_4010, GEARS_4010
)

ACTUATOR_5020 = ElectricActuator(
  reflected_inertia=ARMATURE_5020,
  velocity_limit=37.0,
  effort_limit=25.0,
)
ACTUATOR_7520_14 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14,
  velocity_limit=32.0,
  effort_limit=88.0,
)
ACTUATOR_7520_22 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_22,
  velocity_limit=20.0,
  effort_limit=139.0,
)
ACTUATOR_4010 = ElectricActuator(
  reflected_inertia=ARMATURE_4010,
  velocity_limit=22.0,
  effort_limit=5.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

G1_ACTUATOR_5020 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)
G1_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
G1_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)
G1_ACTUATOR_4010 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  stiffness=STIFFNESS_4010,
  damping=DAMPING_4010,
  effort_limit=ACTUATOR_4010.effort_limit,
  armature=ACTUATOR_4010.reflected_inertia,
)

# Waist pitch/roll and ankles are 4-bar linkages with 2 5020 actuators.
# Due to the parallel linkage, the effective armature at the ankle and waist joints
# is configuration dependent. Since the exact geometry of the linkage is unknown, we
# assume a nominal 1:1 gear ratio. Under this assumption, the joint armature in the
# nominal configuration is approximated as the sum of the 2 actuators' armatures.
G1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)
G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.783675),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 1.28,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.76),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1, ".*": 0},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1, ".*": 0},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_robot_cfg(hands: bool = False) -> EntityCfg:
  """Get a fresh G1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.

  Args:
    hands: Keep the rubber hands rather than the spheres that replace them by default.
      See :func:`get_spec`.
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    # EntityCfg.spec_fn takes no arguments, so the choice is bound here.
    spec_fn=partial(get_spec, hands=hands),
    articulation=G1_ARTICULATION,
  )


G1_ACTION_SCALE: dict[str, float] = {}
for a in G1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    G1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_robot_cfg())

  viewer.launch(robot.spec.compile())
