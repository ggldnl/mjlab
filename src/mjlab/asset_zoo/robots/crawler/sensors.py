"""Crawler sensors."""

from mjlab.asset_zoo.robots.crawler.collisions import (
  BASE_COLLISION_NAME,
  BASE_NAME,
  COXA_COLLISION_NAMES,
  FEMUR_COLLISION_NAMES,
  FOOT_BODY_REGEX,
  FOOT_SITE_NAMES,
  TIBIA_COLLISION_NAMES,
)
from mjlab.sensor import (
  BuiltinSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)

IMU_SITE_NAME = "imu"

# Sensor attachment points - entity-scoped names as they appear after
# the robot entity is instantiated.  The "robot/" prefix is added
# automatically by the scene; do NOT include it in sensor names.
_IMU_SITE = ObjRef(
  type="site", name=IMU_SITE_NAME, entity="robot"
)  # matches <site name='imu'> in the MJCF

# IMU sensors - all attached to the <site name="imu"> on the base body.
IMU_ANG_VEL = BuiltinSensorCfg(
  name="imu_ang_vel",
  sensor_type="gyro",
  obj=_IMU_SITE,
)

IMU_LIN_VEL = BuiltinSensorCfg(
  name="imu_lin_vel",
  sensor_type="velocimeter",
  obj=_IMU_SITE,
)

IMU_LIN_ACC = BuiltinSensorCfg(
  name="imu_lin_acc",
  sensor_type="accelerometer",
  obj=_IMU_SITE,
)

IMU_ORIENTATION = BuiltinSensorCfg(
  name="imu_orientation",
  sensor_type="framequat",
  obj=_IMU_SITE,
)

IMU = (
  IMU_ANG_VEL,
  IMU_LIN_VEL,
  IMU_LIN_ACC,
  IMU_ORIENTATION,
)

# How many steps back in time we detect
HISTORY_LENGTH = 4

# Feet-ground contact detection
FEET_GROUND = ContactSensorCfg(
  name="feet_ground_contact",
  primary=ContactMatch(
    mode="subtree",
    pattern=FOOT_BODY_REGEX,
    entity="robot",
  ),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found", "force"),
  reduce="netforce",
  num_slots=1,
  track_air_time=True,
)

# Legs ground collision detection
LEGS_GROUND = ContactSensorCfg(
  name="legs_ground_contact",
  primary=ContactMatch(
    mode="geom",
    pattern=(
      *COXA_COLLISION_NAMES,
      *FEMUR_COLLISION_NAMES,
      *TIBIA_COLLISION_NAMES,
    ),
    entity="robot",
  ),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found", "force"),
  reduce="none",
  num_slots=1,
  history_length=HISTORY_LENGTH,
)

# Non-feet ground collision detection
NONFEET_GROUND = ContactSensorCfg(
  name="nonfeet_ground_contact",
  primary=ContactMatch(
    mode="geom",
    pattern=(
      BASE_COLLISION_NAME,
      *COXA_COLLISION_NAMES,
      *FEMUR_COLLISION_NAMES,
      *TIBIA_COLLISION_NAMES,
    ),
    entity="robot",
  ),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found", "force"),
  reduce="none",
  num_slots=1,
  history_length=HISTORY_LENGTH,
)

SELF_COLLISION = ContactSensorCfg(
  name="self_collision",
  primary=ContactMatch(
    mode="subtree",
    pattern=BASE_NAME,
    entity="robot",
  ),
  secondary=ContactMatch(
    mode="subtree",
    pattern=BASE_NAME,
    entity="robot",
  ),
  fields=("found", "force"),
  reduce="none",
  num_slots=1,
  history_length=HISTORY_LENGTH,
)

# Measure robot angular momentum
ROOT_ANGMOM = BuiltinSensorCfg(
  name="root_angmom",
  sensor_type="subtreeangmom",
  obj=ObjRef(type="body", name=BASE_NAME, entity="robot"),
)

# Terrain sensors
TERRAIN_SCAN = RayCastSensorCfg(
  name="terrain_scan",
  frame=ObjRef(type="body", name=BASE_NAME, entity="robot"),
  ray_alignment="yaw",
  pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
  max_distance=5.0,
  exclude_parent_body=True,
  include_geom_groups=(0,),
  debug_vis=True,
  viz=RayCastSensorCfg.VizCfg(
    # Cyan spheres
    hit_sphere_radius=0.1,
  ),
)

FOOT_HEIGHT_SCAN = TerrainHeightSensorCfg(
  name="foot_height_scan",
  frame=tuple(
    ObjRef(type="site", name=name, entity="robot") for name in FOOT_SITE_NAMES
  ),
  ray_alignment="yaw",
  pattern=RingPatternCfg.single_ring(radius=0.015, num_samples=4),
  max_distance=1.0,
  exclude_parent_body=True,
  include_geom_groups=(0,),
  debug_vis=True,
  viz=TerrainHeightSensorCfg.VizCfg(
    # Magenta spheres
    show_rays=True,
    hit_sphere_radius=0.1,
    hit_color=(1.0, 0.0, 1.0, 0.8),
    hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
  ),
)

# Expose all sensors
SENSORS = (
  *IMU,
  FEET_GROUND,
  LEGS_GROUND,
  NONFEET_GROUND,
  SELF_COLLISION,
  ROOT_ANGMOM,
  TERRAIN_SCAN,
  FOOT_HEIGHT_SCAN,
)
