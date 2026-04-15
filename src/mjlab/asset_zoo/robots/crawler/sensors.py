"""Crawler sensors."""

from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  BuiltinSensorCfg,
)

from experiments.robots.crawler.constants import (
  CRAWLER_BASE_NAME,
  CRAWLER_IMU_SITE_NAME,
  CRAWLER_BASE_SITE_NAME,
  CRAWLER_FOOT_GEOM_NAMES,
  CRAWLER_FEMUR_GEOM_NAMES,
  CRAWLER_TIBIA_GEOM_NAMES,
)


# Contact Sensor Configurations
# A ContactSensorCfg works by finding contact pairs where primary touches secondary,
# then aggregating the results according to reduce.

# Tracks aggregate contact between each foot body and terrain
FEET_GROUND_CONTACT_SENSOR = ContactSensorCfg(
  name="feet_ground_contact",
  primary=ContactMatch(
    mode="geom",
    pattern=CRAWLER_FOOT_GEOM_NAMES,
    entity="robot",
  ),
  secondary=ContactMatch(
    mode="body",
    pattern="terrain",
  ),
  fields=("found", "force"),
  reduce="netforce",  #  all contact forces on a given primary geom are summed into a single net force vector
  num_slots=1,  # 1 contact pair tracked per primary geom
  history_length=1,
  track_air_time=True,  # adds a running counter per foot tracking how long each has been off the ground
)

NONFEET_GROUND_CONTACT_SENSOR = ContactSensorCfg(
  name="nonfeet_ground_contact",
  primary=ContactMatch(
    mode="geom",
    entity="robot",
    pattern=r".+",
    exclude=CRAWLER_FOOT_GEOM_NAMES,
  ),
  secondary=ContactMatch(
    mode="body",
    pattern="terrain"
  ),
  fields=("found", "force"),
  reduce="none",  # no aggregation, every contact pair is reported individually (one row per non-foot geom)
  num_slots=1,
  history_length=1,
)

# Self-collision sensors

# Detects collisions between robot base and body parts. Useful for penalty/safety rewards
SELF_COLLISION_SENSOR = ContactSensorCfg(
  name="self_collision",
  primary=ContactMatch(
    mode="body",
    pattern=(CRAWLER_BASE_NAME, ),  # only the base
    entity="robot",
  ),
  secondary=ContactMatch(
    mode="geom",
    pattern=(
      *CRAWLER_FOOT_GEOM_NAMES,
      *CRAWLER_TIBIA_GEOM_NAMES,
      *CRAWLER_FEMUR_GEOM_NAMES
    ),
    entity="robot",
  ),
  secondary_policy="any",  # allow matching against multiple secondary bodies
  fields=("found", "force"),
  reduce="netforce",
  num_slots=1,
  history_length=4,
)

# Simulation only sensor

TERRAIN_SCAN = RayCastSensorCfg(
  name="terrain_scan",
  frame=ObjRef(type="body", name=CRAWLER_BASE_NAME, entity="robot"),
  ray_alignment="yaw",
  pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
  max_distance=5.0,
  include_geom_groups=(0,),  # terrain only
  exclude_parent_body=True,
  # debug_vis=True,
  # viz=RayCastSensorCfg.VizCfg(show_normals=True),
)

# Angular momentum sensor (whole robot subtree from base)
ROOT_ANGMOM = BuiltinSensorCfg(
    name="root_angmom",
    sensor_type="subtreeangmom",
    obj=ObjRef(type="body", name=CRAWLER_BASE_SITE_NAME),
)

# IMU sensors

imu_site = ObjRef(type="site", name=CRAWLER_IMU_SITE_NAME)

IMU_ANG_VEL = BuiltinSensorCfg(
  name="imu_ang_vel",
  sensor_type="gyro",
  obj=imu_site,
)

IMU_LIN_VEL = BuiltinSensorCfg(
  name="imu_lin_vel",
  sensor_type="velocimeter",
  obj=imu_site,
)

IMU_LIN_ACC = BuiltinSensorCfg(
  name="imu_lin_acc",
  sensor_type="accelerometer",
  obj=imu_site,
)

IMU_ORIENTATION = BuiltinSensorCfg(
  name="orientation",
  sensor_type="framequat",
  obj=imu_site,
)

IMU = (
  IMU_ANG_VEL,
  IMU_LIN_VEL,
  IMU_LIN_ACC,
  IMU_ORIENTATION,
)
