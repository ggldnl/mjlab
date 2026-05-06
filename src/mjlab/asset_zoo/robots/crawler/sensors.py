"""Crawler sensors."""

from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    BuiltinSensorCfg,
    ObjRef
)

from mjlab.asset_zoo.robots.crawler.crawler_constants import CRAWLER_BASE_NAME


# Sensor attachment points - entity-scoped names as they appear after
# the robot entity is instantiated.  The "robot/" prefix is added
# automatically by the scene; do NOT include it in sensor names.
_IMU_SITE  = ObjRef(type="site", name="imu", entity="robot")   # matches <site name='imu'> in the MJCF

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

FEET_GROUND = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
        mode="subtree",
        pattern=r"^leg_[1-4]_foot",  # one per leg - matches all 4
        entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
)

LEG_SEGMENT_GEOM_NAMES = tuple(
    f"leg_{i}_{seg}_collision"
    for i in (1, 2, 3, 4)
    for seg in ("coxa", "femur", "tibia")
)
LEGS_GROUND = ContactSensorCfg(
    name="legs_ground_contact",
    primary=ContactMatch(
        mode="geom",
        pattern=LEG_SEGMENT_GEOM_NAMES,
        entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
)

SELF_COLLISION = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(
        mode="subtree",
        pattern=CRAWLER_BASE_NAME,
        entity="robot",
    ),
    secondary=ContactMatch(
        mode="subtree",
        pattern=CRAWLER_BASE_NAME,
        entity="robot",
    ),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
)

# Measure robot angular momentum
ROOT_ANGMOM = BuiltinSensorCfg(
    name="root_angmom",
    sensor_type="subtreeangmom",
    obj=ObjRef(type="body", name=CRAWLER_BASE_NAME, entity="robot"),
)

SENSORS = (
    *IMU,
    FEET_GROUND,
    LEGS_GROUND,
    SELF_COLLISION,
    ROOT_ANGMOM,
)