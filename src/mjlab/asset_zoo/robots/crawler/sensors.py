"""Crawler sensors."""

from mjlab.sensor import (
    ObjRef,
    BuiltinSensorCfg,
)

# Sensor attachment points — entity-scoped names as they appear after
# the robot entity is instantiated.  The "robot/" prefix is added
# automatically by the scene; do NOT include it in sensor names.
_IMU_SITE  = ObjRef(type="site", name="imu", entity="robot")   # matches <site name='imu'> in the MJCF

# IMU sensors — all attached to the <site name="imu"> on the base body.
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