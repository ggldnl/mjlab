"""Crawler sensors."""

from mjlab.sensor import (
  ObjRef,
  BuiltinSensorCfg,
)


CRAWLER_IMU_SITE_NAME = "robot/imu"
CRAWLER_BASE_SITE_NAME = "robot/base"

# Angular momentum sensor (whole robot subtree from base)

ROOT_ANGMOM = BuiltinSensorCfg(
    name="robot/root_angmom",
    sensor_type="subtreeangmom",
    obj=ObjRef(type="body", name=CRAWLER_BASE_SITE_NAME),
)

# IMU

imu_site = ObjRef(type="site", name=CRAWLER_IMU_SITE_NAME)

IMU_ANG_VEL = BuiltinSensorCfg(
  name="robot/imu_ang_vel",
  sensor_type="gyro",
  obj=imu_site,
)

IMU_LIN_VEL = BuiltinSensorCfg(
  name="robot/imu_lin_vel",
  sensor_type="velocimeter",
  obj=imu_site,
)

IMU_LIN_ACC = BuiltinSensorCfg(
  name="robot/imu_lin_acc",
  sensor_type="accelerometer",
  obj=imu_site,
)

IMU_ORIENTATION = BuiltinSensorCfg(
  name="robot/orientation",
  sensor_type="framequat",
  obj=imu_site,
)

IMU = (
  IMU_ANG_VEL,
  IMU_LIN_VEL,
  IMU_LIN_ACC,
  IMU_ORIENTATION,
)
