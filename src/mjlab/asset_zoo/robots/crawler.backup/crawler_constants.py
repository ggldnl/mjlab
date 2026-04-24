"""Crawler config main entry point."""

import mujoco
from mjlab.entity import EntityCfg
# from mjlab.utils.os import update_assets

from pathlib import Path

from .actuators import CRAWLER_ARTICULATIONS
from .collisions import CRAWLER_COLLISIONS


LOCAL_FOLDER = Path(__file__).parent
CRAWLER_DESCRIPTION_PATH: Path = LOCAL_FOLDER / "xml" / "crawler.xml"
assert CRAWLER_DESCRIPTION_PATH.exists()

"""
def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, CRAWLER_DESCRIPTION_PATH.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(CRAWLER_DESCRIPTION_PATH))
    spec.assets = get_assets(spec.meshdir)
    return spec
"""

def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(CRAWLER_DESCRIPTION_PATH))

# Keyframes

# Robot standing initially, no need to learn how to get up.
# We need to manually tune the initial height. Using a kinematic
# model to derive this will defy the sole purpose of using DRL:
# it's difficult to have models for complex robots.
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.04),
    joint_pos={
        "base_leg_[1-4]_coxa": 0.0,
        "leg_[1-4]_coxa_leg_[1-4]_femur": -0.25,
        "leg_[1-4]_femur_leg_[1-4]_tibia": -1.75,
    },
    joint_vel={".*": 0.0},
)

# Constants

# Indices for leg diagonal pairs, used for trot and fast gait.
CRAWLER_LEG_DIAGONAL_PAIRS = [(0, 2), (1, 3)]

CRAWLER_FEMUR_GEOM_NAMES = (
    "leg_1_femur_geom",
    "leg_2_femur_geom",
    "leg_3_femur_geom",
    "leg_4_femur_geom",
)

CRAWLER_TIBIA_GEOM_NAMES = (
    "leg_1_tibia_geom",
    "leg_2_tibia_geom",
    "leg_3_tibia_geom",
    "leg_4_tibia_geom",
)

CRAWLER_FOOT_GEOM_NAMES = (
    "leg_1_foot_geom",
    "leg_2_foot_geom",
    "leg_3_foot_geom",
    "leg_4_foot_geom",
)

CRAWLER_FOOT_SITE_NAMES = (
    "leg_1_foot_site",
    "leg_2_foot_site",
    "leg_3_foot_site",
    "leg_4_foot_site",
)

CRAWLER_BASE_NAME = "base"

# Entity-scoped sites: we will attach all sensors through python,
# minimal manual XML editing. We will only need some sites to attach
# the sensors to.

"""
# Entity-scoped site
CRAWLER_IMU_SITE_NAME = "robot/imu"

# Sensor should use plain name (sensors.py)
imu_site = ObjRef(type="site", name=CRAWLER_IMU_SITE_NAME)
IMU_ANG_VEL = BuiltinSensorCfg(
  name="imu_ang_vel",
  sensor_type="gyro",
  obj=imu_site,
)

# Observations should use plain name (observations.py)
obs = scene["imu_ang_vel"]
"""
CRAWLER_IMU_SITE_NAME = "robot/imu"
CRAWLER_BASE_SITE_NAME = "robot/base"


def get_crawler_robot_cfg() -> EntityCfg:
    """Get a fresh Crawler robot configuration instance."""
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(CRAWLER_COLLISIONS,),
        spec_fn=get_spec,
        articulation=CRAWLER_ARTICULATIONS,
    )


if __name__ == "__main__":

    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_crawler_robot_cfg())
    model = robot.spec.compile()
    viewer.launch(model)