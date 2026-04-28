"""Crawler config main entry point."""

import mujoco
from mjlab.entity import EntityCfg

from pathlib import Path

from .actuators import CRAWLER_ARTICULATIONS, INIT_STATE
from .collisions import FULL_COLLISION, FEET_ONLY_COLLISION


LOCAL_FOLDER = Path(__file__).parent
CRAWLER_DESCRIPTION_PATH: Path = LOCAL_FOLDER / "xml" / "crawler.xml"
assert CRAWLER_DESCRIPTION_PATH.exists()


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(CRAWLER_DESCRIPTION_PATH))

def get_crawler_robot_cfg() -> EntityCfg:
    """Get a fresh Crawler robot configuration instance."""
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FEET_ONLY_COLLISION,),
        spec_fn=get_spec,
        articulation=CRAWLER_ARTICULATIONS,
    )

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


if __name__ == "__main__":

    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_crawler_robot_cfg())
    model = robot.spec.compile()
    viewer.launch(model)