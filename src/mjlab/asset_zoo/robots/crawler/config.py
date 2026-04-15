"""Crawler config main entry point."""

from mjlab.entity import EntityCfg

from experiments.robots.crawler.actuators import CRAWLER_ARTICULATIONS
from experiments.robots.crawler.collisions import CRAWLER_COLLISIONS
from experiments.robots.crawler.constants import get_spec, INIT_STATE


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