"""
Imports everything else and wires it together into a single ManagerBasedRlEnvCfg.
It also owns the things that don't fit elsewhere:
- scene_cfg(): terrain, robot entity, sensors, number of envs, spacing
- viewer_cfg(): camera attachment for visualization
- sim_cfg(): MuJoCo solver parameters (timestep, iteration counts, contact model)
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab.terrains import TerrainEntityCfg

from mjlab.asset_zoo.robots.crawler.crawler_constants import get_crawler_robot_cfg
from mjlab.asset_zoo.robots.crawler.collisions import BASE_NAME
from mjlab.asset_zoo.robots.crawler.sensors import SENSORS


def scene_cfg(num_envs: int = 2048) -> SceneCfg:
    return SceneCfg(
        terrain=TerrainEntityCfg(
            terrain_type="plane",
            terrain_generator=None,
        ),
        entities={"robot": get_crawler_robot_cfg()},
        sensors=SENSORS,
        num_envs=num_envs,
        extent=10.0,
    )


def viewer_cfg() -> ViewerConfig:
    return ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name=BASE_NAME,
        distance=4.0,
        elevation=-10.0,
        azimuth=90.0,
    )


def sim_cfg() -> SimulationCfg:
    return SimulationCfg(
        nconmax=50,
        njmax=400,
        contact_sensor_maxmatch=500,
        mujoco=MujocoCfg(
            timestep=0.002,
            iterations=10,
            ls_iterations=20,
            ccd_iterations=500,
            cone="pyramidal",
        ),
    )


def crawler_velocity_abstraction_env_cfg(play: bool = False, num_envs: int = 2048) -> ManagerBasedRlEnvCfg:
    """Create Crawler velocity tracking config using abstraction-based rewards."""
    cfg = ManagerBasedRlEnvCfg(
        scene=scene_cfg(1 if play else num_envs),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum={},   # TODO
        metrics={},
        viewer=viewer_cfg(),
        sim=sim_cfg(),
        decimation=10,       # policy runs at 50 Hz (0.002 * 10)
        episode_length_s=20.0,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}

    return cfg