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
from dataclasses import replace


from mjlab.asset_zoo.robots.crawler.crawler_constants import get_crawler_robot_cfg
from mjlab.asset_zoo.robots.crawler.crawler_constants import CRAWLER_BASE_NAME
from mjlab.asset_zoo.robots.crawler.sensors import SENSORS

from mjlab.tasks.velocity.config.crawler.mdp.actions import actions
from mjlab.tasks.velocity.config.crawler.mdp.commands import commands
from mjlab.tasks.velocity.config.crawler.mdp.curriculums import curriculum
from mjlab.tasks.velocity.config.crawler.mdp.terminations import terminations
from mjlab.tasks.velocity.config.crawler.mdp.events import events
from mjlab.tasks.velocity.config.crawler.mdp import observations
from mjlab.tasks.velocity.config.crawler.mdp.rewards import rewards

from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG


def scene_cfg(num_envs: int = 2048) -> SceneCfg:
  return SceneCfg(
    terrain=TerrainEntityCfg(
      terrain_type="generator",
      terrain_generator=replace(ROUGH_TERRAINS_CFG),
      max_init_terrain_level=5,
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
    body_name=CRAWLER_BASE_NAME,
    distance=3.0,
    elevation=-5.0,
    azimuth=90.0,
  )


def sim_cfg() -> SimulationCfg:
  return SimulationCfg(
    nconmax=45,
    njmax=1500,
    contact_sensor_maxmatch=500,
    mujoco=MujocoCfg(
      timestep=0.005,
      iterations=10,
      ls_iterations=20,
      ccd_iterations=500,
    ),
  )


def crawler_velocity_env_cfg(play: bool = False, num_envs: int = 2048) -> ManagerBasedRlEnvCfg:
  """Create Crawler velocity task configuration."""
  cfg = ManagerBasedRlEnvCfg(
    scene=scene_cfg(1 if play else num_envs),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics={},
    viewer=viewer_cfg(),
    sim=sim_cfg(),
    decimation=4,
    episode_length_s=20.0,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

  return cfg