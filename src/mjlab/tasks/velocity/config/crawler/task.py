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

from experiments.robots.crawler.config import get_crawler_robot_cfg
from experiments.robots.crawler.constants import CRAWLER_BASE_NAME
from experiments.robots.crawler.sensors import (
  FEET_GROUND_CONTACT_SENSOR,
  NONFEET_GROUND_CONTACT_SENSOR,
  SELF_COLLISION_SENSOR,
  TERRAIN_SCAN,
  ROOT_ANGMOM,
  IMU
)

from .terrains import play_terrain_cfg, training_terrain_cfg
from .cact import actions, commands, curriculum, terminations
from .events import events
from .observations import observations
from .rewards import rewards


def scene_cfg(play: bool = False, num_envs: int = 2048) -> SceneCfg:
  terrain = play_terrain_cfg() if play else training_terrain_cfg() # pick at call site
  return SceneCfg(
    terrain=terrain,
    entities={"robot": get_crawler_robot_cfg()},
    sensors=(
      FEET_GROUND_CONTACT_SENSOR,
      NONFEET_GROUND_CONTACT_SENSOR,
      SELF_COLLISION_SENSOR,
      TERRAIN_SCAN,
      ROOT_ANGMOM,
      *IMU
    ),
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


def crawler_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Crawler velocity task configuration."""
  cfg = ManagerBasedRlEnvCfg(
    scene=scene_cfg(play),
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
