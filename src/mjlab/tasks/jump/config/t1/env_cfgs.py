"""Booster T1 jump environment configurations."""

from mjlab.asset_zoo.robots import T1_ACTION_SCALE, get_t1_robot_cfg
from mjlab.asset_zoo.robots.booster_t1.sensors import FEET_GROUND_CONTACT_SENSOR
from mjlab.asset_zoo.robots.booster_t1.t1_constants import BASE_BODY_NAME
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.jump.jump_env_cfg import make_jump_env_cfg
from mjlab.tasks.jump.mdp.abstractions import JumpAbstractionCfg

# Standing trunk height of the T1 (from its init state), used as the takeoff and
# landing reference height for the ballistic abstraction.
T1_NOMINAL_BASE_HEIGHT = 0.665


def booster_t1_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Booster T1 standing-jump configuration."""
  cfg = make_jump_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_t1_robot_cfg()}
  cfg.scene.sensors = (FEET_GROUND_CONTACT_SENSOR,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = T1_ACTION_SCALE

  jump = cfg.abstractions["jump"]
  assert isinstance(jump, JumpAbstractionCfg)
  jump.base_body_name = BASE_BODY_NAME
  jump.nominal_base_height = T1_NOMINAL_BASE_HEIGHT

  cfg.viewer.body_name = BASE_BODY_NAME

  if play:
    cfg.scene.num_envs = 16
    cfg.observations["actor"].enable_corruption = False

  return cfg
