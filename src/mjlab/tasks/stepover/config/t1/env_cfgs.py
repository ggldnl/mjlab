"""Booster T1 step-over environment configurations."""

from mjlab.asset_zoo.robots import T1_ACTION_SCALE, get_t1_robot_cfg
from mjlab.asset_zoo.robots.booster_t1.sensors import FEET_GROUND_CONTACT_SENSOR
from mjlab.asset_zoo.robots.booster_t1.t1_constants import BASE_BODY_NAME
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.stepover.mdp.abstractions import StepOverAbstractionCfg
from mjlab.tasks.stepover.stepover_env_cfg import make_stepover_env_cfg

# Foot sites used to track the swing-foot via-points.
FOOT_SITE_NAMES = ("left_foot", "right_foot")


def booster_t1_stepover_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Booster T1 barrier step-over configuration."""
  cfg = make_stepover_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70
  cfg.sim.njmax = 100  # Avoid nefc overflow during contact-rich step-over.

  cfg.scene.entities = {"robot": get_t1_robot_cfg()}
  cfg.scene.sensors = (FEET_GROUND_CONTACT_SENSOR,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = T1_ACTION_SCALE

  step = cfg.abstractions["stepover"]
  assert isinstance(step, StepOverAbstractionCfg)
  step.base_body_name = BASE_BODY_NAME
  step.foot_site_names = FOOT_SITE_NAMES

  cfg.viewer.body_name = BASE_BODY_NAME

  if play:
    cfg.scene.num_envs = 16
    cfg.observations["actor"].enable_corruption = False
    cfg.episode_length_s = int(1e9)  # Effectively infinite for play.

  return cfg
