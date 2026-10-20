"""T1 flat velocity task plus the LAFAN style reward.

Everything of the stock task is kept except three things:

    style           new, nearest mocap frame reward, see style.py
    pose            narrowed to the head, which the library leaves out. On the body it
                    pulled towards the home pose and fought the style term
    twist ranges    held inside the library's velocity coverage, no speed curriculum

The command ranges are a first guess at what LAFAN walks cover once scaled to the T1.
Compare them with the coverage dataset.py prints and narrow them where the data is thin.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.bridging.config.t1.skills.walk.distance import (
  WalkDistanceCommandCfg,
  goal_reward,
  invalid_walk,
  path_reward,
  walk_goal,
)
from mjlab.tasks.bridging.config.t1.skills.walk.gait import supported_swing
from mjlab.tasks.bridging.config.t1.skills.walk.style import LIBRARY_FILE, motion_style
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.t1.env_cfgs import booster_t1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

STYLE_WEIGHT = 1.0
"""Against 2.0 on each velocity tracking term, so tracking still wins a conflict."""

COMMAND_RANGES = {
  "lin_vel_x": (0.0, 1.0),
  "lin_vel_y": (0.0, 0.0),
  "ang_vel_z": (0.0, 0.0),
}

HEAD_JOINTS = ("AAHead_yaw", "Head_pitch")


def t1_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The T1 flat velocity task with the style reward on top."""
  cfg = booster_t1_flat_env_cfg(play=play)
  cfg.scene.num_envs = 1 if play else 4096

  cfg.rewards["style"] = RewardTermCfg(
    func=motion_style,
    weight=STYLE_WEIGHT,
    params={
      "library_file": str(LIBRARY_FILE),
      "command_name": "twist",
      "joint_pos_std": 0.2,  # rad, rms over joints
      "joint_vel_std": 2.0,  # rad/s, rms over joints
      "lin_vel_std": 0.25,  # m/s
      "ang_vel_std": 0.4,  # rad/s
      "chunk_size": 1024,
    },
  )

  pose = cfg.rewards["pose"]
  pose.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=HEAD_JOINTS)
  for key in ("std_standing", "std_walking", "std_running"):
    pose.params[key] = {".*": 0.05}

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  for name, value in COMMAND_RANGES.items():
    setattr(twist.ranges, name, value)
  twist.heading_command = False
  twist.ranges.heading = None
  twist.rel_heading_envs = 0.0
  twist.rel_world_envs = 0.0
  twist.rel_forward_envs = 0.0
  cfg.curriculum.pop("command_vel", None)

  return cfg


def t1_walk_distance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Natural walking conditioned on distance, including braking and a settled stop"""
  cfg = t1_walk_env_cfg(play=play)
  # Mocap styles the legs while a separate posture reward keeps the arms by the body
  cfg.rewards["style"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=(".*Hip.*", ".*Knee.*", ".*Ankle.*", "Waist")
  )
  cfg.rewards["walk_arms"] = RewardTermCfg(
    func=mdp.variable_posture,
    weight=3.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*Shoulder.*", ".*Elbow.*")),
      "command_name": "twist",
      "std_standing": {".*": 0.1},
      "std_walking": {
        ".*Shoulder_Pitch": 0.2,
        ".*Shoulder_Roll": 0.12,
        ".*Elbow.*": 0.25,
      },
      "std_running": {".*": 0.25},
      "walking_threshold": 0.05,
      "running_threshold": 1.5,
    },
  )
  cfg.rewards["air_time"].weight = 0.5
  cfg.rewards["air_time"].params.update(
    threshold_min=0.08, threshold_max=0.35, command_threshold=0.05
  )
  for name in ("foot_clearance", "foot_swing_height"):
    cfg.rewards[name].params["target_height"] = 0.06
  cfg.rewards["foot_clearance"].weight = -4.0
  cfg.rewards["foot_swing_height"].weight = -1.0
  cfg.rewards["foot_slip"].weight = -0.5
  cfg.rewards["supported_swing"] = RewardTermCfg(
    func=supported_swing,
    weight=1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "foot_height_scan",
    },
  )
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  cfg.commands["twist"] = WalkDistanceCommandCfg(
    entity_name=twist.entity_name,
    ranges=twist.ranges,
    viz=twist.viz,
    resampling_time_range=(1.0e9, 1.0e9),
    heading_command=False,
    rel_heading_envs=0.0,
    rel_standing_envs=0.15,
    rel_world_envs=0.0,
    rel_forward_envs=0.0,
    init_velocity_prob=0.0,
    gui=False,
    debug_vis=play,
  )
  for group in cfg.observations.values():
    group.terms["walk_goal"] = ObservationTermCfg(func=walk_goal)
  cfg.rewards["distance_goal"] = RewardTermCfg(func=goal_reward, weight=4.0)
  cfg.rewards["distance_path"] = RewardTermCfg(func=path_reward, weight=2.0)
  cfg.rewards["termination"] = RewardTermCfg(func=envs_mdp.is_terminated, weight=-100.0)
  cfg.terminations["invalid_walk"] = TerminationTermCfg(func=invalid_walk)
  cfg.events.pop("push_robot", None)
  for axis in cfg.events["reset_base"].params["velocity_range"]:
    cfg.events["reset_base"].params["velocity_range"][axis] = (0.0, 0.0)
  cfg.episode_length_s = 20.0
  return cfg
