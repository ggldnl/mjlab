"""Privileged trajectory-conditioned PPO oracle."""

from dataclasses import replace

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.oracle import mdp
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.oracle.command import (
  MotionSetCommandCfg,
)
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

DEFAULT_MOTIONS = ("data/lafan1_g1/motions/*.npz",)
DEFAULT_MOTION_FILE = "data/lafan1_g1/motions/dance1_subject1.npz"
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def oracle_env_cfg(
  play: bool = False, motion_files: tuple[str, ...] = DEFAULT_MOTIONS
) -> ManagerBasedRlEnvCfg:
  """Build the universal G1 tracking oracle environment."""
  cfg = unitree_g1_flat_tracking_env_cfg(play=False)
  base_motion = cfg.commands["motion"]
  if not isinstance(base_motion, MotionCommandCfg):
    raise TypeError("G1 tracking configuration has no motion command")
  motion = MotionSetCommandCfg(
    entity_name=base_motion.entity_name,
    resampling_time_range=base_motion.resampling_time_range,
    debug_vis=play,
    gui=base_motion.gui,
    motion_file=DEFAULT_MOTION_FILE,
    motion_files=motion_files,
    anchor_body_name=base_motion.anchor_body_name,
    body_names=base_motion.body_names,
    foot_body_names=FEET,
    pose_range={} if play else base_motion.pose_range,
    velocity_range={} if play else base_motion.velocity_range,
    joint_position_range=(0.0, 0.0) if play else base_motion.joint_position_range,
    sampling_mode="uniform",
  )
  cfg.commands["motion"] = motion
  action = cfg.actions["joint_pos"]
  if not isinstance(action, JointPositionActionCfg):
    raise TypeError("G1 tracking configuration has no joint position action")
  cfg.actions["joint_pos"] = replace(
    action,
    actuator_names=(r"^(?!.*wrist).*$",),
    scale={
      name: value for name, value in G1_ACTION_SCALE.items() if "wrist" not in name
    },
  )

  privileged_terms = {
    "proprioception": ObservationTermCfg(
      func=mdp.oracle_proprioception, params={"command_name": "motion"}
    ),
    "goal": ObservationTermCfg(func=mdp.oracle_goal, params={"command_name": "motion"}),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=privileged_terms, concatenate_terms=True, enable_corruption=False
    ),
    "critic": ObservationGroupCfg(
      terms=privileged_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  cfg.events.pop("push_robot", None)
  cfg.events.pop("encoder_bias", None)
  if play:
    cfg.events = {}
    cfg.episode_length_s = int(1e9)

  def tracking(func, weight: float, tight: float, broad: float) -> RewardTermCfg:
    return RewardTermCfg(
      func=func,
      weight=weight,
      params={
        "command_name": "motion",
        "tight": tight,
        "broad": broad,
      },
    )

  regularizers = {
    name: cfg.rewards[name]
    for name in ("action_rate_l2", "joint_limit", "self_collisions")
  }
  regularizers["action_rate_l2"].weight = -0.05
  cfg.rewards = {
    "motion_root_pos": tracking(mdp.root_position_tracking, 2.0, 0.05, 0.30),
    "motion_root_ori": tracking(mdp.root_orientation_tracking, 1.0, 0.05, 0.40),
    "motion_root_lin_vel": tracking(mdp.root_linear_velocity_tracking, 1.0, 0.15, 1.0),
    "motion_root_ang_vel": tracking(
      mdp.root_angular_velocity_tracking, 0.75, 0.30, 3.14
    ),
    "motion_body_pos": tracking(mdp.body_position_tracking, 1.6, 0.05, 0.30),
    "motion_body_ori": tracking(mdp.body_orientation_tracking, 0.5, 0.10, 0.40),
    "motion_body_lin_vel": tracking(mdp.body_linear_velocity_tracking, 0.5, 0.30, 1.0),
    "motion_body_ang_vel": tracking(
      mdp.body_angular_velocity_tracking, 0.5, 0.60, 3.14
    ),
    "motion_lower_joint_pos": tracking(mdp.joint_position_tracking, 0.5, 0.08, 0.80),
    "motion_lower_joint_vel": tracking(mdp.joint_velocity_tracking, 0.4, 0.80, 8.0),
    "motion_upper_joint_pos": tracking(mdp.joint_position_tracking, 0.5, 0.05, 0.50),
    "motion_upper_joint_vel": tracking(mdp.joint_velocity_tracking, 0.4, 0.75, 7.5),
    "motion_foot_pos": tracking(mdp.foot_position_tracking, 3.0, 0.05, 0.25),
    "motion_foot_ori": tracking(mdp.foot_orientation_tracking, 1.0, 0.08, 0.40),
    "motion_foot_lin_vel": tracking(mdp.foot_linear_velocity_tracking, 1.0, 0.20, 1.0),
    "motion_foot_ang_vel": tracking(
      mdp.foot_angular_velocity_tracking, 0.75, 0.40, 3.0
    ),
    **regularizers,
  }
  for name in ("motion_lower_joint_pos", "motion_lower_joint_vel"):
    cfg.rewards[name].params["upper"] = False
  for name in ("motion_upper_joint_pos", "motion_upper_joint_vel"):
    cfg.rewards[name].params["upper"] = True

  end_effectors = cfg.terminations["ee_body_pos"].params["body_names"]
  cfg.terminations.update(
    {
      "anchor_pos": TerminationTermCfg(
        func=mdp.bad_goal_anchor_pos_z,
        params={"command_name": "motion", "threshold": 0.25},
      ),
      "anchor_ori": TerminationTermCfg(
        func=mdp.bad_goal_anchor_ori,
        params={"command_name": "motion", "threshold": 0.8},
      ),
      "ee_body_pos": TerminationTermCfg(
        func=mdp.bad_goal_body_pos_z,
        params={
          "command_name": "motion",
          "threshold": 0.25,
          "body_names": end_effectors,
        },
      ),
    }
  )
  return cfg
