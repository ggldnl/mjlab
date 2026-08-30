"""The backflip environment.

Run:

    uv run python -m \
      mjlab.tasks.bridging.experiments.humanoid.skills.backflip.dataset
    uv run train Mjlab-G1-Backflip --env.scene.num-envs 4096
    uv run play Mjlab-G1-Backflip --checkpoint-file <path>

The jump's setup with the clip swapped and three things moved, all for the same reason: this
motion spends half a second upside down.

  The feet orientation penalty is gone. It asks for level feet at all times, which is a fair
  prior for a jump and an argument against the reference here, where the feet are over the
  head at the top of the turn.

  The tracking failure threshold is looser and stays looser. A body in flight is ballistic,
  so being a few frames early or late on the takeoff shows up as tens of centimetres of
  position error with nothing the policy can do about it mid-air. Tightening this the way the
  jump does would end every airborne episode before the turn.

  The angular velocity tolerance is wider. The reference turns at about eleven radians a
  second, so the jump's tolerance calls any policy that is not already flipping equally
  wrong, and a reward that flat is a reward with no gradient.

The goal terms are kept as they are. A backflip has a landing point, roughly a third of a
metre behind where it started, and landing there rather than somewhere else is a fair part
of what the skill is.
"""

from __future__ import annotations

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.backflip.dataset import MOTION_DIR
from mjlab.tasks.bridging.experiments.humanoid.skills.jump import mdp
from mjlab.tasks.bridging.experiments.humanoid.skills.jump.mdp import JumpCommandCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.jump.motion_lib import (
  discover_motion_files,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

##
# Body groups. Same split as the jump: the legs decide the takeoff and the feet decide
# whether the landing holds
##

TRACKED_BODIES: tuple[str, ...] = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)

FEET_BODIES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")

LOWER_BODIES: tuple[str, ...] = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
)

UPPER_BODIES: tuple[str, ...] = (
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)

ANCHOR_BODY = "torso_link"

# Reference-state initialization noise. Smaller than the jump's on the rotational axes: an
# environment dropped into the flip mid-turn is already asked to match an angular velocity
# of eleven radians a second, and noise on top of that is not a harder version of the same
# problem, it is a different one
POSE_RANGE = {
  "x": (-0.03, 0.03),
  "y": (-0.03, 0.03),
  "z": (-0.01, 0.01),
  "roll": (-0.03, 0.03),
  "pitch": (-0.03, 0.03),
  "yaw": (-0.05, 0.05),
}

VELOCITY_RANGE = {
  "x": (-0.15, 0.15),
  "y": (-0.15, 0.15),
  "z": (-0.1, 0.1),
  "roll": (-0.15, 0.15),
  "pitch": (-0.15, 0.15),
  "yaw": (-0.2, 0.2),
}

PUSH_VELOCITY_RANGE = {
  "x": (-0.3, 0.3),
  "y": (-0.3, 0.3),
  "z": (-0.1, 0.1),
  "roll": (-0.3, 0.3),
  "pitch": (-0.3, 0.3),
  "yaw": (-0.4, 0.4),
}

# Curriculum thresholds are in environment steps. At 24 steps per env per iteration,
# 24_000 steps is about 1000 iterations
_STAGE_1 = 24_000
_STAGE_2 = 72_000


def g1_backflip_env_cfg(
  motion_files: tuple[str, ...] | None = None,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the backflip environment.

  Args:
    motion_files: Converted npz clips. Defaults to whatever dataset.py wrote, which is one
      flip unless it was run more than once with different arguments.
    play: Start every episode at the beginning of the clip, which is the standstill, drop
      the observation noise and the pushes, and leave the reference unperturbed.
  """
  motion_files = motion_files or discover_motion_files(MOTION_DIR)

  ##
  # Observations
  ##

  actor_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "goal": ObservationTermCfg(func=mdp.jump_goal_b, params={"command_name": "motion"}),
    "phase": ObservationTermCfg(func=mdp.jump_phase, params={"command_name": "motion"}),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.3, n_max=0.3),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "goal": ObservationTermCfg(func=mdp.jump_goal_b, params={"command_name": "motion"}),
    "phase": ObservationTermCfg(func=mdp.jump_phase, params={"command_name": "motion"}),
    "landed": ObservationTermCfg(
      func=mdp.jump_airborne, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b, params={"command_name": "motion"}
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b, params={"command_name": "motion"}
    ),
    "body_pos_error": ObservationTermCfg(
      func=mdp.motion_body_pos_error_b, params={"command_name": "motion"}
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=G1_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    # The jump's tracker, with the scale pinned at one. Stretching is how the jump turns
    # five clips into a continuous range of distances; a flip stretched horizontally is a
    # flip that lands somewhere its takeoff could not have put it. The viewer's distance
    # dial goes with the stretching
    "motion": JumpCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      gui=False,
      motion_files=motion_files,
      anchor_body_name=ANCHOR_BODY,
      body_names=TRACKED_BODIES,
      pose_range=POSE_RANGE,
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.05, 0.05),
      scale_range=(1.0, 1.0),
      goal_success_threshold=0.3,
      sampling_mode="adaptive",
    )
  }

  ##
  # Events
  ##

  events: dict[str, EventTermCfg] = {
    # Rarer than the jump's. The clip is a little over two seconds, so at the jump's rate
    # most episodes would take a push, and a push landing inside the flight is a tracking
    # failure the policy had no way to prevent
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(4.0, 8.0),
      params={"velocity_range": PUSH_VELOCITY_RANGE},
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "operation": "add",
        "ranges": {0: (-0.02, 0.02), 1: (-0.03, 0.03), 2: (-0.03, 0.03)},
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=r"^(left|right)_foot[1-7]_collision$"
        ),
        "operation": "abs",
        "ranges": (0.4, 1.2),
        "shared_random": True,
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards: dict[str, RewardTermCfg] = {
    "motion_anchor_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    # The turn itself. Weighted above the jump's, because this is the one quantity that
    # separates a flip from a vertical hop with the same trajectory
    "motion_anchor_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.5},
    ),
    "motion_body_pos_lower": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.25, "body_names": LOWER_BODIES},
    ),
    "motion_body_pos_feet": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.5,
      params={"command_name": "motion", "std": 0.15, "body_names": FEET_BODIES},
    ),
    "motion_body_pos_upper": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.35, "body_names": UPPER_BODIES},
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 1.5},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 6.0},
    ),
    "motion_joint_pos": RewardTermCfg(
      func=mdp.motion_joint_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.5},
    ),
    "motion_joint_vel": RewardTermCfg(
      func=mdp.motion_joint_velocity_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 5.0},
    ),
    # Where the flip has to come down
    "flip_goal_pos": RewardTermCfg(
      func=mdp.jump_goal_position_error_exp,
      weight=2.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "flip_goal_success": RewardTermCfg(
      func=mdp.jump_goal_success,
      weight=20.0,
      params={"command_name": "motion", "threshold": 0.3},
    ),
    # Regularizers
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
    "joint_torques": RewardTermCfg(func=mdp.joint_torques_l2, weight=-2.0e-7),
    "joint_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-2.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    # Half the jump's weight. The tuck brings the thighs onto the torso by design, so some
    # of what this measures is the reference being followed rather than a fault
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-0.5,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    "feet_slip": RewardTermCfg(
      func=mdp.feet_slip_penalty,
      weight=-0.2,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODIES),
      },
    ),
    "landing_impact": RewardTermCfg(
      func=mdp.landing_impact_penalty,
      weight=-2.0e-4,
      params={"sensor_name": "feet_ground_contact", "force_threshold": 500.0},
    ),
    "termination": RewardTermCfg(func=mdp.is_terminated, weight=-100.0),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # Finishing the flip is success, not failure. This must stay time_out=True
    "motion_ended": TerminationTermCfg(
      func=mdp.motion_ended, params={"command_name": "motion"}, time_out=True
    ),
    "motion_far": TerminationTermCfg(
      func=mdp.motion_too_far,
      params={"command_name": "motion", "threshold": 1.2},
    ),
    "anchor_height": TerminationTermCfg(
      func=mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.4},
    ),
    # Measured against the reference, not against upright, so being upside down at the top
    # of the turn is not a failure. Being upside down when the reference is not, is
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
  }

  ##
  # Curriculum
  ##

  curriculum: dict[str, CurriculumTermCfg] = {
    "motion_far_threshold": CurriculumTermCfg(
      func=mdp.termination_curriculum,
      params={
        "termination_name": "motion_far",
        "stages": [
          {"step": 0, "params": {"threshold": 1.2}},
          {"step": _STAGE_1, "params": {"threshold": 0.8}},
          {"step": _STAGE_2, "params": {"threshold": 0.6}},
        ],
      },
    ),
    "action_rate_penalty": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "action_rate",
        "stages": [
          {"step": 0, "weight": -0.02},
          {"step": _STAGE_1, "weight": -0.1},
          {"step": _STAGE_2, "weight": -0.2},
        ],
      },
    ),
    "joint_limits_penalty": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "joint_limits",
        "stages": [
          {"step": 0, "weight": -2.0},
          {"step": _STAGE_1, "weight": -10.0},
        ],
      },
    ),
    "feet_slip_penalty": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "feet_slip",
        "stages": [
          {"step": 0, "weight": -0.2},
          {"step": _STAGE_1, "weight": -1.0},
        ],
      },
    ),
  }

  ##
  # Scene
  ##

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  scene = SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    num_envs=1,
    entities={"robot": get_g1_robot_cfg()},
    sensors=(feet_ground_cfg, self_collision_cfg),
  )

  cfg = ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    # From the side, which is the only angle a flip reads from
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=ANCHOR_BODY,
      distance=3.5,
      fovy=55.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=250,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    # 0.005 * 4 gives 50 Hz control, the rate the clip is built at
    decimation=4,
    # The clip is about two and a quarter seconds. The motion ends the episode before this
    episode_length_s=4.0,
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, JumpCommandCfg)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.sampling_mode = "start"

  return cfg
