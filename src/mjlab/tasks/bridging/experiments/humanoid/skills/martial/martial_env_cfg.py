"""The martial arts environment: track one crop of a LAFAN1 fight performance.

Same recipe as the jump: track a human clip frame by frame, start episodes anywhere in it,
and terminate the moment tracking is lost. The clip is a strike instead of a jump, and it
begins with half a second of held stance, so what the policy learns is to stand still and
then break out of that stance fast enough to follow the reference.

This module builds the environment for every motion in the package. One motion is one clip
and one policy, so nothing here is conditioned on which motion it is: each task is the same
environment pointed at a different motion directory.

Two things are dropped from the jump's setup, both because they are about flight and these
motions have none.

    goal terms         a jump is asked to land somewhere and is paid for it. A strike goes
                       nowhere, so the only thing to say about where it ends up is that it
                       should not have drifted, and the anchor position term already says
                       that at every frame
    feet orientation   the penalty asks for level feet at all times, a fair prior for a
                       jump and wrong here: a kicking foot points where the kick goes. The
                       reference already says where both feet should be, and an
                       unconditional prior can only argue with it

What is raised is the body velocity term. A strike that hits the right poses slowly is not
a strike, and the poses alone do not distinguish the two.

Run

1. Convert the clips.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.martial.dataset

2. Train.

    uv run train Mjlab-G1-Front-Kick --env.scene.num-envs 4096

3. Watch.

    uv run play Mjlab-G1-Front-Kick --checkpoint-file <path>
"""

from __future__ import annotations

from pathlib import Path

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
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous import mdp
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.mdp import (
  JumpCommandCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.motion_lib import (
  discover_motion_files,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

##
# Body groups. A strike is decided by what the hands and feet do, so they get their own
# group and the tightest tolerance in the reward set
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

# Both hands and both feet, whatever the motion. Which of the four is the striking limb is
# something the reference already says, frame by frame, so there is nothing to gain by
# naming it here and a per motion config to maintain if it were
STRIKE_BODIES: tuple[str, ...] = (
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
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

# Reference-state initialization noise, the same small perturbation the jump uses. A strike
# spends its whole length on one or two feet, so there is not much margin to spend here
POSE_RANGE = {
  "x": (-0.03, 0.03),
  "y": (-0.03, 0.03),
  "z": (-0.01, 0.01),
  "roll": (-0.05, 0.05),
  "pitch": (-0.05, 0.05),
  "yaw": (-0.1, 0.1),
}

VELOCITY_RANGE = {
  "x": (-0.2, 0.2),
  "y": (-0.2, 0.2),
  "z": (-0.1, 0.1),
  "roll": (-0.2, 0.2),
  "pitch": (-0.2, 0.2),
  "yaw": (-0.3, 0.3),
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


def g1_martial_env_cfg(
  motion_dir: Path,
  motion_files: tuple[str, ...] | None = None,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build a martial arts environment around one converted clip.

  Args:
    motion_dir: Where dataset.py wrote that motion's npz.
    motion_files: Converted npz clips. Defaults to everything in motion_dir, which for
      these tasks is the one clip.
    play: Start every episode at the beginning of the clip, which is the standstill, drop
      the observation noise and the pushes, and leave the reference unperturbed.
  """
  motion_files = motion_files or discover_motion_files(motion_dir)

  ##
  # Observations
  ##

  actor_terms = {
    # Reference joint targets and phase. There is one clip, so there is nothing to say
    # about which strike this is
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
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
    "phase": ObservationTermCfg(func=mdp.jump_phase, params={"command_name": "motion"}),
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
    # The jump's command is a multi-clip phase tracker with a goal bolted on. Only the
    # tracker is used here: the scale is pinned at one, because stretching a strike
    # horizontally is not a longer strike, and the goal dial it puts in the viewer is
    # turned off with it
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
      sampling_mode="adaptive",
    )
  }

  ##
  # Events
  ##

  events: dict[str, EventTermCfg] = {
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(2.0, 4.0),
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
    # Where the robot is in the world. For a strike this is the term that says stay put
    "motion_anchor_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_anchor_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    # Posture
    "motion_body_pos_lower": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.25, "body_names": LOWER_BODIES},
    ),
    "motion_body_pos_strike": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.5,
      params={"command_name": "motion", "std": 0.15, "body_names": STRIKE_BODIES},
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
    # Twice the jump's weight, and a wider tolerance to go with it. A punch peaks around
    # 5 m/s at the wrist, so the jump's std of 1.0 saturates to zero over the strike and
    # pays the same whether the limb moved or not
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 2.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 3.14},
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
    # Regularizers. Weights start low and the curriculum raises them, so a policy that
    # cannot yet track is not taught that falling over early is the cheap way out
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
    "joint_torques": RewardTermCfg(func=mdp.joint_torques_l2, weight=-2.0e-7),
    "joint_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-2.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    # A tenth of the jump's weight. The support foot pivots under these strikes, and that
    # pivot is contact with a turning foot, which is what this measures
    "feet_slip": RewardTermCfg(
      func=mdp.feet_slip_penalty,
      weight=-0.02,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODIES),
      },
    ),
    "termination": RewardTermCfg(func=mdp.is_terminated, weight=-100.0),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # Finishing the strike is success, not failure. This must stay time_out=True
    "motion_ended": TerminationTermCfg(
      func=mdp.motion_ended, params={"command_name": "motion"}, time_out=True
    ),
    "motion_far": TerminationTermCfg(
      func=mdp.motion_too_far,
      params={"command_name": "motion", "threshold": 1.0},
    ),
    "anchor_height": TerminationTermCfg(
      func=mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.3},
    ),
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
    # Ends tighter than the jump's 0.4 m. Nothing here is airborne, so there is no phase
    # where a large tracking error is the reference's fault rather than the policy's
    "motion_far_threshold": CurriculumTermCfg(
      func=mdp.termination_curriculum,
      params={
        "termination_name": "motion_far",
        "stages": [
          {"step": 0, "params": {"threshold": 1.0}},
          {"step": _STAGE_1, "params": {"threshold": 0.5}},
          {"step": _STAGE_2, "params": {"threshold": 0.35}},
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
          {"step": 0, "weight": -0.02},
          {"step": _STAGE_1, "weight": -0.1},
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
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=ANCHOR_BODY,
      distance=3.0,
      fovy=55.0,
      elevation=-5.0,
      azimuth=140.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=250,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    # 0.005 * 4 gives 50 Hz control, the rate the clips are converted at
    decimation=4,
    # Long enough for any of the clips plus its held opening. The motion ends the episode
    # before this
    episode_length_s=6.0,
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
