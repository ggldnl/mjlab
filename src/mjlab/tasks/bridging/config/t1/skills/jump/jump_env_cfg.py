"""Single trajectory and goal conditioned jump environments for the Booster T1.

The G1 jump_continuous environment with T1 bodies. Same recipe: ASAP style phase based
tracking with reference state initialization, early termination on lost tracking, every
clip sharing one policy with the goal in the observation and the reward, and a per
episode horizontal stretch so the reachable distances are continuous.

The command, observation, reward and termination terms are the G1 package's. None of them
names a body or a joint, so they run on the T1 unchanged. What differs is here:

    bodies          Trunk is the anchor and the root. Waist stands in for the G1 pelvis
    collisions      the T1 asset collides through its foot spheres only, so there is no
                    self collision sensor or penalty
    clips           the G1's five ASAP jumps, retargeted onto the T1 by dataset.py

Run

1. Retarget and convert the clips.

    uv run python -m mjlab.tasks.bridging.config.t1.skills.jump.dataset

2. Train.

    uv run train Mjlab-T1-Jump-Continuous --env.scene.num-envs 4096

3. Watch.

    uv run play Mjlab-T1-Jump-Continuous --checkpoint-file <path>
"""

from __future__ import annotations

from mjlab.asset_zoo.robots import T1_ACTION_SCALE, get_t1_robot_cfg
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
from mjlab.tasks.bridging.config.g1.skills.jump.dataset import CLIP_NAME
from mjlab.tasks.bridging.config.g1.skills.jump.jump_env_cfg import single_clip_env_cfg
from mjlab.tasks.bridging.config.g1.skills.jump_continuous import mdp
from mjlab.tasks.bridging.config.g1.skills.jump_continuous.jump_continuous_env_cfg import (
  LEAD_IN,
  POSE_RANGE,
  PUSH_VELOCITY_RANGE,
  VELOCITY_RANGE,
  deploy_env_cfg,
  student_group,
)
from mjlab.tasks.bridging.config.g1.skills.jump_continuous.mdp import (
  JumpCommandCfg,
)
from mjlab.tasks.bridging.config.g1.skills.jump_continuous.motion_lib import (
  discover_motion_files,
)
from mjlab.tasks.bridging.config.t1.skills.jump.dataset import DEFAULT_OUTPUT_DIR
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

##
# Body groups, the G1 ones mapped onto the T1.
##

FEET_BODIES: tuple[str, ...] = ("left_foot_link", "right_foot_link")

LOWER_BODIES: tuple[str, ...] = (
  "Waist",
  "Hip_Roll_Left",
  "Shank_Left",
  "left_foot_link",
  "Hip_Roll_Right",
  "Shank_Right",
  "right_foot_link",
)

UPPER_BODIES: tuple[str, ...] = (
  "Trunk",
  "AL2",
  "AL3",
  "left_hand_link",
  "AR2",
  "AR3",
  "right_hand_link",
)

# Trunk first, so the root is body zero of the tracked set as the pelvis is on the G1
TRACKED_BODIES: tuple[str, ...] = UPPER_BODIES[:1] + LOWER_BODIES + UPPER_BODIES[1:]

ANCHOR_BODY = "Trunk"

FOOT_GEOMS = r"^(left|right)_foot_sphere.*link$"

_STAGE_1 = 24_000
_STAGE_2 = 72_000


def t1_jump_env_cfg(
  clip: str | None = None, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Track retargeted level 3 from just before the lowest crouch to the original end."""
  cfg = t1_jump_continuous_env_cfg(
    motion_files=(clip or str(DEFAULT_OUTPUT_DIR / f"{CLIP_NAME}.npz"),), play=play
  )
  return single_clip_env_cfg(cfg, play=play)


def t1_jump_continuous_env_cfg(
  motion_files: tuple[str, ...] | None = None,
  play: bool = False,
  deploy: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the T1 jump environment.

  Args:
    motion_files: Converted npz clips. Defaults to every clip dataset.py wrote, ordered by
      distance.
    play: Start every episode at the entry landmark of a clip, drop the observation noise
      and the pushes, and leave the reference unperturbed.
    deploy: Build the deployment env instead, where a distance is the only input and no
      clip is loaded. See the G1 deploy_env_cfg. Implies play.
  """
  motion_files = motion_files or discover_motion_files(DEFAULT_OUTPUT_DIR)

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

  # Same three groups as the G1: "teacher" for the tracking phase, "actor" for the
  # distilled student every inference path loads, "critic" for both
  teacher_group = ObservationGroupCfg(
    terms=actor_terms, concatenate_terms=True, enable_corruption=True
  )
  observations = {
    "actor": student_group(teacher_group),
    "teacher": teacher_group,
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
      scale=T1_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "motion": JumpCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      motion_files=motion_files,
      anchor_body_name=ANCHOR_BODY,
      body_names=TRACKED_BODIES,
      pose_range=POSE_RANGE,
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.05, 0.05),
      # The G1's: five clips from 0.38 m to 1.59 m overlap into roughly 0.27 m to 2.07 m.
      # The G1's distances times 0.81, the ratio of the two leg lengths
      scale_range=(0.7, 1.3),
      goal_success_threshold=0.25,
      sampling_mode="adaptive",
      lead_in=LEAD_IN,
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
        "asset_cfg": SceneEntityCfg("robot", body_names=(ANCHOR_BODY,)),
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
        "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
        "operation": "abs",
        "ranges": (0.4, 1.2),
        "shared_random": True,
      },
    ),
  }

  ##
  # Rewards, the G1's less the self collision penalty
  ##

  rewards: dict[str, RewardTermCfg] = {
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
      params={"command_name": "motion", "std": 1.0},
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
    "jump_goal_pos": RewardTermCfg(
      func=mdp.jump_goal_position_error_exp,
      weight=2.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "jump_goal_success": RewardTermCfg(
      func=mdp.jump_goal_success,
      weight=20.0,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
    "joint_torques": RewardTermCfg(func=mdp.joint_torques_l2, weight=-2.0e-7),
    "joint_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-2.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "feet_orientation": RewardTermCfg(
      func=mdp.feet_orientation_penalty,
      weight=-0.2,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODIES)},
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
    # Finishing the jump is success, not failure. This must stay time_out=True
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
    "motion_far_threshold": CurriculumTermCfg(
      func=mdp.termination_curriculum,
      params={
        "termination_name": "motion_far",
        "stages": [
          {"step": 0, "params": {"threshold": 1.0}},
          {"step": _STAGE_1, "params": {"threshold": 0.6}},
          {"step": _STAGE_2, "params": {"threshold": 0.4}},
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
    "feet_penalties": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "feet_slip",
        "stages": [
          {"step": 0, "weight": -0.2},
          {"step": _STAGE_1, "weight": -1.0},
        ],
      },
    ),
    "feet_orientation_penalty": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "feet_orientation",
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
      pattern=r"^(left_foot_link|right_foot_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  scene = SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    num_envs=1,
    entities={"robot": get_t1_robot_cfg()},
    sensors=(feet_ground_cfg,),
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
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      # Ten contact spheres per foot, so twice the G1's budget
      nconmax=70,
      njmax=300,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    # 0.005 * 4 gives 50 Hz control, the rate the clips are converted at
    decimation=4,
    # Longer than the longest clip, which ends the episode first
    episode_length_s=6.0,
  )

  if play or deploy:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["teacher"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, JumpCommandCfg)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.sampling_mode = "entry"

  if deploy:
    # The stretch carries the clips out to these, see scale_range
    return deploy_env_cfg(cfg, distance_range=(0.3, 1.9))

  return cfg
