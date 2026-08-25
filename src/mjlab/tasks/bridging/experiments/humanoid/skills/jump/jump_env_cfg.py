"""The goal-conditioned jump environment.

This is ASAP's stage-one recipe, phase-based motion tracking, put together out of
mjlab parts. The claim ASAP makes, and the reason this is worth porting, is narrow
and important: a humanoid learns a jump by tracking a jump, frame by frame, with
reference-state initialization and an early termination that fires the moment
tracking is lost. There is no style discriminator, no hand-designed jump reward, no
standing-first curriculum. The reference supplies the shape of the motion and the
policy only has to make it physical.

What the earlier attempts in this folder got wrong, in those terms:

  A style reward plus a goal reward asks the policy to discover the jump and to
  imitate a distribution at the same time; neither signal is dense enough to get
  off the ground. Tracking replaces both with a per-frame target.

  Curriculum from standing spends its early budget learning to stand, which is a
  local optimum the jump has to be pushed out of. Reference-state initialization
  sidesteps it entirely: a third of the environments start mid-flight, so the
  policy sees the airborne part of the task from the first iteration.

The goal conditioning is the one thing added on top of ASAP, which trains a policy
per clip. Five forward jumps of different lengths share a policy here, the goal is
in the observation and in the reward, and each episode stretches its clip a little
so the reachable distances are continuous rather than five discrete points. See
`mdp/commands.py` for how the stretching works.

Run it:

    uv run --with joblib python -m mjlab.tasks.bridging.experiments.humanoid.skills.jump.dataset
    uv run train Mjlab-Parkour-Jump --env.scene.num-envs 4096
    uv run play Mjlab-Parkour-Jump --checkpoint-file <path>
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
from mjlab.tasks.bridging.experiments.humanoid.skills.jump import mdp
from mjlab.tasks.bridging.experiments.humanoid.skills.jump.mdp import JumpCommandCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.jump.motion_lib import (
  discover_motion_files,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

##
# Body groups. The reward set weights these differently, because a jump is decided
# by the legs and the landing is decided by the feet.
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

# Reference-state initialization noise. Deliberately small: the reference is a
# ballistic trajectory, and a large perturbation at takeoff is not a harder version
# of the same problem, it is a different and unreachable one.
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

# Curriculum thresholds are in environment steps, not iterations. With the default
# 24 steps per environment per iteration, 24_000 steps is about 1000 iterations.
_STAGE_1 = 24_000
_STAGE_2 = 72_000


def g1_jump_env_cfg(
  motion_files: tuple[str, ...] | None = None,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the jump environment.

  Args:
    motion_files: Converted npz clips. Defaults to every `jump_forward_level*` file
      in the task's `motions/` directory, ordered by distance.
    play: Start every episode at the beginning of a clip, drop the observation
      noise and the pushes, and leave the reference unperturbed.
  """
  motion_files = motion_files or discover_motion_files()

  ##
  # Observations
  ##

  actor_terms = {
    # Reference joint targets, phase and goal descriptor, in one tensor.
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
    "motion": JumpCommandCfg(
      entity_name="robot",
      # The clip's own length ends the episode, so the resampling clock never fires.
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      motion_files=motion_files,
      anchor_body_name=ANCHOR_BODY,
      body_names=TRACKED_BODIES,
      pose_range=POSE_RANGE,
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.05, 0.05),
      # Wide enough that the five clips overlap into a nearly continuous range of
      # distances (roughly 0.33 m to 2.56 m, with a small gap around 0.7 m where
      # the two shortest clips do not quite meet). Wider still and the reference
      # stops being a jump the recorded takeoff pose could produce.
      scale_range=(0.7, 1.3),
      goal_success_threshold=0.25,
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
    # Where the robot is in the world. This is what makes the jump go somewhere:
    # the relative body terms below are blind to global drift by construction.
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
    # Posture, split the way ASAP splits it.
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
    # The goal, which is what the user actually asked for.
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
    # Regularizers. Weights start low and are raised by the curriculum: applied at
    # full strength from step zero they dominate a policy that cannot yet track,
    # and the cheapest way to stop being penalized is to fall over early.
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
    # Losing tracking has to cost more than the reward left on the table, or
    # terminating early becomes the cheapest way out of a hard clip.
    "termination": RewardTermCfg(func=mdp.is_terminated, weight=-100.0),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # Finishing the jump is success, not failure. This must stay time_out=True.
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
    # Tighten what counts as tracking. ASAP drives this off the running episode
    # length; a step schedule is the same idea with one less thing to tune.
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
      distance=3.5,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=250,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    # 0.005 * 4 gives 50 Hz control, which is the rate the clips are converted at.
    decimation=4,
    # Long enough for the longest clip; the motion ends the episode before this.
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
