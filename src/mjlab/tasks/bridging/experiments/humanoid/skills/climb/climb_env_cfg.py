"""The climbing environment: track one OmniRetarget climb, up the box and down the far side.

The martial recipe against a clip that has an obstacle in it: track a human motion frame by
frame, start episodes anywhere in it, terminate the moment tracking is lost. Two things are
different, and both come from the box being real.

    the box is in the scene   a static, immovable block, placed where the converter measured
                              it. The reference climbs it, so it has to be there or the
                              reference is asking for a stand on thin air
    contacts are not only     the feet spend a third of the clip on the box and the hands
    with the floor            take load on the edge, so the foot sensor counts any contact
                              rather than contact with the terrain

One clip is one policy. The clip covers getting on and getting off, because the source
motion does: the subject walks in, climbs, crosses the top and steps down the far side, and
cutting that in half would leave a skill that ends standing on an obstacle.

The approach walk is cut out in the converter. The controller drives the robot up to the
box with the walk and hands over facing it, so the reference opens with the robot standing
still a quarter of a metre from the near face. What varies about that hand-over is the
angle it arrives at, and that lives in APPROACH_YAW_RANGE below.

What to watch:

    Episode/rew_motion_body_pos_contact   the hands and feet, which is what lands on the box
    Episode/rew_motion_anchor_pos         where the robot is against where it should be
    Episode/termination                   how often it is falling off
    Curriculum/motion_far_threshold       whether the tracking window has tightened yet

A climb that fails looks like a policy that tracks the approach, misses the plant and then
terminates on anchor height every episode at the same phase. rew_motion_body_pos_contact
flat while rew_motion_anchor_pos climbs means the robot is going to the right place without
putting its hands anywhere useful, and the fix is a larger contact weight rather than a
tighter anchor.

Run

1. Convert the clip. Writes to data/omniretarget/clips/climb.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.climb.dataset

2. Train.

    uv run train Mjlab-G1-Climb --env.scene.num-envs 4096

3. Watch.

    uv run play Mjlab-G1-Climb
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from mjlab.asset_zoo.objects.box import get_box_cfg
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.entity import EntityCfg
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
from mjlab.tasks.bridging.experiments.humanoid.skills.climb import mdp
from mjlab.tasks.bridging.experiments.humanoid.skills.climb.dataset import (
  Box,
  box_from_manifest,
)
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
# Body groups. A climb is decided by what takes load on the box, so the hands and feet get
# their own group and the tightest tolerance in the reward set
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

# The four limbs that land on the box. Which of them is bearing weight at any moment is
# something the reference already says frame by frame, so there is nothing to gain by
# naming a leading limb here
CONTACT_BODIES: tuple[str, ...] = (
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

BOX_COLOR: tuple[float, float, float, float] = (0.62, 0.64, 0.67, 1.0)
"""Concrete grey, the colour the parkour demo's arena draws a box in. Restated rather than
imported, because the arena imports the skills and not the other way round."""

APPROACH_YAW_RANGE: tuple[float, float] = (-0.15, 0.15)
"""How far off the reference's heading an episode may start, in radians.

The approach angle, and the only place it lives. The box and the reference are one rigid
thing and neither moves: what is perturbed is where the robot is spawned relative to them,
because every reference observation is the reference expressed in the robot's own frame and
rotating the whole scene together leaves all of them unchanged.

Plus or minus 0.15 is about eight degrees, which the tracker can walk off during the half
second of held stance and the stride into the box. Wider does not work by tracking alone:
the reference plants a hand at a fixed spot on the edge, and past roughly ten degrees the
motion that reaches it is a different motion, not this one started crooked. Re-retargeting
against a turned box is what buys more, not a wider range here."""

POSE_RANGE = {
  "x": (-0.05, 0.05),
  "y": (-0.05, 0.05),
  "z": (-0.01, 0.01),
  "roll": (-0.05, 0.05),
  "pitch": (-0.05, 0.05),
  "yaw": APPROACH_YAW_RANGE,
}
"""Reference state initialization noise. Wider in x and y than the martial motions, which
hold a stance and need the margin they have; a climb opens standing in front of a box with
room to be a few centimetres off."""

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


def box_entity(box: Box) -> EntityCfg:
  """The obstacle, at the pose the converter measured for the clip.

  Static, so it has no freejoint and mjlab wraps it as a mocap body: it collides like a wall
  and physics never moves it. The pose still has to be written at reset, which is what
  ``reset_box`` below is for, or every environment's box stacks at the world origin.
  """
  cfg = get_box_cfg(half_size=box.half_size, color=BOX_COLOR)
  half_yaw = box.yaw / 2.0
  return replace(
    cfg,
    init_state=EntityCfg.InitialStateCfg(
      pos=box.pos,
      rot=(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
      joint_pos={},
    ),
  )


def g1_climb_env_cfg(
  motion_dir: Path,
  motion_files: tuple[str, ...] | None = None,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build a climbing environment around one converted clip and its box.

  Args:
    motion_dir: Where dataset.py wrote the motion's npz and manifest.
    motion_files: Converted npz clips. Defaults to everything in motion_dir, which for this
      task is the one clip.
    play: Start every episode at the beginning of the clip, which is the standstill in front
      of the box, drop the observation noise and the pushes, and leave the reference
      unperturbed.
  """
  motion_files = motion_files or discover_motion_files(motion_dir)
  box = box_from_manifest(motion_dir)

  ##
  # Observations
  ##

  actor_terms = {
    # Reference joint targets and phase. There is one clip, so there is nothing to say
    # about which climb this is
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
    "box_pose": ObservationTermCfg(
      func=mdp.box_pose_b,
      params={"half_size": box.half_size},
      noise=Unoise(n_min=-0.02, n_max=0.02),
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
    "box_pose": ObservationTermCfg(
      func=mdp.box_pose_b, params={"half_size": box.half_size}
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
    # The jump's tracker, with its goal turned off. The scale is pinned at one: stretching a
    # climb horizontally would move the robot off the box it is climbing, which is the one
    # augmentation this clip cannot take
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
    # Puts the box where the manifest says, per environment. Empty ranges because the box
    # is where the reference needs it and nowhere else: what varies is the robot's spawn,
    # and the command term's pose_range owns that
    "reset_box": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {}, "asset_cfg": SceneEntityCfg("box")},
    ),
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
    # The foot geom carries the higher contact priority, so its friction is the one that
    # decides both surfaces. Randomising it covers standing on the box as well as the floor,
    # and randomising the box's own would do nothing at its default priority
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
    # Where the robot is in the world. For a climb this is the term that says the robot got
    # onto the box rather than up to it
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
    # The tightest tolerance in the set, on the four limbs that take load. A hand six
    # centimetres off the edge is a hand that misses the edge, and no other term here can
    # tell that apart from a hand six centimetres off in free space
    "motion_body_pos_contact": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.5,
      params={"command_name": "motion", "std": 0.12, "body_names": CONTACT_BODIES},
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
    # The jump's weight and tolerance, not the martial arts one. A climb is slow: what makes
    # it a climb is the sequence of places the limbs go, and the strike terms were widened
    # for a wrist moving at five metres a second, which nothing here does
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
    # Regularizers. Weights start low and the curriculum raises them, so a policy that
    # cannot yet track is not taught that falling off early is the cheap way out
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
    # A tenth of the jump's weight, as for the martial motions. A foot planted on a box edge
    # rolls as the robot pulls over it, and that roll is contact with a turning foot, which
    # is what this measures
    "feet_slip": RewardTermCfg(
      func=mdp.feet_slip_penalty,
      weight=-0.02,
      params={
        "sensor_name": "feet_contact",
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
    # Reaching the far side is success, not failure. This must stay time_out=True
    "motion_ended": TerminationTermCfg(
      func=mdp.motion_ended, params={"command_name": "motion"}, time_out=True
    ),
    "motion_far": TerminationTermCfg(
      func=mdp.motion_too_far,
      params={"command_name": "motion", "threshold": 1.0},
    ),
    # Against the reference, not against the ground, so this reads the same whether the
    # reference is on the floor or on top of the box
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
    # Ends wider than the martial motions' 0.35. Those stay on one spot, while this one
    # travels two metres over an obstacle, and the frames on the edge are the ones where a
    # centimetre of reference error is the retargeting's fault rather than the policy's
    "motion_far_threshold": CurriculumTermCfg(
      func=mdp.termination_curriculum,
      params={
        "termination_name": "motion_far",
        "stages": [
          {"step": 0, "params": {"threshold": 1.0}},
          {"step": _STAGE_1, "params": {"threshold": 0.6}},
          {"step": _STAGE_2, "params": {"threshold": 0.45}},
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

  # Any contact, not contact with the terrain. Half of what this skill does happens on top
  # of the box, and a foot sensor scoped to the floor would go quiet exactly when the
  # interesting contacts start
  feet_contact_cfg = ContactSensorCfg(
    name="feet_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=None,
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
    entities={"robot": get_g1_robot_cfg(), "box": box_entity(box)},
    sensors=(feet_contact_cfg, self_collision_cfg),
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
      elevation=-10.0,
      azimuth=140.0,
    ),
    sim=SimulationCfg(
      # Well above the martial motions' 35 and 250, which were sized for a humanoid on a
      # plane. A mantle puts two feet, two hands and sometimes a knee on the box while the
      # other foot is still on the floor. A constraint dropped on overflow is a contact that
      # silently did not happen, which looks like a robot sinking into the box rather than
      # an error, so this is deliberately generous
      nconmax=100,
      njmax=500,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    # 0.005 * 4 gives 50 Hz control, the rate the clip is converted at
    decimation=4,
    # The clip is a little over nine seconds with its held opening. The motion ends the
    # episode before this
    episode_length_s=10.0,
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
