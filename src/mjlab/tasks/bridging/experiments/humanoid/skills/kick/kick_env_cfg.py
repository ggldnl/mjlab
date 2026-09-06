"""The kick environment: track a human football kick, with a real ball in the way.

This is the martial arts environment plus a ball. The tracking half is unchanged and is
where the kick comes from: a converted clip says how to walk in, plant, swing and recover,
and the reward pays per frame for following it. The ball half is three terms, and it is
what stops the policy from performing a kick past a ball it never touches.

The motion prior is never annealed. That is the one thing to preserve if this is retuned,
and it is the lesson of PAiD (arXiv 2602.05310), whose staged curriculum runs its imitation
reward at full weight from the first stage to the last. A policy paid mainly for what the
ball does is doing reward shaped kicking and looks like it. An earlier version of this
package annealed the reference away and that is exactly what it produced.

Where the ball goes is not tuned here. dataset.py measures the striking foot at the frame it
moves fastest and stores the point one radius ahead of it, so the ball sits on the swing by
construction. What this module chooses is how far the spawn is allowed to wander off that
point, and the curriculum widens it from nothing.

That widening is what puts the ball in the observation, and what it costs is mostly in the
spread rather than the average. Driving the reference exactly, with no policy at all, over 64
environments at each scatter:

    scatter   contact   launched   speed mean    worst   power
      0.00       1.00       1.00         4.42     4.36    0.96
      0.04       1.00       1.00         4.53     4.15    0.96
      0.08       1.00       1.00         4.20     2.56    0.92
      0.15       0.86       0.86         3.39     0.00    0.74

At the eight centimetres this task ends on, a reference-perfect swing still connects every
time and averages 4.20 m/s against 4.42 unscattered. What changes is the worst case, which
falls from 4.36 to 2.56: some spawns are met square and some are grazed, and which is which
is only knowable from the ball's position. That spread is what a policy reading the ball can
earn back. The row below is why the scatter stops at eight: at fifteen a perfect tracker
misses one ball in seven outright, which is not a harder task but a broken one.

What to check, in this order:

    Episode_Metrics/kick_contact_rate   is the foot reaching the ball. Nothing downstream
                                        can work until this is high, and it saturates early:
                                        once tracking works it is 1.0 and says nothing more
    Episode_Metrics/kick_launch_rate    is the ball leaving rather than being leant on
    Episode_Metrics/kick_ball_speed     how fast, along the clip's kick direction, in m/s.
                                        The real progress metric once contact saturates. A
                                        reference-perfect kick averages 4.2 m/s at the final
                                        scatter
    Episode_Metrics/kick_ball_distance  how far it ends up, in m
    Curriculum/ball_spawn               how far the spawn is being scattered now

A contact_rate near one with a launch_rate well below it means the robot is walking into the
ball rather than kicking it, which is a tracking failure: check the body velocity term before
touching a ball weight.

Run

1. Convert the clip.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.kick.dataset

2. Train.

    uv run train Mjlab-G1-Kick --env.scene.num-envs 4096

3. Watch. The viewer's Ball sliders move the ball while the reference ghost plays the clip,
   which is how to check the converter's choice by eye. See dataset.py, step 4.

    uv run play Mjlab-G1-Kick
"""

from __future__ import annotations

from pathlib import Path

from mjlab.asset_zoo.objects.ball import get_ball_cfg
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.motion_lib import (
  discover_motion_files,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.kick import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

##
# Body groups.
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

# Both feet, whatever the clip. A kick is decided by where the feet are: one swings and the
# other is the only thing holding the robot up while it does. Which is which is in the
# reference frame by frame, so naming it here would add a per clip config and say nothing new
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

##
# Reset noise. The same small perturbation the martial motions use: a kick spends its swing
# on one foot, so there is not much margin to spend here.
##

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

##
# The ball.
##

# How far the spawn may wander off the point the converter picked, in metres, at the end of
# the curriculum. Eight centimetres is the last value a reference-perfect swing still
# connects at every time, see the table in the module docstring, and it stays inside the
# feet group's own 0.15 m tracking tolerance so the correction is affordable. It must also
# match BALL_SCATTER in dataset.py, which is the scatter the ball position was chosen to
# survive: widening it here alone silently invalidates that choice
BALL_FORWARD_RANGE = (-0.08, 0.08)
BALL_LATERAL_RANGE = (-0.08, 0.08)

# Speed, in m/s, at which the power term is worth about two thirds of its weight. Set from
# the clips, whose feet strike at 5 to 8 m/s: a 0.425 kg ball off a foot that fast leaves at
# a few metres a second, and a term that saturated at ten would spend its whole range on
# strikes this robot cannot make
KICK_SPEED_STD = 2.5

# Final weights of the two ball terms, ramped to by the curriculum. Declared here because
# play mode reads them back: the curriculum is dropped there, and the two drifting apart
# would leave the viewer showing a task nobody configured
W_TOUCH = 1.0
W_POWER = 3.0

##
# Curriculum thresholds, in environment steps. At 24 steps per env per iteration, 24_000
# steps is about 1000 iterations.
##

_STAGE_1 = 24_000
_STAGE_2 = 72_000


def g1_kick_env_cfg(
  motion_dir: Path,
  motion_files: tuple[str, ...] | None = None,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the kick environment around one converted clip.

  Args:
    motion_dir: Where dataset.py wrote the clip. Also read at runtime for the ball
      position, which is stored in the clip rather than configured here.
    motion_files: Converted npz clips. Defaults to everything in motion_dir, which for this
      task is the one clip.
    play: Start every episode at the beginning of the clip, drop the observation noise and
      the pushes, leave the reference unperturbed, and put the ball scatter and the ball
      reward weights at the values the curriculum would have reached.
  """
  motion_files = motion_files or discover_motion_files(motion_dir)

  ##
  # Observations
  ##

  actor_terms = {
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
    # The ball. Position is what the swing has to be adjusted onto, velocity and the contact
    # flag are what say the strike has happened and how it went
    "ball_pos": ObservationTermCfg(
      func=mdp.ball_pos_b, noise=Unoise(n_min=-0.02, n_max=0.02)
    ),
    "ball_vel": ObservationTermCfg(
      func=mdp.ball_vel_b, noise=Unoise(n_min=-0.1, n_max=0.1)
    ),
    "ball_contact": ObservationTermCfg(func=mdp.ball_contact),
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
    "ball_pos": ObservationTermCfg(func=mdp.ball_pos_b),
    "ball_vel": ObservationTermCfg(func=mdp.ball_vel_b),
    "ball_contact": ObservationTermCfg(func=mdp.ball_contact),
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
    # The jump's multi-clip phase tracker, with the goal half switched off. The scale is
    # pinned at one: stretching a kick horizontally moves the foot off the ball, which is
    # placed against the unstretched clip
    "motion": mdp.KickCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      # The viewer's ball sliders live on the command term, see mdp.KickCommand.create_gui
      gui=True,
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
    # Starts on the swing and is widened by the curriculum, see the module docstring
    "ball_spawn": EventTermCfg(
      mode="reset",
      func=mdp.reset_ball_at_strike,
      params={"forward_range": (0.0, 0.0), "lateral_range": (0.0, 0.0)},
    ),
  }

  ##
  # Rewards
  ##

  rewards: dict[str, RewardTermCfg] = {
    # Where the robot is in the world. The clip walks in before it strikes, so this is not a
    # stay-put term here: it is what keeps the approach the length the reference says
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
    # Posture. The feet carry the tightest tolerance and the largest weight, because a kick
    # that is right everywhere except the feet is a miss
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
    # Wider than the martial set's 2.0, because these clips are faster: the striking foot
    # peaks at 5 to 8 m/s where a punch reaches about 5. A tolerance that saturates over the
    # strike pays the same whether the foot swung or not, which is the one part of a kick
    # that has to be fast
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 2.5},
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
    # Wider than the martial set's 5.0 for the same reason: the striking knee runs through
    # 10 to 16 rad/s in these clips
    "motion_joint_vel": RewardTermCfg(
      func=mdp.motion_joint_velocity_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 8.0},
    ),
    # The ball. Two latched rungs, both ramped by the curriculum: connect, then send it
    "ball_touched": RewardTermCfg(func=mdp.ball_touched, weight=0.5 * W_TOUCH),
    "kick_power": RewardTermCfg(
      func=mdp.kick_power,
      weight=0.5 * W_POWER,
      params={"std": KICK_SPEED_STD},
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
    # Light, as in the martial set. The support foot pivots under a kick, and that pivot is
    # contact with a turning foot, which is what this measures
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
    # Finishing the kick is success, not failure. This must stay time_out=True
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
    # The ball moves off the swing only once the policy can follow the reference. Widening
    # it earlier asks for a correction from a policy that has no swing to correct
    "ball_spawn": CurriculumTermCfg(
      func=mdp.event_curriculum,
      params={
        "event_name": "ball_spawn",
        "stages": [
          {
            "step": 0,
            "params": {"forward_range": (0.0, 0.0), "lateral_range": (0.0, 0.0)},
          },
          {
            "step": _STAGE_1,
            "params": {"forward_range": (-0.04, 0.04), "lateral_range": (-0.04, 0.04)},
          },
          {
            "step": _STAGE_2,
            "params": {
              "forward_range": BALL_FORWARD_RANGE,
              "lateral_range": BALL_LATERAL_RANGE,
            },
          },
        ],
      },
    ),
    "ball_touched_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "ball_touched",
        "stages": [
          {"step": 0, "weight": 0.5 * W_TOUCH},
          {"step": _STAGE_1, "weight": W_TOUCH},
        ],
      },
    ),
    "kick_power_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "kick_power",
        "stages": [
          {"step": 0, "weight": 0.5 * W_POWER},
          {"step": _STAGE_1, "weight": W_POWER},
        ],
      },
    ),
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
  }

  ##
  # Metrics
  ##

  metrics = {
    "kick_contact_rate": MetricsTermCfg(func=mdp.contact_rate, reduce="last"),
    "kick_launch_rate": MetricsTermCfg(func=mdp.launch_rate, reduce="last"),
    "kick_ball_speed": MetricsTermCfg(func=mdp.ball_speed, reduce="max"),
    "kick_ball_distance": MetricsTermCfg(func=mdp.ball_distance, reduce="max"),
  }

  ##
  # Scene
  ##

  # Either foot against the ball, for the reason given in mdp.touching_ball
  foot_ball_cfg = ContactSensorCfg(
    name=mdp.FOOT_BALL_SENSOR,
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="geom", pattern="ball_collision", entity="ball"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
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
    entities={
      "robot": get_g1_robot_cfg(),
      # The spawn here only has to be legal. Every reset moves the ball onto the point the
      # clip's foot passes through, which is not known until the clip is read
      "ball": get_ball_cfg(),
    },
    sensors=(foot_ball_cfg, feet_ground_cfg, self_collision_cfg),
    env_spacing=3.0,
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
    metrics=metrics,
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
      # Not left on the heuristic, for the reason the pass task spells out: fourteen foot
      # capsules on the ground, joint limit rows as the swinging leg reaches its stops, and
      # the ball's own contacts. The overflow is quiet, so grep a training log for
      # "nefc overflow" before trusting a result
      njmax=300,
      contact_sensor_maxmatch=64,
      mujoco=MujocoCfg(
        timestep=0.005, iterations=10, ls_iterations=20, ccd_iterations=50
      ),
    ),
    # 0.005 * 4 gives 50 Hz control, the rate the clips are converted at
    decimation=4,
    # The longest published clip is 7.8 s and most are near 5. The motion ending stops the
    # episode before this, so it is a ceiling rather than a length
    episode_length_s=9.0,
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # The curriculum only ever raises these, so dropping it would show the task at its
    # opening stage: an unscattered ball and half weight ball terms
    cfg.curriculum = {}
    cfg.events["ball_spawn"].params["forward_range"] = BALL_FORWARD_RANGE
    cfg.events["ball_spawn"].params["lateral_range"] = BALL_LATERAL_RANGE
    cfg.rewards["ball_touched"].weight = W_TOUCH
    cfg.rewards["kick_power"].weight = W_POWER

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, mdp.KickCommandCfg)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.sampling_mode = "start"

  return cfg
