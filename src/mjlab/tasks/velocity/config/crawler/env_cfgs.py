"""Crawler velocity environment configurations.

Two flat-terrain velocity-tracking tasks for the ~60 g crawler quadruped:

- ``crawler_velocity_flat_env_cfg``: the *classic* setup. It reuses the shared
  ``make_velocity_env_cfg`` factory (the same one Go1/T1 use) and re-tunes it for
  a tiny, tip-prone robot. The single most important change is matching the
  command speeds, the tracking kernel, the observation noise, and the foot-height
  targets to the robot's actual scale - the factory defaults are sized for
  metre-scale robots moving at 1-3 m/s, which gives this robot a flat,
  gradient-free reward and explains why training never took off.

- ``crawler_velocity_abstraction_env_cfg``: the classic setup *plus* the trot
  gait-clock + velocity-path abstraction, which hands the policy the rhythm and
  the "where is forward" signal it is too tip-prone to discover on its own.

Both run on flat ground with no terrain/command curriculum so the only thing the
policy has to solve is "trot at the commanded velocity without tipping".
"""

from __future__ import annotations

import math

from mjlab.asset_zoo.robots.crawler.actuators import (
  ACTION_SCALE,
  COXA_JOINT_REGEX,
  FEMUR_JOINT_REGEX,
  LEG_PHASE_OFFSETS,
  TIBIA_JOINT_REGEX,
)
from mjlab.asset_zoo.robots.crawler.collisions import (
  BASE_NAME,
  FOOT_COLLISION_NAMES,
  FOOT_SITE_NAMES,
)
from mjlab.asset_zoo.robots.crawler.crawler_constants import get_crawler_robot_cfg
from mjlab.asset_zoo.robots.crawler.sensors import (
  FEET_GROUND,
  FOOT_HEIGHT_SCAN,
  IMU,
  NONFEET_GROUND,
  SELF_COLLISION,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity.config.crawler import mdp
from mjlab.tasks.velocity.config.crawler.mdp.abstractions import TrotGaitAbstractionCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg, self_collision_cost
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# The crawler's open-loop diagonal trot tops out around ~0.12 m/s; command the
# robot inside its physical envelope so the velocity-tracking reward actually has
# a reachable target (and therefore a gradient).
_LIN_VEL_X = (-0.12, 0.15)
_LIN_VEL_Y = (-0.08, 0.08)
_ANG_VEL_Z = (-1.0, 1.0)

# Velocity-tracking kernel widths. These must be small relative to the commanded
# speeds: with the factory's std=0.5 a robot standing still while commanded
# 0.12 m/s still collects ~94% of the reward, so moving is "not worth it".
_TRACK_LIN_STD = 0.05
_TRACK_ANG_STD = 0.25

# Swing-foot clearance for a ~5 cm-tall robot. The factory's 0.1 m target is
# taller than the whole robot.
_SWING_HEIGHT = 0.015

_ABSTRACTION_NAME = "trot"


def _flat_sensors():
  """Scene sensors for the flat crawler task.

  The crawler's IMU sensors live in ``sensors.py`` (not the MJCF), so they must
  be attached here for the factory's ``robot/imu_*`` observations to resolve. No
  terrain raycast on flat ground; the foot-height scan stays for foot clearance.
  """
  return (*IMU, FEET_GROUND, NONFEET_GROUND, SELF_COLLISION, FOOT_HEIGHT_SCAN)


def crawler_velocity_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Classic flat-terrain velocity-tracking config for the crawler."""
  cfg = make_velocity_env_cfg()

  ##
  # Simulation: finer timestep for the small, stiff (50 Hz) position servos and
  # the many small contacts. decimation 10 @ 0.002 s keeps 50 Hz control.
  ##
  cfg.sim.nconmax = 35
  cfg.sim.njmax = 300
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.mujoco.timestep = 0.002
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.mujoco.impratio = 10
  cfg.decimation = 10

  ##
  # Robot + flat terrain.
  ##
  cfg.scene.entities = {"robot": get_crawler_robot_cfg()}
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = _flat_sensors()

  ##
  # Actions: per-joint scale tuned to the propulsion/lift each joint needs (see
  # actuators.py). use_default_offset centers actions on the neutral stance.
  ##
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = ACTION_SCALE
  joint_pos_action.use_default_offset = True

  ##
  # Observations: drop the height scan (no terrain to scan) and rescale the noise
  # to the robot. The factory's ±0.5 m/s velocimeter noise swamps a 0.12 m/s
  # signal; ±1.5 rad/s joint-velocity noise is similarly oversized.
  ##
  for group in ("actor", "critic"):
    cfg.observations[group].terms.pop("height_scan", None)
  actor = cfg.observations["actor"].terms
  actor["base_lin_vel"].noise = Unoise(n_min=-0.05, n_max=0.05)
  actor["joint_vel"].noise = Unoise(n_min=-0.5, n_max=0.5)

  ##
  # Commands: small ranges, no heading auto-turn fighting the tiny ang command.
  ##
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = _LIN_VEL_X
  twist.ranges.lin_vel_y = _LIN_VEL_Y
  twist.ranges.ang_vel_z = _ANG_VEL_Z
  twist.ranges.heading = (-math.pi, math.pi)
  twist.viz.z_offset = 0.1

  ##
  # Rewards.
  ##
  cfg.rewards["track_linear_velocity"].weight = 2.0
  cfg.rewards["track_linear_velocity"].params["std"] = _TRACK_LIN_STD
  cfg.rewards["track_angular_velocity"].weight = 1.5
  cfg.rewards["track_angular_velocity"].params["std"] = _TRACK_ANG_STD

  # Keeping upright is the robot's hardest constraint, so weight it strongly.
  cfg.rewards["upright"].weight = 0.0
  cfg.rewards["upright"].params["asset_cfg"].body_names = (BASE_NAME,)

  # Posture: hold the neutral stance tightly when standing, open up while moving
  # so the gait is free to deviate.
  cfg.rewards["pose"].weight = 0.0
  cfg.rewards["pose"].params["asset_cfg"].joint_names = (
    COXA_JOINT_REGEX,
    FEMUR_JOINT_REGEX,
    TIBIA_JOINT_REGEX,
  )
  cfg.rewards["pose"].params["std_standing"] = {
    COXA_JOINT_REGEX: 0.05,
    FEMUR_JOINT_REGEX: 0.1,
    TIBIA_JOINT_REGEX: 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    COXA_JOINT_REGEX: 0.2,
    FEMUR_JOINT_REGEX: 0.3,
    TIBIA_JOINT_REGEX: 0.4,
  }
  cfg.rewards["pose"].params["std_running"] = dict(cfg.rewards["pose"].params["std_walking"])
  cfg.rewards["pose"].params["walking_threshold"] = 0.03
  cfg.rewards["pose"].params["running_threshold"] = 0.15

  # A small positive air-time reward to break the standing local optimum; stride
  # timing is short for a fast, small gait (~2.5 Hz).
  cfg.rewards["air_time"].weight = 0.5
  cfg.rewards["air_time"].params["threshold_min"] = 0.1
  cfg.rewards["air_time"].params["threshold_max"] = 0.3
  cfg.rewards["air_time"].params["command_threshold"] = 0.03

  # Foot shaping, re-targeted to the robot's height.
  cfg.rewards["foot_clearance"].weight = -0.5
  cfg.rewards["foot_clearance"].params["target_height"] = _SWING_HEIGHT
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = FOOT_SITE_NAMES
  cfg.rewards["foot_clearance"].params["command_threshold"] = 0.03
  cfg.rewards["foot_swing_height"].weight = -0.1
  cfg.rewards["foot_swing_height"].params["target_height"] = _SWING_HEIGHT
  cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.03
  cfg.rewards["foot_slip"].weight = -0.05
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = FOOT_SITE_NAMES
  cfg.rewards["foot_slip"].params["command_threshold"] = 0.03

  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (BASE_NAME,)
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["action_rate_l2"].weight = -0.05

  # Structural guards: penalize the base/legs scraping the ground and self
  # collisions. Forces are tiny at this scale, so the thresholds are low.
  cfg.rewards["nonfeet_ground"] = RewardTermCfg(
    func=self_collision_cost,
    weight=-1.0,
    params={"sensor_name": "nonfeet_ground_contact", "force_threshold": 0.5},
  )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=self_collision_cost,
    weight=-0.2,
    params={"sensor_name": "self_collision", "force_threshold": 0.5},
  )

  ##
  # Events: shrink the disturbances to the robot's scale. A 0.5 m/s push (factory
  # default) is several times the robot's top speed.
  ##
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.3, 0.3),
    "y": (-0.3, 0.3),
    "z": (0.0, 0.01),
    "yaw": (-math.pi, math.pi),
  }
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)
  cfg.events["push_robot"].interval_range_s = (2.0, 5.0)
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.1, 0.1),
    "y": (-0.1, 0.1),
    "z": (-0.05, 0.05),
    "roll": (-0.2, 0.2),
    "pitch": (-0.2, 0.2),
    "yaw": (-0.3, 0.3),
  }
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_COLLISION_NAMES
  cfg.events["foot_friction"].params["ranges"] = (0.6, 1.4)
  cfg.events["base_com"].params["asset_cfg"].body_names = (BASE_NAME,)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.005, 0.005),
    1: (-0.005, 0.005),
    2: (-0.005, 0.005),
  }

  ##
  # Terminations: only time-out and tipping on flat ground.
  ##
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(60.0)

  ##
  # No terrain/command curriculum on flat: keep the task stationary so the policy
  # only has to solve "trot at the commanded speed".
  ##
  cfg.curriculum = {}

  ##
  # Viewer: close in on the small robot.
  ##
  cfg.viewer.body_name = BASE_NAME
  cfg.viewer.distance = 0.6
  cfg.viewer.elevation = -10.0
  cfg.viewer.azimuth = 90.0

  cfg.episode_length_s = 20.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

  return cfg


def crawler_velocity_abstraction_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat crawler velocity task guided by the trot gait-clock + path abstraction.

  Starts from the classic flat config and layers on the abstraction: its
  reference (gait clock + path error) is fed to the policy, and its three dense
  signals (gait, clearance, path) become positive reward terms. The hand-built
  ``air_time`` shaping is dropped because the gait signal supersedes it.
  """
  cfg = crawler_velocity_flat_env_cfg(play=play)

  ##
  # Abstraction.
  ##
  cfg.abstractions = {
    _ABSTRACTION_NAME: TrotGaitAbstractionCfg(
      base_body_name=BASE_NAME,
      foot_site_names=FOOT_SITE_NAMES,
      leg_phase_offsets=tuple(LEG_PHASE_OFFSETS.tolist()),
      swing_height=_SWING_HEIGHT,
      path_std=_TRACK_LIN_STD,
      debug_vis=True,
    )
  }

  ##
  # Observations: expose the gait clock and the path error to both actor and
  # critic so the policy can phase-lock to the schedule it is scored against.
  ##
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    terms["gait_clock"] = ObservationTermCfg(
      func=mdp.abstraction_obs,
      params={"abstraction_name": _ABSTRACTION_NAME, "key": "gait_clock"},
    )
    terms["path_error"] = ObservationTermCfg(
      func=mdp.abstraction_obs,
      params={"abstraction_name": _ABSTRACTION_NAME, "key": "path_error"},
    )

  ##
  # Rewards: the abstraction signals drive the behavior. The gait signal replaces
  # the hand-built air-time shaping; the path signal complements the (kept, but
  # lighter) instantaneous velocity-tracking reward.
  ##
  cfg.rewards.pop("air_time", None)
  cfg.rewards["track_linear_velocity"].weight = 1.0
  cfg.rewards["track_angular_velocity"].weight = 1.0
  cfg.rewards["trot_path"] = RewardTermCfg(
    func=mdp.abstraction_signal,
    weight=2.0,
    params={"abstraction_name": _ABSTRACTION_NAME, "signal_name": "path"},
  )
  cfg.rewards["trot_gait"] = RewardTermCfg(
    func=mdp.abstraction_signal,
    weight=1.0,
    params={"abstraction_name": _ABSTRACTION_NAME, "signal_name": "gait"},
  )
  cfg.rewards["trot_clearance"] = RewardTermCfg(
    func=mdp.abstraction_signal,
    weight=1.0,
    params={"abstraction_name": _ABSTRACTION_NAME, "signal_name": "clearance"},
  )

  return cfg
