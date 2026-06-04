"""Waving (greeting) task configuration.

Factory for a stand-and-wave task. Robot-specific configs call the factory and
fill in the placeholders (base body, waving-arm joints, standing joints, action
scale, and the per-joint wave shape).

The robot starts standing on flat ground and must hold a stable standing posture
while one arm tracks a phase-driven greeting wave. The wave target is the arm's
default pose plus a fixed raise offset plus a sinusoidal swing; everything else
is held near the standing pose. No locomotion command and no curriculum.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.waving import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

# Placeholders set per-robot.
_BASE_BODY_NAME = ""
_WAVE_JOINT_NAMES: tuple[str, ...] = ()
_STANDING_JOINT_NAMES: tuple[str, ...] = (".*",)

# Wave cycle frequency in Hz. Shared by the phase observation and the wave
# reward so they read the same clock.
WAVE_FREQUENCY = 1.0


def make_waving_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base stand-and-wave task configuration."""

  ##
  # Observations.
  ##

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "wave_phase": ObservationTermCfg(
      func=mdp.wave_phase,
      params={"frequency": WAVE_FREQUENCY},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=dict(actor_terms),
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=dict(actor_terms),
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions.
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Events.
  ##

  events = {
    # Spawn standing, facing +x, with light pose noise.
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.1, 0.1),
          "y": (-0.1, 0.1),
          "z": (0.0, 0.02),
          "yaw": (-0.3, 0.3),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Light pushes so the standing balance is robust, not brittle.
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(2.0, 4.0),
      params={
        "velocity_range": {
          "x": (-0.3, 0.3),
          "y": (-0.3, 0.3),
          "yaw": (-0.4, 0.4),
        },
      },
    ),
  }

  ##
  # Rewards.
  ##
  # The wave-tracking term drives the greeting; a tight posture term holds the
  # rest of the body standing; an upright penalty plus a small alive baseline
  # keep standing strictly better than tipping over; the rest is light
  # regularization. Weights are initial guesses meant for tuning.

  rewards = {
    "wave": RewardTermCfg(
      func=mdp.wave_arm,
      weight=2.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=_WAVE_JOINT_NAMES),
        "frequency": WAVE_FREQUENCY,
        "center": {},  # Set per-robot.
        "amplitude": {},  # Set per-robot.
      },
    ),
    # Hold the non-waving joints near the standing pose.
    "posture": RewardTermCfg(
      func=mdp.posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=_STANDING_JOINT_NAMES),
        "std": {},  # Set per-robot.
      },
    ),
    # Keep the base level (gravity should project straight down in body frame).
    "flat_orientation": RewardTermCfg(
      func=mdp.flat_orientation_l2,
      weight=-2.0,
    ),
    # Small positive baseline so "tip over and die" is never optimal.
    "alive": RewardTermCfg(func=mdp.is_alive, weight=0.5),
    # Regularization.
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "joint_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1e-5),
  }

  ##
  # Terminations.
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(45.0)},
    ),
  }

  ##
  # Assemble.
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=4096,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=_BASE_BODY_NAME,  # Set per-robot.
      distance=3.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
