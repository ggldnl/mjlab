"""Jump task configuration.

Factory for a base standing-jump task. Robot-specific configs call the factory
and fill in the placeholders (robot entity, body/sensor names, action scale).

The episode is a single jump: the robot stands on the near platform, crouches,
launches across the gap, and lands on the far platform near the sampled target.
Stability rewards are gated to the grounded phases; the ballistic abstraction
supplies the in-air guidance (takeoff velocity, parabola tracking, landing
accuracy).
"""

from __future__ import annotations

import math

from mjlab.abstraction import AbstractionCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.jump import mdp
from mjlab.tasks.jump.mdp.abstractions import JumpAbstractionCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

# Placeholder set per-robot.
_BASE_BODY_NAME = ""
_CONTACT_SENSOR_NAME = "feet_ground_contact"
_ABSTRACTION_NAME = "jump"

# Curriculum terrain: each row opens a wider gap (row 0 is ~flat). Envs start at
# row 0 and are promoted/demoted by the jump_terrain_levels curriculum.
GAP_NUM_LEVELS = 8

GAP_TERRAIN_CFG = TerrainGeneratorCfg(
  size=(8.0, 4.0),
  num_rows=GAP_NUM_LEVELS,
  num_cols=1,  # Ignored in curriculum mode (one column per sub-terrain).
  border_width=2.0,
  border_height=1.0,
  curriculum=True,
  color_scheme="none",
  sub_terrains={
    "gap": mdp.GapTerrainCfg(
      near_length=2.5,
      gap_range=(0.0, 0.6),  # Row 0 is flat; the widest row opens a 0.6 m gap.
    )
  },
)


def make_jump_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base standing-jump task configuration."""

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
    "jump_target": ObservationTermCfg(
      func=mdp.abstraction_obs,
      params={"abstraction_name": _ABSTRACTION_NAME, "key": "target"},
    ),
    "jump_takeoff_velocity": ObservationTermCfg(
      func=mdp.abstraction_obs,
      params={"abstraction_name": _ABSTRACTION_NAME, "key": "takeoff_velocity"},
    ),
    "jump_phase": ObservationTermCfg(
      func=mdp.abstraction_obs,
      params={"abstraction_name": _ABSTRACTION_NAME, "key": "phase"},
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
  # Abstractions.
  ##

  abstractions: dict[str, AbstractionCfg] = {
    _ABSTRACTION_NAME: JumpAbstractionCfg(
      entity_name="robot",
      base_body_name=_BASE_BODY_NAME,  # Set per-robot.
      contact_sensor_name=_CONTACT_SENSOR_NAME,
      # forward/lateral are fallbacks only; with the gap terrain the target
      # comes from the far-platform landing patches (distance scales with gap).
      forward_range=(1.1, 2.0),
      lateral_range=(-0.4, 0.4),
      apex_height_range=(0.2, 0.45),
      nominal_base_height=0.665,  # Override per-robot.
      debug_vis=True,
    )
  }

  ##
  # Events.
  ##

  events = {
    # Spawn standing on the near platform, facing +x toward the gap.
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.1, 0.1),
          "y": (-0.1, 0.1),
          "z": (0.0, 0.02),
          "yaw": (0.0, 0.0),
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
  }

  ##
  # Rewards.
  ##
  # Abstraction signals dominate; stability terms are grounded-gated; the rest
  # is light regularization. Weights are initial guesses meant for tuning.

  rewards = {
    # Abstraction signals (the task objective + in-air guidance).
    "takeoff": RewardTermCfg(
      func=mdp.abstraction_signal,
      weight=20.0,  # Sparse (fires once at liftoff): large weight to compensate.
      params={"abstraction_name": _ABSTRACTION_NAME, "signal_name": "takeoff"},
    ),
    "tracking": RewardTermCfg(
      func=mdp.abstraction_signal,
      # Dense across the flight phase, back-loaded along the arc (see
      # JumpAbstractionCfg.tracking_progress_rate): following the whole
      # trajectory is what pays, not a brief tip at the start.
      weight=5.0,
      params={"abstraction_name": _ABSTRACTION_NAME, "signal_name": "tracking"},
    ),
    "landing": RewardTermCfg(
      func=mdp.abstraction_signal,
      weight=20.0,  # Sparse (fires once at touchdown): the true objective.
      params={"abstraction_name": _ABSTRACTION_NAME, "signal_name": "landing"},
    ),
    # Terminal stability (only once landed near the target).
    # Gated to LANDED *and* weighted by target proximity, so the robot cannot
    # farm these by standing at the start or hopping in place.
    "upright": RewardTermCfg(
      func=mdp.landed_upright,
      weight=1.0,
      params={
        "abstraction_name": _ABSTRACTION_NAME,
        "std": math.sqrt(0.2),
        "proximity_std": 0.5,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "posture": RewardTermCfg(
      func=mdp.landed_posture,
      weight=0.5,
      params={
        "abstraction_name": _ABSTRACTION_NAME,
        "std": math.sqrt(0.5),
        "proximity_std": 0.5,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Penalize falling (fell_over / fell_in_gap) so that ending the episode
    # early to escape the small per-step costs is no longer optimal. Without
    # this, "tip over and die" beats both standing and jumping.
    "termination_penalty": RewardTermCfg(
      func=mdp.termination_penalty,
      weight=-25.0,
    ),
    # Regularization (always on).
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "joint_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1e-5),
  }

  ##
  # Terminations.
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_in_gap": TerminationTermCfg(
      func=mdp.fell_in_gap,
      params={"minimum_height": -0.3},
    ),
    "fell_over": TerminationTermCfg(
      func=mdp.grounded_bad_orientation,
      params={
        "abstraction_name": _ABSTRACTION_NAME,
        "limit_angle": math.radians(70.0),
      },
    ),
  }

  ##
  # Curriculum.
  ##

  curriculum = {
    # Widen the gap (and thus the jump distance) per env as it succeeds; make it
    # easier when it jumps and fails. Envs start at the flat row and only
    # advance once they reliably land near the target.
    "gap_levels": CurriculumTermCfg(
      func=mdp.jump_terrain_levels,
      params={"abstraction_name": _ABSTRACTION_NAME, "success_distance": 0.5},
    ),
  }

  ##
  # Assemble.
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=GAP_TERRAIN_CFG,
        max_init_terrain_level=0,  # All envs start on the flat row.
      ),
      num_envs=4096,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    abstractions=abstractions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=_BASE_BODY_NAME,  # Set per-robot.
      distance=4.0,
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
    episode_length_s=6.0,
  )
