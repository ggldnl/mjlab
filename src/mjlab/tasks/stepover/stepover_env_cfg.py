"""Step-over task configuration.

Factory for a barrier step-over task. Robot-specific configs call the factory
and fill in the placeholders (base body, foot sites, action scale).

The episode places the robot behind a full-width barrier; it must walk up and
step over it one leg at a time, then settle on the far side. The swing-foot
via-point abstraction supplies the dense guidance (clearance, approach progress,
crossing), and a barrier-height curriculum raises the bar as envs succeed.
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
from mjlab.tasks.stepover import mdp
from mjlab.tasks.stepover.mdp.abstractions import StepOverAbstractionCfg
from mjlab.tasks.stepover.mdp.terrain import BarrierTerrainCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

# Placeholders set per-robot.
_BASE_BODY_NAME = ""
_FOOT_SITE_NAMES: tuple[str, ...] = ()
_ABSTRACTION_NAME = "stepover"

# Curriculum terrain: each row raises the barrier (row 0 is flat). Envs start at
# row 0 and are promoted/demoted by the barrier_terrain_levels curriculum.
BARRIER_NUM_LEVELS = 8

BARRIER_TERRAIN_CFG = TerrainGeneratorCfg(
  size=(4.0, 3.0),
  num_rows=BARRIER_NUM_LEVELS,
  num_cols=1,  # Ignored in curriculum mode (one column per sub-terrain).
  border_width=2.0,
  border_height=1.0,
  curriculum=True,
  color_scheme="none",
  sub_terrains={
    "barrier": BarrierTerrainCfg(
      spawn_distance=0.4,  # Spawn standing right in front of the barrier.
      barrier_height_range=(0.0, 0.3),  # Row 0 is flat; widest row is 0.3 m.
    )
  },
)


def make_stepover_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base barrier step-over task configuration."""

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
    "barrier": ObservationTermCfg(
      func=mdp.abstraction_obs,
      params={"abstraction_name": _ABSTRACTION_NAME, "key": "barrier"},
    ),
    "via_points": ObservationTermCfg(
      func=mdp.abstraction_obs,
      params={"abstraction_name": _ABSTRACTION_NAME, "key": "via_points"},
    ),
    "phase": ObservationTermCfg(
      func=mdp.abstraction_obs,
      params={"abstraction_name": _ABSTRACTION_NAME, "key": "phase"},
    ),
    "feet_crossed": ObservationTermCfg(
      func=mdp.abstraction_obs,
      params={"abstraction_name": _ABSTRACTION_NAME, "key": "feet_crossed"},
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
    _ABSTRACTION_NAME: StepOverAbstractionCfg(
      entity_name="robot",
      base_body_name=_BASE_BODY_NAME,  # Set per-robot.
      foot_site_names=_FOOT_SITE_NAMES,  # Set per-robot.
      debug_vis=True,
    )
  }

  ##
  # Events.
  ##

  events = {
    # Spawn standing behind the barrier, facing +x.
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
  # Abstraction signals drive the behavior; a positive alive/posture baseline
  # keeps standing strictly better than tipping over; the rest is light
  # regularization. Weights are initial guesses meant for tuning.

  rewards = {
    # Step-over guidance: lift the swing foot over the barrier via-point.
    "clearance": RewardTermCfg(
      func=mdp.abstraction_signal,
      weight=3.0,
      params={"abstraction_name": _ABSTRACTION_NAME, "signal_name": "clearance"},
    ),
    # The objective: get (and keep) both feet on the far side.
    "cross": RewardTermCfg(
      func=mdp.abstraction_signal,
      weight=3.0,
      params={"abstraction_name": _ABSTRACTION_NAME, "signal_name": "cross"},
    ),
    # Settle standing still on the far side (gated to the trunk being beyond the
    # barrier, so it cannot be farmed by standing on the near side).
    "crossed_upright": RewardTermCfg(
      func=mdp.crossed_upright,
      weight=1.0,
      params={
        "abstraction_name": _ABSTRACTION_NAME,
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "crossed_posture": RewardTermCfg(
      func=mdp.crossed_posture,
      weight=1.0,
      params={
        "abstraction_name": _ABSTRACTION_NAME,
        "std": math.sqrt(0.5),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Small positive baseline so "tip over and die" is never optimal, kept well
    # below the crossing rewards so standing still is not a stable optimum.
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
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  ##
  # Curriculum.
  ##

  curriculum = {
    "barrier_levels": CurriculumTermCfg(
      func=mdp.barrier_terrain_levels,
      params={"abstraction_name": _ABSTRACTION_NAME},
    ),
  }

  ##
  # Assemble.
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=BARRIER_TERRAIN_CFG,
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
