"""DAgger environment for the route-conditioned endpoint bridge."""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal import mdp
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.collect import (
  DEFAULT_ORACLE_DATASET,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.command import (
  GoalCommandCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.oracle import (
  mdp as oracle_mdp,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
  CHANNELS,
)
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

BRIDGE = "bridge"
FOOT_CHANNELS = ("foot_pos", "foot_ori", "foot_lin_vel", "foot_ang_vel")


def goal_env_cfg(
  play: bool = False,
  split: str = "train",
  dataset_path: Path = DEFAULT_ORACLE_DATASET,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  if not play:
    cfg.scene.num_envs = 128
  motion = cfg.commands["motion"]
  if not isinstance(motion, MotionCommandCfg):
    raise TypeError("G1 tracking configuration has no motion command")
  cfg.commands = {
    BRIDGE: GoalCommandCfg(
      entity_name="robot",
      dataset_path=dataset_path,
      split=split,
      body_names=motion.body_names,
      foot_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=play,
    )
  }

  actor_terms = {
    "endpoint": ObservationTermCfg(func=mdp.endpoint, params={"command_name": BRIDGE}),
    "root_height": ObservationTermCfg(func=mdp.root_height),
    "base_lin_vel": ObservationTermCfg(
      func=base_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=base_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "gravity": ObservationTermCfg(
      func=base_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=base_mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "last_action": ObservationTermCfg(func=base_mdp.last_action),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=not play
    ),
    "initial": ObservationGroupCfg(
      terms={
        "condition": ObservationTermCfg(
          func=mdp.initial, params={"command_name": BRIDGE}
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "posterior": ObservationGroupCfg(
      terms={
        "path": ObservationTermCfg(func=mdp.posterior, params={"command_name": BRIDGE})
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "route": ObservationGroupCfg(
      terms={
        "label": ObservationTermCfg(func=mdp.route, params={"command_name": BRIDGE})
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "teacher": ObservationGroupCfg(
      terms={
        "proprioception": ObservationTermCfg(
          func=oracle_mdp.oracle_proprioception, params={"command_name": BRIDGE}
        ),
        "goal": ObservationTermCfg(
          func=oracle_mdp.oracle_goal, params={"command_name": BRIDGE}
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(
        mode="subtree",
        pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found",),
      reduce="netforce",
      num_slots=1,
    ),
  )
  cfg.events = {}
  cfg.rewards = {}
  cfg.metrics = {
    **{
      f"target_error_{name}": MetricsTermCfg(
        func=mdp.target_error,
        params={"command_name": BRIDGE, "channel": index},
        reduce="last",
      )
      for index, name in enumerate(CHANNELS)
    },
    **{
      f"target_error_{name}": MetricsTermCfg(
        func=mdp.foot_error,
        params={"command_name": BRIDGE, "channel": index},
        reduce="last",
      )
      for index, name in enumerate(FOOT_CHANNELS)
    },
    "target_success": MetricsTermCfg(
      func=mdp.target_success, params={"command_name": BRIDGE}, reduce="last"
    ),
  }
  cfg.terminations = {
    "deadline": TerminationTermCfg(
      func=mdp.deadline, params={"command_name": BRIDGE}, time_out=True
    ),
    "fell_over": TerminationTermCfg(
      func=mdp.fell_over,
      params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.2},
    ),
  }
  cfg.episode_length_s = 1.0e9
  cfg.viewer.body_name = "pelvis"
  return cfg
