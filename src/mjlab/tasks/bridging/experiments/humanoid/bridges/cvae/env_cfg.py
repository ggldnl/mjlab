"""Parallel G1 environment for endpoint-conditioned DAgger."""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae import mdp
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.command import (
  CvaeCommandCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.teachers import (
  load_teachers,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DEFAULT_DATASET,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import CHANNELS
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

MOTION = "motion"
BRIDGE = "bridge"


def cvae_env_cfg(
  play: bool = False,
  split: str = "train",
  dataset_path: Path = DEFAULT_DATASET,
  motion_file: str = "",
) -> ManagerBasedRlEnvCfg:
  """Build the flat-ground bridge environment."""
  motion_file = motion_file or str(load_teachers()[0].motion)
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  if not play:
    cfg.scene.num_envs = 128
  motion = cfg.commands[MOTION]
  if not isinstance(motion, MotionCommandCfg):
    raise TypeError("G1 tracking configuration has no motion command")
  motion.motion_file = motion_file
  motion.pose_range = {}
  motion.velocity_range = {}
  motion.joint_position_range = (0.0, 0.0)
  motion.sampling_mode = "start"
  motion.debug_vis = False

  cfg.commands = {
    MOTION: motion,
    BRIDGE: CvaeCommandCfg(
      entity_name="robot",
      dataset_path=dataset_path,
      split=split,
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=play,
    ),
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
    "projected_gravity": ObservationTermCfg(
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
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact, params={"sensor_name": "feet_ground_contact"}
    ),
    "last_action": ObservationTermCfg(func=base_mdp.last_action),
  }
  teacher_terms = {
    "command": ObservationTermCfg(
      func=mdp.teacher_motion, params={"command_name": BRIDGE}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.teacher_anchor_pos_b, params={"command_name": BRIDGE}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.teacher_anchor_ori_b, params={"command_name": BRIDGE}
    ),
    "base_lin_vel": ObservationTermCfg(
      func=base_mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=base_mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "joint_pos": ObservationTermCfg(
      func=base_mdp.joint_pos_rel, params={"biased": True}
    ),
    "joint_vel": ObservationTermCfg(func=base_mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=base_mdp.last_action),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=not play
    ),
    "posterior": ObservationGroupCfg(
      terms={
        "path": ObservationTermCfg(
          func=mdp.posterior_path, params={"command_name": BRIDGE}
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "handoff": ObservationGroupCfg(
      terms={
        "target": ObservationTermCfg(
          func=mdp.handoff_target, params={"command_name": BRIDGE}
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "teacher": ObservationGroupCfg(
      terms=teacher_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  feet_ground = ContactSensorCfg(
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
  )
  cfg.scene.sensors = (*cfg.scene.sensors, feet_ground)
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
    "target_success": MetricsTermCfg(
      func=mdp.target_success,
      params={"command_name": BRIDGE},
      reduce="last",
    ),
    "target_action_error": MetricsTermCfg(
      func=mdp.target_action_error,
      params={"command_name": BRIDGE},
      reduce="last",
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
