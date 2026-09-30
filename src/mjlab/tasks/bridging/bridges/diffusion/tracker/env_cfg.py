"""Training environment for the diffusion bridge's universal tracker."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.bridging.bridges.diffusion.config import (
  motion_patterns as robot_motions,
)
from mjlab.tasks.bridging.bridges.diffusion.tracker import mdp
from mjlab.tasks.bridging.bridges.diffusion.tracker.actions import (
  ReferenceJointPositionActionCfg,
)
from mjlab.tasks.bridging.bridges.diffusion.tracker.command import (
  TrackerCommandCfg,
)
from mjlab.tasks.bridging.bridges.imitation.command import CHANNELS
from mjlab.tasks.bridging.config import get_robot
from mjlab.tasks.tracking.mdp.rewards import self_collision_cost
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

COMMAND = "path"


def tracker_env_cfg(
  play: bool = False,
  split: str = "train",
  dataset_path: Path | None = None,
  motion_patterns: tuple[str, ...] | None = None,
  sources: tuple[str, ...] | None = None,
  robot: str = "g1",
) -> ManagerBasedRlEnvCfg:
  """Build a short-window path tracker with uniform route tracking."""
  selected = get_robot(robot)
  if motion_patterns is None and dataset_path is None:
    motion_patterns = robot_motions(robot, "val" if split == "eval" else "train")
  cfg = make_tracking_env_cfg()
  cfg.scene.entities = {"robot": selected.get_entity_cfg()}
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(
        mode="subtree", pattern=selected.base_body_name, entity="robot"
      ),
      secondary=ContactMatch(
        mode="subtree", pattern=selected.base_body_name, entity="robot"
      ),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    ),
  )
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = selected.foot_geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = (selected.base_body_name,)
  cfg.scene.num_envs = 1 if play else 4096
  cfg.commands = {
    COMMAND: TrackerCommandCfg(
      entity_name="robot",
      robot=robot,
      dataset_path=dataset_path,
      motion_patterns=motion_patterns or (),
      split=split,
      sources=sources,
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=play,
      initial_position_noise=0.0 if play else 0.02,
      initial_yaw_noise=0.0 if play else 0.05,
      initial_velocity_noise=0.0 if play else 0.10,
      initial_joint_position_noise=0.0 if play else 0.02,
      initial_joint_velocity_noise=0.0 if play else 0.15,
    )
  }

  bridge = ObservationTermCfg(
    func=base_mdp.generated_commands, params={"command_name": COMMAND}
  )
  actor_state = {
    "root_height": ObservationTermCfg(
      func=mdp.history_root_height,
      params={"command_name": COMMAND},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.history_base_lin_vel,
      params={"command_name": COMMAND},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.history_base_ang_vel,
      params={"command_name": COMMAND},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.history_projected_gravity,
      params={"command_name": COMMAND},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.history_joint_pos,
      params={"command_name": COMMAND},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.history_joint_vel,
      params={"command_name": COMMAND},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "action_history": ObservationTermCfg(func=mdp.action_history),
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={"path": bridge, **actor_state},
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms={
        "path": replace(bridge),
        **{name: replace(term, noise=None) for name, term in actor_state.items()},
        "route_error": ObservationTermCfg(
          func=mdp.tracking_errors, params={"command_name": COMMAND}
        ),
        "endpoint_error": ObservationTermCfg(
          func=mdp.endpoint_errors, params={"command_name": COMMAND}
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  cfg.actions = {
    "joint_pos": ReferenceJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=selected.action_scale,
      use_default_offset=False,
      command_name=COMMAND,
      lookahead=1,
    )
  }
  # No self collision penalty: about a quarter of the reference frames put an arm
  # into the body. Physics still stops the penetration, so the robot follows as
  # closely as contact allows instead of steering its whole body away
  cfg.rewards = {
    "trajectory_tracking": RewardTermCfg(
      func=mdp.trajectory_tracking, weight=10.0, params={"command_name": COMMAND}
    ),
    "action_rate": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.05),
    "action_acc": RewardTermCfg(func=base_mdp.action_acc_l2, weight=-0.002),
    "joint_limits": RewardTermCfg(
      func=base_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "failed": RewardTermCfg(func=base_mdp.is_terminated, weight=-20.0),
  }
  cfg.terminations = {
    "route_done": TerminationTermCfg(
      func=mdp.route_done, params={"command_name": COMMAND}, time_out=True
    ),
    "fell_over": TerminationTermCfg(
      func=mdp.fell_over,
      params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.7},
    ),
  }
  cfg.metrics = {
    **{
      f"target_error_{name}": MetricsTermCfg(
        func=mdp.target_error,
        params={"command_name": COMMAND, "channel": index},
        reduce="max",
      )
      for index, name in enumerate(CHANNELS)
    },
    **{
      f"tracking_error_{name}": MetricsTermCfg(
        func=mdp.tracking_error,
        params={"command_name": COMMAND, "channel": index},
      )
      for index, name in enumerate(CHANNELS)
    },
    "within_endpoint_box": MetricsTermCfg(
      func=mdp.within_endpoint_box, params={"command_name": COMMAND}, reduce="max"
    ),
    "self_collision_hits": MetricsTermCfg(
      func=self_collision_cost,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    "route_score": MetricsTermCfg(
      func=mdp.route_score, params={"command_name": COMMAND}
    ),
  }
  if play:
    cfg.events = {}
  # route_done always ends the episode first (at most 2 s plus the 32 tick tail).
  # The value only normalizes the logged Episode_Reward terms.
  cfg.episode_length_s = 3.0
  cfg.is_finite_horizon = True
  cfg.viewer.body_name = selected.base_body_name
  return cfg
