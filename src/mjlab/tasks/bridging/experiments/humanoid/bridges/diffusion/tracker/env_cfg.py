"""Training environment for the diffusion bridge's universal tracker."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DEFAULT_DATASET,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker import mdp
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker.actions import (
  ReferenceJointPositionActionCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker.command import (
  TrackerCommandCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import CHANNELS
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

COMMAND = "path"


def tracker_env_cfg(
  play: bool = False,
  split: str = "train",
  dataset_path: Path = DEFAULT_DATASET,
  sources: tuple[str, ...] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Build a short-window path tracker with endpoint-aware objectives."""
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  cfg.scene.num_envs = 1 if play else 4096
  cfg.commands = {
    COMMAND: TrackerCommandCfg(
      entity_name="robot",
      dataset_path=dataset_path,
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
      scale=G1_ACTION_SCALE,
      use_default_offset=False,
      command_name=COMMAND,
      lookahead=1,
    )
  }
  cfg.rewards = {
    "trajectory_tracking": RewardTermCfg(
      func=mdp.trajectory_tracking, weight=1.0, params={"command_name": COMMAND}
    ),
    "endpoint_focus": RewardTermCfg(
      func=mdp.endpoint_focus, weight=3.0, params={"command_name": COMMAND}
    ),
    "terminal_target": RewardTermCfg(
      func=mdp.terminal_target, weight=12.0, params={"command_name": COMMAND}
    ),
    "action_rate": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.05),
    "action_acc": RewardTermCfg(func=base_mdp.action_acc_l2, weight=-0.002),
    "joint_limits": RewardTermCfg(
      func=base_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collisions": cfg.rewards["self_collisions"],
    "failed": RewardTermCfg(func=base_mdp.is_terminated, weight=-20.0),
  }
  cfg.terminations = {
    "deadline": TerminationTermCfg(
      func=mdp.deadline, params={"command_name": COMMAND}, time_out=True
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
        reduce="last",
      )
      for index, name in enumerate(CHANNELS)
    },
    "target_success": MetricsTermCfg(
      func=mdp.target_success, params={"command_name": COMMAND}, reduce="last"
    ),
    "route_score": MetricsTermCfg(
      func=mdp.route_score, params={"command_name": COMMAND}
    ),
  }
  if play:
    cfg.events = {}
  cfg.episode_length_s = 1.0e9
  cfg.is_finite_horizon = True
  cfg.viewer.body_name = "pelvis"
  return cfg
