"""Training environment for the time-aligned imitation bridge."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DEFAULT_DATASET,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation import mdp
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
  ImitationCommandCfg,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

COMMAND = "bridge"


def imitation_env_cfg(
  play: bool = False,
  split: str = "train",
  dataset_path: Path = DEFAULT_DATASET,
  sources: tuple[str, ...] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Build a fixed-duration trajectory-tracking bridge environment."""
  command = ImitationCommandCfg(
    entity_name="robot",
    dataset_path=dataset_path,
    split=split,
    sources=sources,
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=play,
  )
  bridge = ObservationTermCfg(
    func=base_mdp.generated_commands, params={"command_name": COMMAND}
  )
  proprioception = {
    "base_lin_vel": ObservationTermCfg(
      func=base_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
    ),
    "base_ang_vel": ObservationTermCfg(
      func=base_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
    ),
    "projected_gravity": ObservationTermCfg(
      func=base_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    "joint_pos": ObservationTermCfg(
      func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
    ),
    "joint_vel": ObservationTermCfg(
      func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "last_action": ObservationTermCfg(func=base_mdp.last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(
      terms={"bridge": bridge, **proprioception},
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms={
        "bridge": replace(bridge),
        **{name: replace(term, noise=None) for name, term in proprioception.items()},
      },
      enable_corruption=False,
    ),
  }
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=G1_ACTION_SCALE,
      use_default_offset=True,
    )
  }
  rewards = {
    "trajectory_tracking": RewardTermCfg(
      func=mdp.trajectory_tracking, weight=1.0, params={"command_name": COMMAND}
    ),
    "terminal_target": RewardTermCfg(
      func=mdp.terminal_target, weight=8.0, params={"command_name": COMMAND}
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
  terminations = {
    "deadline": TerminationTermCfg(
      func=mdp.deadline, params={"command_name": COMMAND}, time_out=True
    ),
    "fell_over": TerminationTermCfg(
      func=mdp.fell_over,
      params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.7},
    ),
  }
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      entities={"robot": get_g1_robot_cfg()},
    ),
    observations=observations,
    actions=actions,
    commands={COMMAND: command},
    events={},
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="pelvis",
      distance=3.0,
      elevation=-10.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=35, njmax=250, mujoco=MujocoCfg(timestep=0.005, iterations=10)
    ),
    decimation=4,
    episode_length_s=1.0e9,
  )
