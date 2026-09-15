"""Standard mjlab environment for the docking bridge."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DEFAULT_DATASET,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.docking import mdp
from mjlab.tasks.bridging.experiments.humanoid.bridges.docking.command import (
  DockingCommandCfg,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

COMMAND = "bridge"
MAX_WINDOW_S = 1.5


def docking_env_cfg(
  play: bool = False,
  split: str = "train",
  dataset_path: Path = DEFAULT_DATASET,
  sources: tuple[str, ...] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Build the docking bridge training or playback environment."""
  command = DockingCommandCfg(
    entity_name="robot",
    dataset_path=dataset_path,
    split=split,
    sources=sources,
    resampling_time_range=(1.0e9, 1.0e9),
    docking_probability=0.0 if play else 0.7,
    debug_vis=play,
  )
  bridge_obs = ObservationTermCfg(
    func=base_mdp.generated_commands, params={"command_name": COMMAND}
  )
  action_obs = ObservationTermCfg(func=base_mdp.last_action)
  observations = {
    "actor": ObservationGroupCfg(
      terms={"bridge": bridge_obs, "last_action": action_obs},
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms={
        "bridge": replace(bridge_obs),
        "last_action": replace(action_obs),
      },
      enable_corruption=False,
    ),
  }
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": mdp.DockingJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=G1_ACTION_SCALE,
      use_default_offset=True,
      command_name=COMMAND,
      residual_scale=command.residual_scale,
    )
  }
  rewards = {
    "progress": RewardTermCfg(
      func=mdp.progress, weight=6.0, params={"command_name": COMMAND}
    ),
    "capture": RewardTermCfg(
      func=mdp.capture, weight=12.0, params={"command_name": COMMAND}
    ),
    "alive": RewardTermCfg(func=base_mdp.is_alive, weight=0.1),
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
    "handoff": TerminationTermCfg(
      func=mdp.handoff, params={"command_name": COMMAND}, time_out=True
    ),
    "missed_deadline": TerminationTermCfg(
      func=mdp.missed_deadline, params={"command_name": COMMAND}
    ),
    "strayed": TerminationTermCfg(
      func=mdp.strayed, params={"command_name": COMMAND, "margin": 1.5}
    ),
    "fell_over": TerminationTermCfg(
      func=mdp.fell_over,
      params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.7},
    ),
  }
  commands: dict[str, CommandTermCfg] = {COMMAND: command}
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      entities={"robot": get_g1_robot_cfg()},
    ),
    observations=observations,
    actions=actions,
    commands=commands,
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
    episode_length_s=1.0e9 if play else MAX_WINDOW_S,
  )
