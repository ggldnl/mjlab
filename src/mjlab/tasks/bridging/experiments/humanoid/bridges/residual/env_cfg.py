"""Training environment for the terminal residual component."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DEFAULT_DATASET,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.env_cfg import (
  imitation_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.residual import mdp
from mjlab.tasks.bridging.experiments.humanoid.bridges.residual.command import (
  ResidualCommandCfg,
)

COMMAND = "bridge"
ACTION = "joint_pos"


def residual_env_cfg(
  play: bool = False,
  split: str = "train",
  dataset_path: Path = DEFAULT_DATASET,
  sources: tuple[str, ...] | None = None,
  imitation_checkpoint: Path | None = None,
) -> ManagerBasedRlEnvCfg:
  """Reuse imitation's arena while training only short terminal corrections."""
  cfg = imitation_env_cfg(
    play=play, split=split, dataset_path=dataset_path, sources=sources
  )
  command = ResidualCommandCfg(
    entity_name="robot",
    dataset_path=dataset_path,
    split=split,
    sources=sources,
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=play,
  )
  cfg.commands = {COMMAND: command}
  cfg.actions = {
    ACTION: mdp.ResidualJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=G1_ACTION_SCALE,
      use_default_offset=True,
      command_name=COMMAND,
      residual_scale=command.residual_scale,
      imitation_checkpoint=imitation_checkpoint,
    )
  }

  last_action = ObservationTermCfg(
    func=mdp.effective_action, params={"action_name": ACTION}
  )
  for group in cfg.observations.values():
    group.terms["last_action"] = replace(last_action)

  cfg.rewards = {
    "progress": RewardTermCfg(
      func=mdp.progress, weight=6.0, params={"command_name": COMMAND}
    ),
    "capture": RewardTermCfg(
      func=mdp.capture, weight=20.0, params={"command_name": COMMAND}
    ),
    "residual_effort": RewardTermCfg(
      func=mdp.residual_effort, weight=-0.02, params={"action_name": ACTION}
    ),
    "action_rate": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.02),
    "joint_limits": RewardTermCfg(
      func=base_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "failed": RewardTermCfg(func=base_mdp.is_terminated, weight=-20.0),
  }
  cfg.terminations = {
    "captured": TerminationTermCfg(
      func=mdp.captured, params={"command_name": COMMAND}, time_out=True
    ),
    "missed_deadline": TerminationTermCfg(
      func=mdp.missed_deadline, params={"command_name": COMMAND}
    ),
    "fell_over": TerminationTermCfg(
      func=mdp.fell_over,
      params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.7},
    ),
  }
  return cfg
