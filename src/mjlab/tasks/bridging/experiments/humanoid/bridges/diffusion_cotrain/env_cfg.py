"""Recorded-only pretraining and online co-training environments."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  TRACKER_DATASET,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker.command import (
  TrackerCommandCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker.env_cfg import (
  COMMAND,
  tracker_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion_cotrain.command import (
  CoTrainingCommandCfg,
)


def tracker_pretrain_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Train only on physically recorded corpus trajectories."""
  return tracker_env_cfg(
    play=play,
    split="eval" if play else "train",
    dataset_path=TRACKER_DATASET,
  )


def cotrain_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = tracker_pretrain_env_cfg(play)
  original = cfg.commands[COMMAND]
  if not isinstance(original, TrackerCommandCfg):
    raise TypeError("tracker environment did not create a TrackerCommandCfg")
  values = vars(original).copy()
  values["duration_s_range"] = (0.3, 1.2)
  values["cross_trajectory_start"] = 1.0 if play else 0.0
  values["cross_trajectory_end"] = 1.0 if play else 0.5
  cfg.commands[COMMAND] = CoTrainingCommandCfg(**values)
  return cfg
