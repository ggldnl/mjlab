"""The arena a diffusion crossing happens in.

The same one the imitation bridge trains in, and deliberately not a second copy of it. What
a bridge is asked to cross is a property of the corpus and of the window command that draws
from it, not of the architecture that answers, so the window term, the robot, the floor, the
action convention and the termination rules are shared and only the policy differs. Two
architectures scored by bridges/evaluate.py are then comparable line by line, which is the
entire reason that script takes --bridge instead of existing once per architecture.

The reward terms come along and are never read. No reinforcement learning happens in this
environment: the model is fitted offline by train.py and this environment only ever runs it.
They are left in place because a manager based task has to declare them and because a run of
the imitation bridge in the same arena should not be a different arena.

The observation is likewise along for the ride. The controller reads the robot and the
command term directly, which is strictly more than an observation vector carries, so the
actor group exists to satisfy the environment and the task config tests rather than to feed
anything.

Run

1. Build the corpus. It is shared, so it lives one level up.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.tracker

2. Train the model offline.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.train

3. Watch it.

    uv run play Mjlab-G1-Diffusion-Bridge
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  DEFAULT_DATASET,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.env_cfg import (
  bridge_env_cfg,
)


def diffusion_env_cfg(
  play: bool = False,
  split: str = "train",
  dataset_path: Path = DEFAULT_DATASET,
  sources: tuple[str, ...] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Build the environment a diffusion crossing runs in.

  Args:
    play: no observation noise, no start perturbation, no episode length cap.
    split: "train" or "eval". Split by recording environment, not by frame.
    dataset_path: which corpus the windows are drawn from.
    sources: restrict windows to these clips. None means any.
  """
  return bridge_env_cfg(
    play=play, split=split, dataset_path=dataset_path, sources=sources
  )
