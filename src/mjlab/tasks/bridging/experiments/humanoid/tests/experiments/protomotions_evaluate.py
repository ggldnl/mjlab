"""Measure the ProtoMotions G1 BONES tracker on a LAFAN1 clip.

Run:

    uv run python -m \
      mjlab.tasks.bridging.experiments.humanoid.tests.experiments.protomotions_evaluate
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.execution.protomotions import (
  DEFAULT_CHECKPOINT,
  ProtoMotionsPolicy,
  protomotions_g1_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.tests.experiments.unitracker_evaluate import (
  Config as EvaluationConfig,
)
from mjlab.tasks.bridging.experiments.humanoid.tests.experiments.unitracker_evaluate import (
  evaluate_tracker,
)


@dataclass
class Config(EvaluationConfig):
  checkpoint: Path = DEFAULT_CHECKPOINT
  output_dir: Path | None = Path("data/protomotions/evaluation")


def evaluate(cfg: Config) -> torch.Tensor:
  return evaluate_tracker(
    cfg,
    protomotions_g1_env_cfg(play=True),
    lambda env: ProtoMotionsPolicy(env, cfg.checkpoint),
    "protomotions",
  )


if __name__ == "__main__":
  evaluate(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
