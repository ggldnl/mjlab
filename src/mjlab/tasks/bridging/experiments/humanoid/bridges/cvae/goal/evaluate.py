"""Evaluate a trained goal CVAE on held-out physical handoffs.

Run:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.evaluate \
      --checkpoint logs/rsl_rl/g1_goal_cvae_bridge/<run>/model_5000.pt
"""

import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.evaluate_oracle import (
  EvaluateCfg,
  evaluate,
)

if __name__ == "__main__":
  cfg = tyro.cli(EvaluateCfg, config=mjlab.TYRO_FLAGS)
  cfg.student = True
  evaluate(cfg)
