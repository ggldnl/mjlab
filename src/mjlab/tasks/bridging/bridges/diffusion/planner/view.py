"""View held out kinematic planner trajectories in Viser.

Run:

    uv run python -m mjlab.tasks.bridging.bridges.diffusion.planner.view \
      --checkpoint logs/rsl_rl/g1_kinematic_diffusion_planner/<run>/model_30000.pt
"""

import tyro

import mjlab
from mjlab.tasks.bridging.bridges.diffusion.evaluation.view import ViewCfg, serve

if __name__ == "__main__":
  serve(tyro.cli(ViewCfg, config=mjlab.TYRO_FLAGS))
