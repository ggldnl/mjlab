"""A karate strike for the Unitree G1, tracked from a standstill.

Run:

    1. Cut the clips out of LAFAN1 and convert them. Writes to data/lafan1_g1/karate.

       uv run python -m \
         mjlab.tasks.bridging.experiments.humanoid.skills.karate.dataset

    2. Train.

       uv run train Mjlab-G1-Karate --env.scene.num-envs 4096

    3. Watch.

       uv run play Mjlab-G1-Karate

The jump's recipe with a different clip: reference-state initialization, a dense per-frame
tracking reward, and termination the moment tracking is lost. What makes this one its own
skill is where the clip comes from and how it opens.

The source is LAFAN1's fight performances, retargeted to the 29 joint G1 by Unitree. Those
are minutes long, so dataset.py cuts a few seconds out of one, and the cut is a front kick,
a punch combination or a roundhouse depending on where the window falls.

Each clip is then given half a second of held stance at the front. A crop out of a fight
starts mid-bounce, and without the hold the policy would only ever be asked to continue a
motion that was already under way. With it, every episode that starts at frame zero starts
from a robot standing still, which is the state a composition would hand it.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.jump import jump_ppo_runner_cfg
from mjlab.tasks.bridging.experiments.humanoid.skills.karate.karate_env_cfg import (
  g1_karate_env_cfg,
)
from mjlab.tasks.registry import register_mjlab_task

KARATE_TASK_ID = "Mjlab-G1-Karate"


def karate_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The jump's PPO config, under this experiment's own name.

  Nothing about the algorithm changes, because nothing about the problem does: the same
  tracking objective on the same robot with an observation of the same shape. It runs
  shorter than the jump because there is no goal to cover, only one clip to follow.
  """
  return replace(jump_ppo_runner_cfg("g1_karate"), max_iterations=10_000)


register_mjlab_task(
  task_id=KARATE_TASK_ID,
  env_cfg=g1_karate_env_cfg(),
  play_env_cfg=g1_karate_env_cfg(play=True),
  rl_cfg=karate_ppo_runner_cfg(),
)
