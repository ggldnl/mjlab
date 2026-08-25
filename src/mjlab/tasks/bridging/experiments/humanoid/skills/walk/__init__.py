"""The walking skill: mjlab's G1 flat velocity task, wired under a parkour name.

    uv run train Mjlab-Parkour-Walk --env.scene.num-envs 4096
    uv run play Mjlab-Parkour-Walk

There is no new environment here and there should not be. Ordinary commanded-velocity
walking is exactly what the corridor needs between obstacles, mjlab's version of it is
tuned, and a copy would only drift from the original.

What the alias buys is the experiment's ability to name its own skills. The pool is
built from task ids, checkpoints are found through each task's experiment name, and a
skill called `Mjlab-Velocity-Flat-Unitree-G1` would collect every G1 velocity run ever
trained in this repo, including ones with different reward weights or terrain. Under
`parkour_walk` the corridor's walking policy has its own log directory and its own
checkpoint history, and the experiment can say which one it means.

The registered config is the stock one, unmodified. If the corridor ever needs walking
to differ from mjlab's (a narrower command range, say, or a taller obstacle clearance)
that change belongs here, in a wrapper around `unitree_g1_flat_env_cfg`, the way
run/run_env_cfg.py wraps it for speed.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

WALK_TASK_ID = "Mjlab-Parkour-Walk"


def walk_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The stock G1 velocity PPO config, under this experiment's own name."""
  return replace(unitree_g1_ppo_runner_cfg(), experiment_name="parkour_walk")


register_mjlab_task(
  task_id=WALK_TASK_ID,
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=walk_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
