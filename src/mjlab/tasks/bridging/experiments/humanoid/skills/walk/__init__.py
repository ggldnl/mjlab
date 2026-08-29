"""Walking: mjlab's G1 flat velocity task, registered under our own name.

Run:

    uv run train Mjlab-G1-Walk --env.scene.num-envs 4096
    uv run play Mjlab-G1-Walk

The registered config is the stock one, unmodified. Only the experiment name changes, so
the checkpoints land under g1_walk where the dataset collector looks for them.

If walking ever has to differ from mjlab's (a narrower command range, say), that change
belongs here, in a wrapper around `unitree_g1_flat_env_cfg`, the way run/run_env_cfg.py
wraps it for speed.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

WALK_TASK_ID = "Mjlab-G1-Walk"


def walk_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The stock G1 velocity PPO config, under this experiment's own name."""
  return replace(unitree_g1_ppo_runner_cfg(), experiment_name="g1_walk")


register_mjlab_task(
  task_id=WALK_TASK_ID,
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=walk_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
