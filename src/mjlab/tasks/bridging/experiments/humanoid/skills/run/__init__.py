"""Running: velocity tracking at speeds the stock walk task does not reach.

Run:

    uv run train Mjlab-G1-Run --env.scene.num-envs 4096
    uv run play Mjlab-G1-Run

Same machinery as Mjlab-Velocity-Flat-Unitree-G1, tuned for forward speed. See
run_env_cfg.py for what changed.

The curriculum climbs five speed stages, so this needs a long run and `max_iterations` is
set accordingly. Two things to watch:

    Curriculum/command_vel/lin_vel_x_max   which stage it is on
    the velocity tracking reward           whether it holds the commanded speed or not
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.run.run_env_cfg import (
  g1_run_env_cfg,
)
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

RUN_TASK_ID = "Mjlab-G1-Run"


def run_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The stock G1 velocity PPO config, under its own experiment name.

  Nothing about the algorithm needs changing to go faster. What needs changing is how long
  it gets: the speed curriculum has five stages to climb and each is a harder control
  problem than the last.
  """
  return replace(
    unitree_g1_ppo_runner_cfg(),
    experiment_name="g1_run",
    max_iterations=15_000,
  )


register_mjlab_task(
  task_id=RUN_TASK_ID,
  env_cfg=g1_run_env_cfg(),
  play_env_cfg=g1_run_env_cfg(play=True),
  rl_cfg=run_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
