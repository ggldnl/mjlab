"""The running skill: velocity tracking at speeds the stock task does not reach.

    uv run train Mjlab-Parkour-Run --env.scene.num-envs 4096
    uv run play Mjlab-Parkour-Run

Same machinery as `Mjlab-Velocity-Flat-Unitree-G1`, retuned for one thing: forward
speed. See run_env_cfg.py for what changed and why.

The curriculum runs through five speed stages, so this wants a long training run;
`max_iterations` is set accordingly. Watch `Curriculum/command_vel/lin_vel_x_max`
to see which stage it is on, and the velocity tracking reward to see whether it is
actually holding the commanded speed or merely being asked for it.
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

RUN_TASK_ID = "Mjlab-Parkour-Run"


def run_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The stock G1 velocity PPO config, under its own experiment name.

  Nothing about the algorithm needs changing to go faster; what needs changing is
  how long it gets, because the speed curriculum has five stages to climb and each
  one is a genuinely harder control problem than the last.
  """
  return replace(
    unitree_g1_ppo_runner_cfg(),
    experiment_name="parkour_run",
    max_iterations=15_000,
  )


register_mjlab_task(
  task_id=RUN_TASK_ID,
  env_cfg=g1_run_env_cfg(),
  play_env_cfg=g1_run_env_cfg(play=True),
  rl_cfg=run_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
