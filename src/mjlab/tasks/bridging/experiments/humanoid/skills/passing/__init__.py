"""Passing: a standing G1 sends a football off at a commanded launch velocity.

Run:

    uv run train Mjlab-G1-Pass --env.scene.num-envs 4096
    uv run play Mjlab-G1-Pass

The ball is a size 5 football from the asset zoo, spawned in reach of the right foot, so
there is no locomotion here. The robot stands, waits out a stance window, then puts the
ball where it was asked: the command is a launch speed and a heading offset, and the reward
scores the ball's own velocity against it. This is essentially the robot passing a ball to
someone, not a proper kick.

What to check:

    Metrics/pass/launch_rate      is the ball struck at all. Everything else is
                                  meaningless until this is high
    Metrics/pass/vel_error        is the policy answering the command or repeating one
    Metrics/pass/heading_error    memorized pass. A policy ignoring the command still
                                  launches and still scores on pass_quality whenever the
                                  sampled command lands near its one pass; what it cannot
                                  do is drive these two down
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.passing.pass_env_cfg import (
  g1_pass_env_cfg,
)
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

PASS_TASK_ID = "Mjlab-G1-Pass"


def pass_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The stock G1 velocity PPO config, under its own experiment name.

  Unchanged apart from the name and the length. This is a reward shaped locomotion style
  problem on the same robot with a comparably sized observation, which is what that config
  is tuned for.

  If training plateaus, read Loss/learning_rate before touching the reward. The KL adaptive
  schedule clamps to a floor of 1e-5 and prints as 0.00000 once it gets there, which looks
  exactly like a reward problem and is not one. desired_kl 0.02 with num_learning_epochs 3
  is what has worked here before.
  """
  return replace(
    unitree_g1_ppo_runner_cfg(),
    experiment_name="g1_pass",
    save_interval=200,
    max_iterations=10_000,
  )


register_mjlab_task(
  task_id=PASS_TASK_ID,
  env_cfg=g1_pass_env_cfg(),
  play_env_cfg=g1_pass_env_cfg(play=True),
  rl_cfg=pass_ppo_runner_cfg(),
)
