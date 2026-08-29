"""Kicking: a standing G1 strikes a football at a commanded launch velocity.

Run:

    uv run train Mjlab-G1-Kick --env.scene.num-envs 4096
    uv run play Mjlab-G1-Kick

The ball is a size 5 football from the asset zoo, spawned in reach of the right foot, so
there is no locomotion here. The robot stands, waits out a stance window, then puts the
ball where it was asked: the command is a launch speed and a heading offset, and the reward
scores the ball's own velocity against it.

Reward shaped, not reference tracked, because there is no clip to track. That makes
exploration the whole problem, and the answer is a three rung ladder plus a curriculum
holding the kick weights at zero until standing is solved. See kick_env_cfg.py for the
ladder and mdp.py for the shape of each rung.

What to check before believing a run worked:

    Metrics/kick/launch_rate      is the ball struck at all. Everything else is
                                  meaningless until this is high
    Metrics/kick/vel_error        is the policy answering the command or repeating one
    Metrics/kick/heading_error    memorized kick. A policy ignoring the command still
                                  launches and still scores on kick_quality whenever the
                                  sampled command lands near its one kick; what it cannot
                                  do is drive these two down

Not wired into any composed scenario. The bridge reads a shared proprioception group and
the kick's observation carries ball state that group has no channel for. Adding it means
giving the arena a ball, which is a change to the scenario rather than to this task.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.kick.kick_env_cfg import (
  g1_kick_env_cfg,
)
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

KICK_TASK_ID = "Mjlab-G1-Kick"


def kick_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
    experiment_name="g1_kick",
    save_interval=200,
    max_iterations=10_000,
  )


register_mjlab_task(
  task_id=KICK_TASK_ID,
  env_cfg=g1_kick_env_cfg(),
  play_env_cfg=g1_kick_env_cfg(play=True),
  rl_cfg=kick_ppo_runner_cfg(),
)
