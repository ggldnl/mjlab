"""The kicking skill: a standing G1 strikes a football at a commanded launch velocity.

    uv run train Mjlab-Parkour-Kick --env.scene.num-envs 4096
    uv run play Mjlab-Parkour-Kick

The ball is a size 5 football from the asset zoo, spawned within reach of the right foot,
so there is no locomotion in this task. The robot stands, waits out a stance window, and
then puts the ball where it was asked to: the command is a launch speed and a heading
offset, and the reward scores the ball's own velocity against it.

Unlike the jump, this is reward shaped rather than reference tracked, because there is no
clip to track. That makes exploration the whole problem, and the design answers it with a
three rung ladder and a curriculum that holds the kick weights at zero until standing is
solved. See kick_env_cfg.py for the ladder and mdp.py for why each rung is shaped the way
it is.

Two things to check before concluding that a run has worked:

  Metrics/kick/launch_rate tells you whether the ball is being struck at all. Everything
  else is meaningless until this is high.

  Metrics/kick/vel_error and Metrics/kick/heading_error tell you whether the policy is
  answering the command or has memorised one kick. A policy that ignores the command
  still launches, and still scores on kick_quality whenever the sampled command happens
  to land near its one kick; what it cannot do is drive these two down.

This skill is not wired into the parkour arena. The corridor's bridge reads a shared
proprioception group (see ../../../parkour/arena.py), and the kick's observation carries
ball state that group has no channel for. Adding it means giving the arena a ball, which
is a change to the corridor rather than to this task.
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
  is tuned for, and there is nothing to be gained from guessing at differences before a
  run has shown one.

  If training plateaus, check Loss/learning_rate before touching the reward. The KL
  adaptive schedule clamps to a floor of 1e-5 and prints as 0.00000 once it gets there,
  which looks exactly like a reward problem and is not one. Raising desired_kl to 0.02 and
  dropping num_learning_epochs to 3 is what has worked on this repo before.
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
