"""Pushing: a G1 drives a 1 m crate along with its hands. Any other contact is illegal.

The crate is the asset zoo's parametric box, given a mass so it gets a freejoint and
physics can move it, spawned a stride or two ahead of the robot in the robot's own heading
frame.

This starts from locomotion rather than from scratch. The environment is mjlab's flat G1
velocity task with the crate added, so the robot first learns to walk and track a commanded
twist, then to push. The goal is that twist, not a second command: the robot is told to
travel at a velocity, and the crate in front of it has to travel at that velocity too.

What to check:

    Episode_Metrics/push_hands_contact_rate   is the crate being touched by the hands at
                                              all. Everything else is meaningless until
                                              this is high
    Episode_Metrics/push_body_contact_rate    are the hands the only body part touching
                                              the crate. Should fall toward zero. High
                                              alongside a healthy displacement means
                                              the policy is paying the penalty and the
                                              body is checking the crate anyway
    Episode_Metrics/push_box_displacement     is the crate going anywhere. Furthest it got
                                              in an episode, in metres

Run

    uv run train Mjlab-G1-Push --env.scene.num-envs 4096
    uv run play Mjlab-G1-Push
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.push.push_env_cfg import (
  g1_push_env_cfg,
)
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

PUSH_TASK_ID = "Mjlab-G1-Push"


def push_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The stock G1 velocity PPO config, under its own experiment name.

  Unchanged apart from the name and the length. This is the velocity task with extra reward
  terms and a slightly wider observation, which is what that config is tuned for. Longer
  than the walk task because the command curriculum has three stages to climb and the crate
  makes each one harder than the bare locomotion version.

  If training plateaus, read Loss/learning_rate before touching the reward. The KL adaptive
  schedule clamps to a floor of 1e-5 and prints as 0.00000 once it gets there, which looks
  exactly like a reward problem and is not one.
  """
  return replace(
    unitree_g1_ppo_runner_cfg(),
    experiment_name="g1_push",
    save_interval=200,
    max_iterations=15_000,
  )


register_mjlab_task(
  task_id=PUSH_TASK_ID,
  env_cfg=g1_push_env_cfg(),
  play_env_cfg=g1_push_env_cfg(play=True),
  rl_cfg=push_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
