"""The pushing skill: a G1 pushes along a 1m cube with its hands; contact with other
body parts is not allowed.

    uv run train Mjlab-G1-Push --env.scene.num-envs 4096
    uv run play Mjlab-G1-Push

The cube is the asset zoo's parametric box, given a mass so that it has a freejoint and
physics can move it, spawned a stride or two ahead of the robot in the robot's own
heading frame.

This starts from locomotion rather than from scratch. The environment is mjlab's flat G1
velocity task with the box added; the robot first learns to walk and to track a  commanded
twist and then to push.

The goal is the twist itself, not a second command. The robot is told to travel at a
velocity, the box in front of it has to travel at that velocity too.

Three things to check before concluding that a run has worked:

    Episode_Metrics/push_hands_contact_rate tells you whether the box is being touched by
    the hands at all. Everything else is meaningless until this is high, and a policy that
    has learned to walk around the box scores well on velocity tracking with this near
    zero. One half is a one handed push.

    Episode_Metrics/push_body_contact_rate tells you whether the hands-only constraint took.
    It should fall toward zero. High alongside a healthy displacement means the policy is
    paying the penalty and body checking the box anyway.

    Episode_Metrics/push_box_displacement tells you whether the box is going anywhere. It
    is the furthest the box got in an episode, in metres.
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

  Unchanged apart from the name and the length. This is the velocity task with extra
  reward terms and a slightly wider observation, which is exactly what that config is
  tuned for. The length is higher than the walk task's, because the command
  curriculum has three stages to climb and the box makes each one harder than
  the bare locomotion version.

  If training plateaus, check Loss/learning_rate before touching the reward. The KL
  adaptive schedule clamps to a floor of 1e-5 and prints as 0.00000 once it gets there,
  which looks exactly like a reward problem and is not one.
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
