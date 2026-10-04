"""Push a 1 m box a variable number of whole cells straight ahead, hands only.

The G1 mirror of Mjlab-T1-Push: same command, observations, rewards and scene, built by
the shared builder in config/t1/skills/push. See that package for how the task works.

    push_env_cfg.py   the G1 arm pose, hands and box numbers

The twist conditioned push is Mjlab-G1-Push-Twist, in push_twist.

Run

1. Train.

    uv run train Mjlab-G1-Push --env.scene.num-envs 4096

2. Play the latest checkpoint, or a fixed distance.

    uv run play Mjlab-G1-Push
    uv run play Mjlab-G1-Push --env.commands.push.min-cells 3 --env.commands.push.max-cells 3
"""

from dataclasses import replace

from mjlab.tasks.bridging.config.g1.skills.push.push_env_cfg import g1_push_env_cfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

PUSH_TASK_ID = "Mjlab-G1-Push"

register_mjlab_task(
  task_id=PUSH_TASK_ID,
  env_cfg=g1_push_env_cfg(),
  play_env_cfg=g1_push_env_cfg(play=True),
  rl_cfg=replace(
    unitree_g1_ppo_runner_cfg(),
    experiment_name="g1_push",
    save_interval=200,
    max_iterations=15_000,
  ),
  runner_cls=VelocityOnPolicyRunner,
)
