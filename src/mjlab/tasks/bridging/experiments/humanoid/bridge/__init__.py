"""Bridge task. One policy that drives the robot from a start dynamic state to a target
dynamic state, in a given time period.

Task id Mjlab-G1-Bridge, checkpoints under logs/rsl_rl/g1_bridge.

    in     state (root velocities, gravity, joint angles and rates, last action)
           + per channel gap to the target state
           + seconds left + window length
    out    29 joint position targets

One episode is one window: teleport onto a start state, reach the target state before the
deadline.

Endpoints come from tracker rollouts, never from motion capture. A retargeted human clip is
a description, not a state a G1 is ever in. Both endpoints come from one rollout a fixed
time apart, which makes the pair reachable by construction. The motion recorded between
them is used as a learning signal (mdp.guidance, annealed to zero), never as an input: the
policy reads only its own state and the gap to the target, in training and at inference
alike.

Run

1. Build the corpus. Look at datasets/tracker.py.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.tracker

2. Inspect it: per source counts, then a window replayed as a ghost.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.view

3. Train.

    uv run train Mjlab-G1-Bridge --env.scene.num-envs 4096

4. Score it against a robot that does nothing.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.evaluate

5. Watch it. Amber ghost is the target, blue ghost is the recorded crossing.

    uv run play Mjlab-G1-Bridge

Layout

    datasets/      where start and target states come from
    mdp/           commands (the window), rewards, terminations
    env_cfg.py     the mjlab task
    evaluate.py    scoring against the do nothing baseline

API

Two methods on the command term drive the bridge from outside:

    place(env_ids, start, target, duration_s)    teleport onto a start, then cross
    open_window(env_ids, duration_s)             cross from wherever the robot already is

Durations are seconds, not control ticks. Tick counts change with the decimation.
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.bridging.experiments.humanoid.bridge.env_cfg import bridge_env_cfg
from mjlab.tasks.registry import register_mjlab_task

BRIDGE_TASK_ID = "Mjlab-G1-Bridge"


def bridge_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO config. Copied from the jump: same robot, same rate, both goal conditioned.

  Three values differ from the rsl_rl defaults:

      init_std 0.6            1.0 is too much per joint noise at this action scale, and
                              the shortest window (0.3 s) is too short to recover from it
      num_learning_epochs 4   fewer chances per iteration for the KL schedule to ratchet
      desired_kl 0.015        the learning rate down to the rsl_rl floor of 1e-5, where a
                              run looks plateaued but is only crawling

  Check the learning rate first when a run stalls.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.6,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=4,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.015,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_bridge",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=15_000,
  )


register_mjlab_task(
  task_id=BRIDGE_TASK_ID,
  env_cfg=bridge_env_cfg(),
  play_env_cfg=bridge_env_cfg(play=True, split="eval"),
  rl_cfg=bridge_ppo_runner_cfg(),
)
