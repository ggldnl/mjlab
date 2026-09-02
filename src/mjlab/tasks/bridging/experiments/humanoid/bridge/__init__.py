"""One policy that drives the robot from where a skill stopped to where the next can start.

Task id `Mjlab-G1-Bridge`, checkpoints under `logs/rsl_rl/g1_bridge`.

    in     own state (root velocities, gravity, joint angles and rates, last action)
           + gap to the target state
           + seconds left, and how long the window was
    out    joint position targets, 29 of them

One episode is one window: teleport onto a start state, cross to a target state, deadline
in between. Only the arrival is scored. What the robot does in the middle is free.

Run
---

    1. Build the corpus. See datasets/tracker.py, which has the full recipe.

        uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.tracker

    2. Check what came out before training on it. Per-source counts, then a window drawn
       as a ghost.

        uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.view

    3. Train.

        uv run train Mjlab-G1-Bridge --env.scene.num-envs 4096

    4. Score it against a robot that does nothing. A bridge that cannot beat the statue
       has not learned anything.

        uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.evaluate

    5. Watch it. Amber ghost is the target, blue ghost walks the recorded crossing.

        uv run play Mjlab-G1-Bridge

Layout
------

    datasets/      where start and target states come from
    mdp/           the window (commands), what it pays (rewards), how it ends (terminations)
    env_cfg.py     the mjlab task
    evaluate.py    scoring against the do-nothing baseline

Interface
---------

Aim the bridge from outside with two calls on the command term:

    place(env_ids, start, target, duration_s)    teleport onto a start, then cross
    open_window(env_ids, duration_s)             cross from wherever the robot already is

Durations are seconds, not control ticks. A tick count changes with the decimation and
means nothing to a caller choosing between a 1.0 s target and a 0.5 s one.
`BridgeCommand.steps_for` is the only conversion in the task.

Design rules
------------

Nothing scores the middle. Many motions connect two states, so scoring one of them
penalizes the rest. Earlier versions regressed the recorded in-between and put a foot
through the floor in 2 windows out of 5.

Endpoints come from tracker rollouts, never from motion capture. A retargeted human clip
is a description, not a state a G1 is ever in.

Both endpoints come from one rollout, a fixed time apart. Two individually reachable states
can be an unreachable pair. A contiguous segment needs no feasibility model: the
displacement, velocity change, joint travel and time available were demonstrated together.

The recorded crossing is a training signal, never an input. `mdp.guidance` pays for staying
near it and anneals to zero, so the policy reads only its own state and the gap to the
target, at training time and at inference alike.
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.bridging.experiments.humanoid.bridge.env_cfg import bridge_env_cfg
from mjlab.tasks.registry import register_mjlab_task

BRIDGE_TASK_ID = "Mjlab-G1-Bridge"


def bridge_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO for the bridge. Copied from the jump: same robot, same rate, both goal conditioned.

  Three settings differ from the rsl_rl defaults:

      init_std 0.6            1.0 is too much per-joint noise at this action scale, and the
                              shortest window is 0.3 s, too short to recover from it
      num_learning_epochs 4   fewer chances per iteration for the KL schedule to ratchet
      desired_kl 0.015        the learning rate down to rsl_rl's 1e-5 floor, where a run
                              looks plateaued but is only crawling

  Read the learning rate first when a run stalls.
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
