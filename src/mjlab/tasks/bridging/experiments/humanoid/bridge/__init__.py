"""The bridge: one policy that gets the robot from where skill A stopped to where skill B
can start.

Input is proprioception plus a target state (root pose, root velocities, joint angles,
joint rates) and a countdown. Output is joint targets. Only the arrival is scored; what
the robot does in between is free.

Run:

    1. Build a dataset of states the robot can actually be in.

       uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.skills

    2. Calibrate the arrival tolerances against that dataset. Skipping this is how a
       robot standing still ends up scoring the same as a trained policy.

       uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.evaluate \
         --calibrate True

       Paste the printed Tolerances(...) into mdp/commands.py.

    3. Train.

       uv run train Mjlab-G1-Bridge --env.scene.num-envs 4096

    4. Score it, always next to the do-nothing baseline.

       uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.evaluate

    5. Watch it. The target is drawn as a translucent robot.

       uv run play Mjlab-G1-Bridge

Layout:

    datasets/      where the start and target states come from
    mdp/           the window, the reward, the terminations
    env_cfg.py     the mjlab task
    evaluate.py    scoring and tolerance calibration
    warm_start.py  the runner, plus an optional actor seed from a locomotion policy

Two design rules, both learned from failed attempts:

  Nothing scores the middle. Earlier versions asked a model to reproduce the motion a
  human recorded between two frames. Many motions connect two states, so a squared error
  against one of them penalizes the others, and their average puts a foot through the
  floor.

  Endpoints come from rollouts of trained policies, never from motion capture. A
  retargeted clip is a description, not a state the G1 is ever in, and two such frames
  are usually an impossible pair.

Not built yet: a kinematic reference stage to densify the reward, and chaining against
live skills instead of dataset states.
"""

from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.bridging.experiments.humanoid.bridge.env_cfg import bridge_env_cfg
from mjlab.tasks.bridging.experiments.humanoid.bridge.warm_start import (
  BridgeOnPolicyRunner,
  BridgeRunnerCfg,
)
from mjlab.tasks.registry import register_mjlab_task

BRIDGE_TASK_ID = "Mjlab-G1-Bridge"


def bridge_ppo_runner_cfg() -> BridgeRunnerCfg:
  """PPO for the bridge. Copied from the jump: same robot, same control rate, both goal
  conditioned.

  Three settings that are not the rsl_rl defaults:

    init_std 0.6            1.0 is too much per-joint noise at this action scale, and the
                            shortest window is 0.3 s, too short to recover from it
    num_learning_epochs 4   fewer chances per iteration for the KL schedule to ratchet
    desired_kl 0.015        the learning rate down to rsl_rl's 1e-5 floor, where a run
                            looks plateaued but is only crawling. Read the learning rate
                            first when a run stalls.

  warm_start is off by default. See warm_start.py.
  """
  return BridgeRunnerCfg(
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
  runner_cls=BridgeOnPolicyRunner,
)
