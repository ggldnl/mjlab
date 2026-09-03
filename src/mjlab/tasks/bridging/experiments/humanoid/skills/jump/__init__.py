"""A goal conditioned jump for the Unitree G1, learned the ASAP way.

ASAP (RSS 2025, https://agile.human2humanoid.com/) gets a G1 to jump by tracking a
retargeted human jump frame by frame: reference state initialization, a dense per frame
tracking reward, and termination the moment tracking is lost.

The clips ship with the repo, already retargeted to a 23 DoF G1. Five of them are forward
jumps of increasing length, which is what turns a single skill tracker into a goal
conditioned one: the policy sees the target displacement, and each episode stretches its
clip so the reachable distances are continuous.

Run

1. Fetch and convert the clips. Writes to data/asap/motions.

    uv run --with joblib python -m \
      mjlab.tasks.bridging.experiments.humanoid.skills.jump.dataset

2. Train.

    uv run train Mjlab-G1-Jump --env.scene.num-envs 4096

3. Watch.

    uv run play Mjlab-G1-Jump
"""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.jump.jump_env_cfg import (
  g1_jump_env_cfg,
)
from mjlab.tasks.registry import register_mjlab_task

JUMP_TASK_ID = "Mjlab-G1-Jump"


def jump_ppo_runner_cfg(
  experiment_name: str = "g1_jump",
) -> RslRlOnPolicyRunnerCfg:
  """PPO for the jump. Close to mjlab's tracking config, since this is a tracking task.

  Two deliberate differences:

    init_std 0.6            1.0 is a lot of per-joint noise every step at this action
                            scale, and the airborne part of the motion has no time to
                            recover from it
    num_learning_epochs 4   fewer chances per iteration for the KL schedule to ratchet
    desired_kl 0.015        the learning rate down to its 1e-5 floor, where a run looks
                            plateaued but is only crawling
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
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
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
    experiment_name=experiment_name,
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=15_000,
  )


register_mjlab_task(
  task_id=JUMP_TASK_ID,
  env_cfg=g1_jump_env_cfg(),
  play_env_cfg=g1_jump_env_cfg(play=True),
  rl_cfg=jump_ppo_runner_cfg(),
)
