"""A goal-conditioned jump for the Unitree G1, learned the ASAP way.

    uv run --with joblib python -m mjlab.tasks.bridging.experiments.humanoid.skills.jump.dataset
    uv run train Mjlab-Parkour-Jump --env.scene.num-envs 4096
    uv run play Mjlab-Parkour-Jump

ASAP (RSS 2025, https://agile.human2humanoid.com/) gets a G1 to jump by tracking a
retargeted human jump frame by frame: reference-state initialization, a dense
per-frame tracking reward, and termination the moment tracking is lost. That is the
whole of stage one, and it is what produced the jumps in their videos. The delta
action model that gives ASAP its name is stage two and is about the sim-to-real gap;
it is not needed to make the jump happen in simulation.

The clips ship with the repo, already retargeted to a 23-DoF G1. Five of them are
forward jumps of increasing length, which is what turns a single-skill tracker into
a goal-conditioned one: the policy sees the target displacement, and each episode
stretches its clip so the reachable distances are continuous.
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

JUMP_TASK_ID = "Mjlab-Parkour-Jump"


def jump_ppo_runner_cfg(
  experiment_name: str = "parkour_jump",
) -> RslRlOnPolicyRunnerCfg:
  """PPO for the jump.

  Close to mjlab's tracking config, which is the right starting point because this
  is a tracking task. Two deliberate differences.

  `init_std` is 0.6 rather than 1.0. At this action scale a unit standard deviation
  is a large amount of noise on every joint every step, and the airborne part of the
  motion has no time to recover from it.

  `num_learning_epochs` is 4 and `desired_kl` is 0.015. The KL-adaptive schedule
  walks the learning rate down to its 1e-5 floor on tasks with wide, mixed-unit
  observations, and a rate stuck at the floor looks exactly like a plateau. Fewer
  epochs is fewer chances to ratchet it down within one iteration.
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
