"""RL configuration for the crawler velocity tasks."""

from dataclasses import replace

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def crawler_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO runner for the classic crawler velocity task.

  Exploration is deliberately gentle: a low initial action std keeps early
  actions small so the tip-prone robot survives long enough to bootstrap a gait
  (with the usual ~1.0 std it tips within a handful of control steps before any
  learning signal accumulates).
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.3,
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="crawler_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
  )


def crawler_abstraction_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO runner for the abstraction-guided crawler task.

  Identical hyperparameters to the classic runner (the difference is in the
  environment's observations and rewards, not the learner); only the experiment
  name changes so the two runs log separately.
  """
  return replace(
    crawler_ppo_runner_cfg(), experiment_name="crawler_velocity_abstraction"
  )
