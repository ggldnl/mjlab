"""
RL configuration for the Crawler velocity task.

Defines the learning algorithm. Here we can set PPO hyperparameters
(learning rate, clip range, GAE lambda, mini-batches, etc.) and the
neural network architecture (hidden layers, activation functions).
It's fully decoupled from the environment — we could swap in SAC
or any other algorithm without touching anything else.
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def crawler_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for the Crawler velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.75,
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      entropy_coef=0.005,
    ),
    experiment_name="crawler_velocity_ppo",
    max_iterations=10_000,
  )
