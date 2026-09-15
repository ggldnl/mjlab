"""Time-aligned bridge trained from demonstrated robot trajectories."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.env_cfg import (
  imitation_env_cfg,
)
from mjlab.tasks.registry import register_mjlab_task

IMITATION_TASK_ID = "Mjlab-G1-Imitation-Bridge"
IMITATION_EXPERIMENT = "g1_imitation_bridge"


def imitation_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
      entropy_coef=0.005,
      num_learning_epochs=4,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      desired_kl=0.015,
    ),
    experiment_name=IMITATION_EXPERIMENT,
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=15_000,
  )


register_mjlab_task(
  task_id=IMITATION_TASK_ID,
  env_cfg=imitation_env_cfg(),
  play_env_cfg=imitation_env_cfg(play=True, split="eval"),
  rl_cfg=imitation_ppo_runner_cfg(),
)
