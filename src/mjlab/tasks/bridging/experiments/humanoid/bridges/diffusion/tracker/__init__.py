"""Endpoint-precise universal tracker trained on dynamic LAFAN1 rollouts."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker.env_cfg import (
  tracker_env_cfg,
)
from mjlab.tasks.registry import register_mjlab_task

TRACKER_TASK_ID = "Mjlab-G1-Diffusion-Universal-Tracker"
TRACKER_EXPERIMENT = "g1_diffusion_universal_tracker"


def tracker_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(1024, 512, 256),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.7,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(1024, 512, 256), activation="elu", obs_normalization=True
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
    experiment_name=TRACKER_EXPERIMENT,
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


register_mjlab_task(
  task_id=TRACKER_TASK_ID,
  env_cfg=tracker_env_cfg(),
  play_env_cfg=tracker_env_cfg(play=True, split="eval"),
  rl_cfg=tracker_ppo_runner_cfg(),
)

__all__ = ["TRACKER_EXPERIMENT", "TRACKER_TASK_ID", "tracker_env_cfg"]
