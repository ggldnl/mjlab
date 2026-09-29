"""Universal trajectory tracker trained on retargeted motion clips."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.bridging.bridges.diffusion.config import (
  tracker_experiment,
  tracker_task_id,
)
from mjlab.tasks.bridging.bridges.diffusion.tracker.env_cfg import (
  tracker_env_cfg,
)
from mjlab.tasks.bridging.config import ROBOTS
from mjlab.tasks.registry import register_mjlab_task

TRACKER_TASK_ID = tracker_task_id("g1")
TRACKER_EXPERIMENT = tracker_experiment("g1")


def tracker_ppo_runner_cfg(robot: str = "g1") -> RslRlOnPolicyRunnerCfg:
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
    experiment_name=tracker_experiment(robot),
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


for _robot in ROBOTS:
  register_mjlab_task(
    task_id=tracker_task_id(_robot),
    env_cfg=tracker_env_cfg(robot=_robot),
    play_env_cfg=tracker_env_cfg(play=True, split="eval", robot=_robot),
    rl_cfg=tracker_ppo_runner_cfg(_robot),
  )

__all__ = ["TRACKER_EXPERIMENT", "TRACKER_TASK_ID", "tracker_env_cfg"]
