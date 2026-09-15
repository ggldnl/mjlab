"""Docking bridge task.

Run

    uv run train Mjlab-G1-Docking-Bridge --env.scene.num-envs 4096
    uv run play Mjlab-G1-Docking-Bridge
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.docking.bridge import (
  DockingBridge as DockingBridge,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.docking.env_cfg import (
  docking_env_cfg,
)
from mjlab.tasks.registry import register_mjlab_task

DOCKING_TASK_ID = "Mjlab-G1-Docking-Bridge"
DOCKING_EXPERIMENT = "g1_docking_bridge"


def docking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
    experiment_name=DOCKING_EXPERIMENT,
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=15_000,
  )


register_mjlab_task(
  task_id=DOCKING_TASK_ID,
  env_cfg=docking_env_cfg(),
  play_env_cfg=docking_env_cfg(play=True, split="eval"),
  rl_cfg=docking_ppo_runner_cfg(),
)
