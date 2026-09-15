"""Terminal residual correction over the frozen imitation bridge.

Run

    uv run train Mjlab-G1-Residual-Bridge --env.scene.num-envs 4096
    uv run play Mjlab-G1-Residual-Bridge
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.residual.env_cfg import (
  residual_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.residual.runner import (
  ResidualRunner,
)
from mjlab.tasks.registry import register_mjlab_task

RESIDUAL_TASK_ID = "Mjlab-G1-Residual-Bridge"
RESIDUAL_EXPERIMENT = "g1_residual_bridge"


def residual_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.15,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      entropy_coef=0.002,
      num_learning_epochs=4,
      num_mini_batches=4,
      learning_rate=5.0e-4,
      desired_kl=0.01,
    ),
    experiment_name=RESIDUAL_EXPERIMENT,
    save_interval=200,
    num_steps_per_env=16,
    max_iterations=10_000,
  )


register_mjlab_task(
  task_id=RESIDUAL_TASK_ID,
  env_cfg=residual_env_cfg(),
  play_env_cfg=residual_env_cfg(play=True, split="eval"),
  rl_cfg=residual_ppo_runner_cfg(),
  runner_cls=ResidualRunner,
)
