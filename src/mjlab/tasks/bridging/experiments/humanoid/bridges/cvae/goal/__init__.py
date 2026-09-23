"""Goal-conditioned CVAE distilled from the trajectory oracle.

Collect physical oracle rollouts with goal.collect, then train with:

    uv run train Mjlab-G1-Goal-CVAE-Bridge \
      --agent.teacher-checkpoint logs/rsl_rl/g1_cvae_oracle/<run>/model_5000.pt
"""

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import (
  RslRlBaseRunnerCfg,
  RslRlDistillationAlgorithmCfg,
  RslRlModelCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.env_cfg import (
  goal_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.runner import (
  GoalRunner,
)
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.rl_cfg import (
  unitree_g1_tracking_ppo_runner_cfg,
)

GOAL_CVAE_TASK_ID = "Mjlab-G1-Goal-CVAE-Bridge"
GOAL_CVAE_EXPERIMENT = "g1_goal_cvae_bridge"


@dataclass
class GoalCvaeModelCfg:
  class_name: str = (
    "mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.model:GoalCvaeModel"
  )
  hidden_dims: tuple[int, ...] = (512, 256, 128)
  latent_dim: int = 16
  route_slots: int = 6
  posterior_obs_set: str = "posterior"
  route_obs_set: str = "route"
  activation: str = "elu"
  obs_normalization: bool = True


@dataclass
class GoalDistillationCfg(RslRlDistillationAlgorithmCfg):
  class_name: str = "mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.algorithm:GoalDistillation"
  loss_type: Literal["mse", "huber"] = "huber"
  gradient_length: int = 15
  max_grad_norm: float | None = 1.0
  kl_beta_start: float = 1.0e-4
  kl_beta_end: float = 1.0e-2
  kl_schedule_updates: int = 5000
  route_weight: float = 1.0
  foot_path_weight: float = 0.1
  upper_action_weight: float = 0.3


@dataclass
class GoalRunnerCfg(RslRlBaseRunnerCfg):
  student: GoalCvaeModelCfg = field(default_factory=GoalCvaeModelCfg)
  teacher: RslRlModelCfg = field(
    default_factory=lambda: unitree_g1_tracking_ppo_runner_cfg().actor
  )
  algorithm: GoalDistillationCfg = field(default_factory=GoalDistillationCfg)
  teacher_checkpoint: str = ""


def goal_runner_cfg() -> GoalRunnerCfg:
  return GoalRunnerCfg(
    obs_groups={
      "student": ("actor", "initial"),
      "posterior": ("posterior",),
      "route": ("route",),
      "teacher": ("teacher",),
    },
    experiment_name=GOAL_CVAE_EXPERIMENT,
    save_interval=200,
    num_steps_per_env=30,
    max_iterations=10_000,
  )


register_mjlab_task(
  task_id=GOAL_CVAE_TASK_ID,
  env_cfg=goal_env_cfg(),
  play_env_cfg=goal_env_cfg(play=True, split="eval"),
  rl_cfg=goal_runner_cfg(),
  runner_cls=GoalRunner,
)
