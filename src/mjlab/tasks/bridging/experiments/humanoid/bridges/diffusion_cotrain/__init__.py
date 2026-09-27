"""Alternating diffusion planner and physical tracker co-training."""

from dataclasses import dataclass

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.tracker import (
  tracker_ppo_runner_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion_cotrain.env_cfg import (
  cotrain_env_cfg,
  tracker_pretrain_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion_cotrain.training import (
  AlternatingRunner,
)
from mjlab.tasks.registry import register_mjlab_task

PRETRAIN_TASK_ID = "Mjlab-G1-CoTrain-Tracker-Pretrain"
COTRAIN_TASK_ID = "Mjlab-G1-Diffusion-Tracker-CoTrain"
PRETRAIN_EXPERIMENT = "g1_cotrain_tracker_pretrain"
COTRAIN_EXPERIMENT = "g1_diffusion_tracker_cotrain"


@dataclass
class CoTrainRunnerCfg(RslRlOnPolicyRunnerCfg):
  tracker_checkpoint: str = ""
  alternate_every: int = 25
  planner_updates: int = 4
  planner_batch: int = 32
  critic_updates: int = 8
  planner_learning_rate: float = 1e-5
  critic_learning_rate: float = 1e-4
  self_imitation_weight: float = 0.1


def pretrain_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = tracker_ppo_runner_cfg()
  cfg.experiment_name = PRETRAIN_EXPERIMENT
  return cfg


def cotrain_runner_cfg() -> CoTrainRunnerCfg:
  base = tracker_ppo_runner_cfg()
  return CoTrainRunnerCfg(
    seed=base.seed,
    num_steps_per_env=base.num_steps_per_env,
    max_iterations=base.max_iterations,
    obs_groups=base.obs_groups,
    save_interval=base.save_interval,
    experiment_name=COTRAIN_EXPERIMENT,
    actor=base.actor,
    critic=base.critic,
    algorithm=base.algorithm,
  )


register_mjlab_task(
  task_id=PRETRAIN_TASK_ID,
  env_cfg=tracker_pretrain_env_cfg(),
  play_env_cfg=tracker_pretrain_env_cfg(play=True),
  rl_cfg=pretrain_runner_cfg(),
)
register_mjlab_task(
  task_id=COTRAIN_TASK_ID,
  env_cfg=cotrain_env_cfg(),
  play_env_cfg=cotrain_env_cfg(play=True),
  rl_cfg=cotrain_runner_cfg(),
  runner_cls=AlternatingRunner,
)

__all__ = [
  "COTRAIN_EXPERIMENT",
  "COTRAIN_TASK_ID",
  "PRETRAIN_EXPERIMENT",
  "PRETRAIN_TASK_ID",
]
