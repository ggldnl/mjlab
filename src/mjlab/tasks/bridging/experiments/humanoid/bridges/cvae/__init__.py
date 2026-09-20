"""Residual CVAE bridge distilled online from several trajectory trackers.

Rebuild the tracker corpus once so it contains contacts and active actions:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.trajectory_tracking.collect

Edit configs/cvae_teachers.yaml to select clip and checkpoint pairs, then train:

    uv run train Mjlab-G1-CVAE-Bridge --env.scene.num-envs 4096

Each teacher gets 128 parallel environments by default. Equal-sized groups are pooled
for one student update, while each frozen tracker labels its own visited states.
"""

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import (
  RslRlBaseRunnerCfg,
  RslRlDistillationAlgorithmCfg,
  RslRlModelCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.env_cfg import (
  cvae_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.runner import CvaeRunner
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.teachers import (
  DEFAULT_MANIFEST,
)
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.rl_cfg import (
  unitree_g1_tracking_ppo_runner_cfg,
)

CVAE_TASK_ID = "Mjlab-G1-CVAE-Bridge"
CVAE_EXPERIMENT = "g1_cvae_bridge"


@dataclass
class ResidualCvaeModelCfg:
  class_name: str = (
    "mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.model:ResidualCvaeModel"
  )
  hidden_dims: tuple[int, ...] = (512, 256, 128)
  latent_dim: int = 16
  activation: str = "elu"
  obs_normalization: bool = True
  posterior_obs_set: str = "posterior"


@dataclass
class CvaeDistillationCfg(RslRlDistillationAlgorithmCfg):
  class_name: str = (
    "mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.algorithm:CvaeDistillation"
  )
  loss_type: Literal["mse", "huber"] = "huber"
  gradient_length: int = 15
  max_grad_norm: float | None = 1.0
  kl_beta_start: float = 1.0e-4
  kl_beta_end: float = 1.0e-2
  kl_schedule_updates: int = 5000
  action_continuity_weight: float = 0.1


@dataclass
class CvaeRunnerCfg(RslRlBaseRunnerCfg):
  student: ResidualCvaeModelCfg = field(default_factory=ResidualCvaeModelCfg)
  teacher: RslRlModelCfg = field(
    default_factory=lambda: unitree_g1_tracking_ppo_runner_cfg().actor
  )
  algorithm: CvaeDistillationCfg = field(default_factory=CvaeDistillationCfg)
  teacher_manifest: str = str(DEFAULT_MANIFEST)


def cvae_runner_cfg() -> CvaeRunnerCfg:
  return CvaeRunnerCfg(
    obs_groups={
      "student": ("actor",),
      "posterior": ("posterior",),
      "teacher": ("teacher",),
    },
    experiment_name=CVAE_EXPERIMENT,
    save_interval=200,
    num_steps_per_env=30,
    max_iterations=10_000,
  )


register_mjlab_task(
  task_id=CVAE_TASK_ID,
  env_cfg=cvae_env_cfg(),
  play_env_cfg=cvae_env_cfg(play=True, split="eval"),
  rl_cfg=cvae_runner_cfg(),
  runner_cls=CvaeRunner,
)
