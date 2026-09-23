"""Universal privileged oracle for CVAE DAgger.

Train on every compatible LAFAN motion with:

    uv run train Mjlab-G1-CVAE-Oracle --env.scene.num-envs 4096
"""

from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.oracle.env_cfg import (
  oracle_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.oracle.runner import (
  OracleRunner,
)
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

ORACLE_TASK_ID = "Mjlab-G1-CVAE-Oracle"
ORACLE_EXPERIMENT = "g1_cvae_oracle"


def oracle_runner_cfg():
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = ORACLE_EXPERIMENT
  cfg.max_iterations = 15_000
  return cfg


register_mjlab_task(
  task_id=ORACLE_TASK_ID,
  env_cfg=oracle_env_cfg(),
  play_env_cfg=oracle_env_cfg(play=True),
  rl_cfg=oracle_runner_cfg(),
  runner_cls=OracleRunner,
)
