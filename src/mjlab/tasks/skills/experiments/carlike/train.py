"""Train arch_1 on the diffdrive experiment.

Parked for later: this wires the generic arch_1 training loop
(`skills.architectures.arch_1.train`) to the diffdrive pool, using the analytical
experts so no checkpoints are needed. It has not been run yet, and the success
oracle below is still a placeholder.

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.train
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.skills.architectures.arch_1 import Arch1
from mjlab.tasks.skills.architectures.arch_1.train import train
from mjlab.tasks.skills.experiments.diffdrive.dynamics import (
  analytical_drive,
  analytical_turn,
)
from mjlab.tasks.skills.skill import SkillPool

TASK_ID = "Mjlab-Diffdrive-Drive"
ENTITY_NAME = "robot"


def always_success(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Placeholder success oracle: every hand-off counts as a success.

  TODO Replace with a real "did turn take over safely?" signal (e.g. heading error
  against the commanded angle around the switch), otherwise the switch-decider just
  learns to hand over as early as possible.
  """
  return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)


def main(num_envs: int = 64, device: str | None = None) -> None:
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  pool = SkillPool([analytical_drive(), analytical_turn()])
  meta = Arch1(env, pool)
  train(env, pool, ENTITY_NAME, meta, {i: always_success for i in range(len(pool))})
  env.close()


if __name__ == "__main__":
  main()
