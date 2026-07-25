"""Train a bridging architecture on the parkour experiment.

Builds the skill pool, builds the chosen architecture by id, runs *that
architecture's own* training, and saves the result so the demo can load it back.
Training is common to every architecture: this script only supplies the
experiment-specific pieces (the pool, the entity to harvest states from, and a
per-target success oracle) and lets the architecture do the rest.

arch_0 (the direct hand-off baseline) has nothing to train; running this on it just
saves an empty run so the demo has a checkpoint to point at, uniform with the others.

    uv run python -m mjlab.tasks.skills.experiments.parkour.train --architecture 0

arch_1 carries out the experiment since it has bridges to merge the individual skills.

    uv run python -m mjlab.tasks.skills.experiments.parkour.train --architecture 1
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.skills.architectures import ARCHITECTURES, TRAINERS
from mjlab.tasks.skills.experiments.parkour import (
  ENTITY_NAME,
  EXPERIMENT_NAME,
  JUMP_TASK_ID,
  RUN_TASK_ID,
  WALK_TASK_ID,
  build_pool,
)
from mjlab.tasks.skills.experiments.parkour.controller import JUMP, RUN, WALK
from mjlab.tasks.skills.utils import new_architecture_run_dir

# A success oracle: given the env after a bridging window, returns a bool per env
# saying whether the target skill actually took over safely. Privileged and external,
# never read off the skill itself (see Skill).
SuccessFn = Callable[[ManagerBasedRlEnv], torch.Tensor]


def _make_standing_success(
  env: ManagerBasedRlEnv, min_height: float = 0.5, max_tilt: float = 0.8
) -> SuccessFn:
  """Hand-off into a locomotion skill succeeds when the robot is still on its feet.

  Every skill here (walk, run, jump) is only usable while the robot is upright and has
  not collapsed, so "took over safely" is exactly "did not fall": at the end of the
  bridging window the base is above `min_height` [m] and tilted less than `max_tilt`
  [rad] from upright. A hand-off that leaves the humanoid stumbling or on the ground
  fails, which is what pushes the switch-decider to hand over only from a recoverable
  state. Reads base height and orientation off the entity directly.
  """
  entity = env.scene[ENTITY_NAME]

  def success(env: ManagerBasedRlEnv) -> torch.Tensor:
    height = entity.data.root_link_pos_w[:, 2]
    # projected_gravity_b[:, 2] is -1 when perfectly upright; tilt = acos(-that).
    tilt = torch.acos((-entity.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
    return (height > min_height) & (tilt < max_tilt)

  return success


@dataclass(frozen=True)
class TrainConfig:
  # Which architecture to train (see architectures/__init__.py). Defaults to 1,
  # the per-target distribution-matching bridge
  architecture: int = 1

  walk_task_id: str = WALK_TASK_ID
  run_task_id: str = RUN_TASK_ID
  jump_task_id: str = JUMP_TASK_ID

  # Checkpoints for the skills. None picks the latest trained one
  walk_checkpoint: str | None = None
  run_checkpoint: str | None = None
  jump_checkpoint: str | None = None

  # Bridge training runs many parallel envs; more is faster and steadier
  num_envs: int = 1024
  device: str | None = None


def run_train(cfg: TrainConfig) -> None:
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  if cfg.architecture not in ARCHITECTURES:
    raise ValueError(
      f"Unknown architecture {cfg.architecture}; registered: {sorted(ARCHITECTURES)}."
    )

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Built from the walk task; all three skills share one arena.
  env_cfg = load_env_cfg(cfg.walk_task_id, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  pool = build_pool(
    env,
    device,
    walk_task_id=cfg.walk_task_id,
    run_task_id=cfg.run_task_id,
    jump_task_id=cfg.jump_task_id,
    walk_checkpoint=cfg.walk_checkpoint,
    run_checkpoint=cfg.run_checkpoint,
    jump_checkpoint=cfg.jump_checkpoint,
  )

  # One success oracle per target skill. Every skill needs the robot on its feet to
  # take over, so they share the same "did not fall" test.
  standing = _make_standing_success(env)
  success_fns: dict[int, SuccessFn] = {WALK: standing, RUN: standing, JUMP: standing}

  meta = ARCHITECTURES[cfg.architecture](env, pool)
  TRAINERS[cfg.architecture](env, pool, ENTITY_NAME, meta, success_fns)

  # Saving is common to all architectures: a fresh run directory, then let the
  # architecture write whatever it needs into it (arch_0 writes nothing).
  run_dir = new_architecture_run_dir(EXPERIMENT_NAME, cfg.architecture)
  meta.save(run_dir)
  print(f"[train] architecture {cfg.architecture} saved to {run_dir}")

  env.close()


if __name__ == "__main__":
  run_train(tyro.cli(TrainConfig))
