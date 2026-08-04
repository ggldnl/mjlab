"""Train a bridging architecture on the parkour experiment.

Builds the arena, loads the three frozen skills, trains the chosen architecture, and
saves the result so the demo can load it back. This script supplies only the
experiment-specific pieces and lets the architecture do the rest.

All three skills must be trained first; there are no analytical stand-ins here.

    uv run train Mjlab-Parkour-Walk
    uv run train Mjlab-Parkour-Run
    uv run train Mjlab-Parkour-Jump
    uv run python -m mjlab.tasks.skills.experiments.parkour.train --architecture 1

arch_0 has nothing to train; running this on it just saves an empty run so the demo has a
checkpoint to point at, uniform with the others.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.architectures import Budgets, train
from mjlab.tasks.skills.architectures.arch_1.switch import SuccessFn
from mjlab.tasks.skills.experiments.parkour import (
  BUDGETS,
  ENTITY_NAME,
  EXPERIMENT_NAME,
  JUMP_TASK_ID,
  RUN_TASK_ID,
  WALK_TASK_ID,
  build_experiment,
)
from mjlab.tasks.skills.experiments.parkour.arena import parkour_arena_env_cfg
from mjlab.tasks.skills.experiments.parkour.controller import JUMP, RUN, WALK
from mjlab.tasks.skills.utils import new_architecture_run_dir


def _make_upright_success(env: ManagerBasedRlEnv, max_tilt: float = 0.6) -> SuccessFn:
  """Still on its feet.

  The floor of what any hand-off into a locomotion skill has to achieve, and the right
  test for walk and run: both can pick up a gait from a wide range of speeds, so there is
  no narrow regime to arrive in, only a robot that has to still be standing.

  This means something only because the caller pairs it with the termination flags. On its
  own it is blind to the failure it exists to catch: the arena auto-resets a fallen robot
  and a freshly spawned one is upright, so reading live state after a fall reports a
  success. `train_switch` latches terminations across the window and ANDs them in.
  """
  entity = env.scene[ENTITY_NAME]

  def success(env: ManagerBasedRlEnv) -> torch.Tensor:
    del env
    tilt = torch.acos((-entity.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
    return tilt < max_tilt

  return success


def _make_jump_success(
  env: ManagerBasedRlEnv,
  max_tilt: float = 0.5,
  max_speed: float = 1.2,
  max_yaw_rate: float = 1.0,
) -> SuccessFn:
  """Upright, slowed down, and pointed where the jump expects to go.

  This is the hand-off the experiment is built around, so it gets the real test. The jump
  tracks a clip that opens from a stand: it crouches, extends, and commits to a ballistic
  arc whose length was decided at takeoff. Arriving with a run's momentum means the crouch
  never happens on the reference's schedule and the takeoff is mistimed, which is a fall
  rather than a short jump.

  `max_speed` is the number that matters, and it is deliberately not zero. The clips do
  open from a stand, but a bridge forced all the way to a standstill is doing what the
  naive hand-off already does badly -- stopping dead -- and throwing away the momentum the
  thesis wants exploited. It is set to a brisk walk.
  """
  entity = env.scene[ENTITY_NAME]

  def success(env: ManagerBasedRlEnv) -> torch.Tensor:
    del env
    tilt = torch.acos((-entity.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
    speed = torch.norm(entity.data.root_link_lin_vel_b[:, :2], dim=-1)
    yaw_rate = entity.data.root_link_ang_vel_b[:, 2].abs()
    return (tilt < max_tilt) & (speed < max_speed) & (yaw_rate < max_yaw_rate)

  return success


@dataclass(frozen=True)
class TrainConfig:
  # Which architecture to train (see architectures/__init__.py). Defaults to 1,
  # the per-target distribution-matching bridge
  architecture: int = 1

  walk_task_id: str = WALK_TASK_ID
  run_task_id: str = RUN_TASK_ID
  jump_task_id: str = JUMP_TASK_ID

  # Skill checkpoints. None picks the latest trained one for that task
  walk_checkpoint: str | None = None
  run_checkpoint: str | None = None
  jump_checkpoint: str | None = None

  # Bridge training runs many parallel envs; more is faster and steadier
  num_envs: int = 4096
  device: str | None = None

  # How long to train, per architecture, defaulting to what this experiment declares in
  # __init__.py. Every field is reachable from the command line, e.g.
  # --budgets.arch-1.bridge.num-iterations 5000. The window plan lives beside it there;
  # it is keyed by skill name, so it is edited in __init__.py rather than overridden here
  budgets: Budgets = BUDGETS


def run_train(cfg: TrainConfig) -> None:
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # No obstacles during training. The architecture practises hand-offs from harvested
  # states, thousands of environments at a time, and never walks the corridor; boxes
  # would only be contacts to trip over in states that have nothing to do with them.
  env_cfg = parkour_arena_env_cfg(obstacles=None)
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  exp = build_experiment(
    env,
    device,
    walk_task_id=cfg.walk_task_id,
    run_task_id=cfg.run_task_id,
    jump_task_id=cfg.jump_task_id,
    walk_checkpoint=cfg.walk_checkpoint,
    run_checkpoint=cfg.run_checkpoint,
    jump_checkpoint=cfg.jump_checkpoint,
  )

  # arch_1 only. Jump is the one the experiment turns on, so it gets the real test; walk
  # and run only have to be arrived at upright.
  success_fns: dict[int, SuccessFn] = {
    WALK: _make_upright_success(env),
    RUN: _make_upright_success(env),
    JUMP: _make_jump_success(env),
  }

  meta = train(env, cfg.architecture, exp, cfg.budgets, success_fns)

  # Saving is common to all architectures: a fresh run directory, then let the
  # architecture write whatever it needs into it (arch_0 writes nothing).
  run_dir = new_architecture_run_dir(EXPERIMENT_NAME, cfg.architecture)
  meta.save(run_dir)
  print(f"[train] architecture {cfg.architecture} saved to {run_dir}")

  env.close()


if __name__ == "__main__":
  run_train(tyro.cli(TrainConfig, config=mjlab.TYRO_FLAGS))
