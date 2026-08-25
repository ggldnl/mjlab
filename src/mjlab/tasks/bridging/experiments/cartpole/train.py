"""Train a bridging architecture on the cartpole experiment.

Builds the skill pool (analytical experts by default), trains the chosen architecture,
and saves the result so the demo can load it back. This script supplies only the
experiment-specific pieces and lets the architecture do the rest.

arch_0 has nothing to train; running this on it just saves an empty run so the demo has a
checkpoint to point at, uniform with the others.

    uv run python -m mjlab.tasks.bridging.experiments.cartpole.train --architecture 1

Use trained RL skills instead of the analytical experts (discouraged):

    uv run python -m mjlab.tasks.bridging.experiments.cartpole.train \\
        --architecture 1 --no-analytical
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.architectures import Budgets, train
from mjlab.tasks.bridging.architectures.arch_1.switch import SuccessFn, always_ok
from mjlab.tasks.bridging.experiments.cartpole import (
  BALANCE_TASK_ID,
  BUDGETS,
  ENTITY_NAME,
  EXPERIMENT_NAME,
  SPINUP_TASK_ID,
  build_experiment,
)
from mjlab.tasks.bridging.experiments.cartpole.controller import BALANCE, SPIN_UP
from mjlab.tasks.bridging.utils import new_architecture_run_dir
from mjlab.tasks.registry import load_env_cfg


def _make_balance_success(
  env: ManagerBasedRlEnv, upright_cos: float = 0.9, max_speed: float = 2.0
) -> SuccessFn:
  """Hand-off into balance succeeds when the pole ends up up and slow.

  Up (hinge cosine above upright_cos) and slow (hinge speed below max_speed) is exactly
  the basin the LQR balancer can hold, so this rewards the decider for handing over only
  once the pole is genuinely catchable. The joint index is resolved once here.
  """
  entity = env.scene[ENTITY_NAME]
  hinge = entity.find_joints("hinge_1")[0][0]

  def success(env: ManagerBasedRlEnv) -> torch.Tensor:
    del env
    angle = entity.data.joint_pos[:, hinge]
    speed = entity.data.joint_vel[:, hinge].abs()
    return (torch.cos(angle) > upright_cos) & (speed < max_speed)

  return success


@dataclass(frozen=True)
class TrainConfig:
  # Which architecture to train (see architectures/__init__.py). Defaults to 1,
  # the per-target distribution-matching bridge
  architecture: int = 1

  # Use the analytical experts from dynamics.py. With --no-analytical, train the
  # architecture on top of the frozen RL checkpoints instead (discouraged)
  analytical: bool = True

  spinup_task_id: str = SPINUP_TASK_ID
  balance_task_id: str = BALANCE_TASK_ID

  # Checkpoints for the non-analytical skills. None picks the latest trained one
  spinup_checkpoint: str | None = None
  balance_checkpoint: str | None = None

  # Bridge training runs many parallel envs; more is faster and steadier
  num_envs: int = 1024
  device: str | None = None

  # How long to train, per architecture, defaulting to what this experiment declares in
  # __init__.py. Every field is reachable from the command line, e.g.
  # --budgets.arch-1.bridge.num-iterations 1000. The window plan lives beside it there;
  # it is keyed by skill name, so it is edited in __init__.py rather than overridden here
  budgets: Budgets = BUDGETS


def run_train(cfg: TrainConfig) -> None:
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Built from the swingup task so the pool trains from the pole hanging, the same arena
  # the demo runs in.
  env_cfg = load_env_cfg(cfg.spinup_task_id, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  exp = build_experiment(
    env,
    device,
    analytical=cfg.analytical,
    spinup_task_id=cfg.spinup_task_id,
    balance_task_id=cfg.balance_task_id,
    spinup_checkpoint=cfg.spinup_checkpoint,
    balance_checkpoint=cfg.balance_checkpoint,
  )

  # arch_1 only. balance is the target the demo bridges into, so it gets the real "is the
  # pole catchable?" test; spin_up gets the trivial one, since it pumps energy from any
  # state and is never actually a bridge target here.
  success_fns: dict[int, SuccessFn] = {
    SPIN_UP: always_ok,
    BALANCE: _make_balance_success(env),
  }

  meta = train(env, cfg.architecture, exp, cfg.budgets, success_fns)

  # Saving is common to all architectures: a fresh run directory, then let the
  # architecture write whatever it needs into it (arch_0 writes nothing).
  run_dir = new_architecture_run_dir(EXPERIMENT_NAME, cfg.architecture)
  meta.save(run_dir)
  print(f"[train] architecture {cfg.architecture} saved to {run_dir}")

  env.close()


if __name__ == "__main__":
  run_train(tyro.cli(TrainConfig))
