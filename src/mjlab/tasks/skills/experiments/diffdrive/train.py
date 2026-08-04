"""Train a bridging architecture on the diffdrive experiment.

Builds the skill pool (analytical experts by default), trains the chosen architecture,
and saves the result so the demo can load it back. This script supplies only the
experiment-specific pieces and lets the architecture do the rest.

arch_0 has nothing to train; running this on it just saves an empty run so the demo has a
checkpoint to point at, uniform with the others.

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.train --architecture 1
    uv run python -m mjlab.tasks.skills.experiments.diffdrive.train --architecture 3 \\
        --budgets.arch-3.tail-steps 128

Use trained RL skills instead of the analytical experts (discouraged):

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.train \\
        --architecture 1 --no-analytical
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.skills.architectures import Budgets, train
from mjlab.tasks.skills.architectures.arch_1.switch import SuccessFn, always_ok
from mjlab.tasks.skills.experiments.diffdrive import (
  BUDGETS,
  DRIVE_TASK_ID,
  EXPERIMENT_NAME,
  TURN_TASK_ID,
  build_experiment,
)
from mjlab.tasks.skills.experiments.diffdrive.controller import DRIVE_STRAIGHT, TURN
from mjlab.tasks.skills.utils import new_architecture_run_dir


def _make_turn_success(
  env: ManagerBasedRlEnv, max_tilt: float = 0.4, max_speed: float = 0.6
) -> SuccessFn:
  """Hand-off into turn succeeds when the robot ends up upright and slow.

  Upright (tilt below max_tilt [rad]) and slow (forward speed below max_speed [m/s]) is
  exactly turn's safe regime: the arc's lateral acceleration only stays under the tip-over
  threshold at low speed. This rewards the decider for handing over only once the robot
  has been braked down.

  This only means anything because the caller pairs it with the termination flags. On its
  own it is blind to the very failure the experiment is built around: the env auto-resets
  a tipped robot, and a freshly spawned one is upright and stationary, so an oracle
  reading live state after a tip reports a success. `train_switch` latches terminations
  across the window and ANDs them in.
  """
  entity = env.scene["robot"]

  def success(env: ManagerBasedRlEnv) -> torch.Tensor:
    del env
    # projected_gravity_b[:, 2] is -1 when perfectly upright; tilt = acos(-that).
    tilt = torch.acos((-entity.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
    speed = entity.data.root_link_lin_vel_b[:, 0].abs()
    return (tilt < max_tilt) & (speed < max_speed)

  return success


@dataclass(frozen=True)
class TrainConfig:
  # Which architecture to train (see architectures/__init__.py). Defaults to 1,
  # the per-target distribution-matching bridge
  architecture: int = 1

  # Use the analytical experts from dynamics.py. With --no-analytical, train the
  # architecture on top of the frozen RL checkpoints instead (discouraged)
  analytical: bool = True

  drive_task_id: str = DRIVE_TASK_ID
  turn_task_id: str = TURN_TASK_ID

  # Checkpoints for the non-analytical skills. None picks the latest trained one
  drive_checkpoint: str | None = None
  turn_checkpoint: str | None = None

  # Bridge training runs many parallel envs; more is faster and steadier
  num_envs: int = 4096
  device: str | None = None

  # How long to train, per architecture, defaulting to what this experiment declares in
  # __init__.py. Every field is reachable from the command line, e.g.
  # --budgets.arch-1.bridge.num-iterations 1000. The window plan lives beside it there;
  # it is keyed by skill name, so it is edited in __init__.py rather than overridden here
  budgets: Budgets = BUDGETS


def run_train(cfg: TrainConfig) -> None:
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  # Built from the drive task; both skills share one arena.
  env_cfg = load_env_cfg(cfg.drive_task_id, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  exp = build_experiment(
    env,
    device,
    analytical=cfg.analytical,
    drive_task_id=cfg.drive_task_id,
    turn_task_id=cfg.turn_task_id,
    drive_checkpoint=cfg.drive_checkpoint,
    turn_checkpoint=cfg.turn_checkpoint,
  )

  # arch_1 only. turn is the target the demo bridges into, so it gets the real "is the
  # robot upright and slow?" test. drive gets the trivial one: it can pick up a cruise
  # from anywhere, and the test is read long after the hand-over, by which point any test
  # of drive's *start* state would be false by construction and would score every
  # hand-over into it a failure whatever the bridge did. Survival is still required on
  # top, so a bridge into drive that tips the robot is still a failure.
  success_fns: dict[int, SuccessFn] = {
    DRIVE_STRAIGHT: always_ok,
    TURN: _make_turn_success(env),
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
