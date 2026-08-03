"""Train a bridging architecture on the diffdrive experiment.

Builds the skill pool (analytical experts by default), builds the chosen
architecture by id, runs that architecture's own training, and saves the result
so the demo can load it back. Training is common to every architecture: this script
only supplies the experiment-specific pieces (the pool, the entity to harvest states
from, and a per-target success oracle) and lets the architecture do the rest.

arch_0 (the direct hand-off baseline) has nothing to train; running this on it just
saves an empty run so the demo has a checkpoint to point at, uniform with the others.

RL skills are discouraged here: prefer the analytical experts (the default). Train
the per-target bridge architecture (arch_1) with them:

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.train --architecture 1

Use trained RL skills instead of the analytical experts (discouraged):

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.train \\
        --architecture 1 --no-analytical
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.skills.architectures import ARCHITECTURES, TRAINERS
from mjlab.tasks.skills.architectures.arch_1.config import BridgeTraining
from mjlab.tasks.skills.experiments.diffdrive import (
  BRIDGE_VIEW,
  DRIVE_TASK_ID,
  ENTITY_NAME,
  EXPERIMENT_NAME,
  TRAINING,
  TURN_TASK_ID,
  WINDOWS,
  build_pool,
)
from mjlab.tasks.skills.experiments.diffdrive.controller import DRIVE_STRAIGHT, TURN
from mjlab.tasks.skills.utils import new_architecture_run_dir

# A success oracle: given the env after a bridging window, returns a bool per env
# saying whether the target skill actually took over safely. Privileged and external,
# never read off the skill itself (see Skill).
SuccessFn = Callable[[ManagerBasedRlEnv], torch.Tensor]


def _always_success(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Every hand-off counts as a success.

  Used for drive as a target: it can pick up a straight-line cruise from any state,
  so there is no "unsafe" moment to hand into it. (In this experiment drive is never
  actually a bridge target anyway; the controller only ever switches drive -> turn.)
  """
  return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)


def _make_drive_success(
  env: ManagerBasedRlEnv,
  max_tilt: float = 0.4,
  max_speed: float = 0.15,
  max_yaw_rate: float = 0.2,
) -> SuccessFn:
  """Upright, nearly stopped, and no longer turning: drive's own start distribution.

  WARNING: do not wire this in as it stands. An oracle is read `eval_steps` after the
  hand-over, not at it (train_switch, phase 2), and by then drive has been driving for
  a couple of seconds and is at its 2.5 m/s cruise, so `forward_speed < 0.15` is false
  by construction and every hand-over into drive is scored a failure whatever the bridge
  did. That is what it used to do, and it is why the drive-target switch-decider trained
  on an all-negative signal and learned nothing.

  This describes a *pre* hand-over condition, so it would only be usable by a phase 2
  that read its oracle at the moment of the switch. Kept because that is a reasonable
  thing to want, and writing it down beats rediscovering it. `_always_success` is what
  this experiment actually uses for drive.
  """
  entity = env.scene[ENTITY_NAME]

  def success(env: ManagerBasedRlEnv) -> torch.Tensor:
    tilt = torch.acos((-entity.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))

    forward_speed = entity.data.root_link_lin_vel_b[:, 0].abs()
    yaw_rate = entity.data.root_link_ang_vel_b[:, 2].abs()

    return (tilt < max_tilt) & (forward_speed < max_speed) & (yaw_rate < max_yaw_rate)

  return success


def _make_turn_success(
  env: ManagerBasedRlEnv, max_tilt: float = 0.4, max_speed: float = 0.6
) -> SuccessFn:
  """Hand-off into turn succeeds when the robot ends up upright and slow.

  Upright (tilt below max_tilt [rad]) and slow (forward speed below max_speed
  [m/s]) is exactly turn's safe regime: the arc's lateral acceleration only stays
  under the tip-over threshold at low speed. This rewards the switch-decider for
  handing over only once the robot has been braked down, not while it is still fast
  (which would tip). Reads orientation and forward speed off the entity directly.

  This only means anything because the caller pairs it with the termination flags. On
  its own it is blind to the very failure the experiment is built around: the env
  auto-resets a tipped robot, and a freshly spawned robot is upright and stationary,
  so an oracle reading live state after a tip reports a success. `train_switch`
  latches terminations across the window and ANDs them in.
  """
  entity = env.scene[ENTITY_NAME]

  def success(env: ManagerBasedRlEnv) -> torch.Tensor:
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

  # How long to train, defaulting to what this experiment declares in __init__.py.
  # Every field is reachable from the command line, e.g.
  # --training.bridge.num-iterations 1000. The window plan lives beside it there; it is
  # keyed by skill name, so it is edited in __init__.py rather than overridden here
  training: BridgeTraining = TRAINING


def run_train(cfg: TrainConfig) -> None:
  import mjlab.tasks  # noqa: F401  (populates the task registry)

  if cfg.architecture not in ARCHITECTURES:
    raise ValueError(
      f"Unknown architecture {cfg.architecture}; registered: {sorted(ARCHITECTURES)}."
    )

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  # Built from the drive task; both skills share one arena.
  env_cfg = load_env_cfg(cfg.drive_task_id, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  pool = build_pool(
    env,
    device,
    analytical=cfg.analytical,
    drive_task_id=cfg.drive_task_id,
    turn_task_id=cfg.turn_task_id,
    drive_checkpoint=cfg.drive_checkpoint,
    turn_checkpoint=cfg.turn_checkpoint,
  )

  # One success oracle per target skill. turn is the target the demo bridges into, so it
  # gets the real "is the robot upright and slow?" test. drive gets the trivial one: it
  # can pick up a cruise from anywhere, and an oracle is read long after the hand-over,
  # by which point any test of drive's *start* state is meaningless (see
  # _make_drive_success). Survival is still required by train_switch on top of this, so
  # a bridge into drive that tips the robot is still scored a failure.
  success_fns: dict[int, SuccessFn] = {
    DRIVE_STRAIGHT: _always_success,
    TURN: _make_turn_success(env),
  }

  meta = ARCHITECTURES[cfg.architecture](env, pool, BRIDGE_VIEW.resolve(env))
  TRAINERS[cfg.architecture](
    env, pool, ENTITY_NAME, meta, success_fns, WINDOWS, cfg.training
  )

  # Saving is common to all architectures: a fresh run directory, then let the
  # architecture write whatever it needs into it (arch_0 writes nothing).
  run_dir = new_architecture_run_dir(EXPERIMENT_NAME, cfg.architecture)
  meta.save(run_dir)
  print(f"[train] architecture {cfg.architecture} saved to {run_dir}")

  env.close()


if __name__ == "__main__":
  run_train(tyro.cli(TrainConfig))
