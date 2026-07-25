"""Shared helpers for the skills experiments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mjlab.tasks.registry import load_rl_cfg


def retrieve_latest_checkpoint(task_id: str) -> str:
  """Path to the most recently written checkpoint for ``task_id``'s experiment.

  Mirrors the training log layout play.py uses:
  ``logs/rsl_rl/<experiment_name>/<run>/<model_*.pt>``. Raises if the task has
  never been trained.
  """
  experiment = load_rl_cfg(task_id).experiment_name
  root = Path("logs") / "rsl_rl" / experiment
  checkpoints = sorted(root.glob("*/*.pt"), key=lambda p: p.stat().st_mtime)
  if not checkpoints:
    raise FileNotFoundError(
      f"No checkpoint found for task '{task_id}' under {root}. "
      f"Train it first with `uv run train {task_id}`."
    )
  return str(checkpoints[-1])


# Where a trained architecture saves its state. Kept apart from the rsl_rl skill
# checkpoints above (logs/rsl_rl/...): those are per-skill policies, these are a
# whole bridging architecture for one experiment. Each saved run is a directory,
# since different architectures write different (and possibly several) files.
_ARCHITECTURE_ROOT = Path("logs") / "skills"


def architecture_checkpoint_root(experiment: str, architecture: int) -> Path:
  """Directory collecting every saved run of one architecture on one experiment."""
  return _ARCHITECTURE_ROOT / experiment / f"arch_{architecture}"


def new_architecture_run_dir(experiment: str, architecture: int) -> Path:
  """A fresh, timestamped run directory for training to save into (created here)."""
  run = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  path = architecture_checkpoint_root(experiment, architecture) / run
  path.mkdir(parents=True, exist_ok=True)
  return path


def retrieve_latest_architecture_checkpoint(experiment: str, architecture: int) -> Path:
  """The most recently saved run directory for ``architecture`` on ``experiment``.

  Raises if that architecture has never been trained for this experiment.
  """
  root = architecture_checkpoint_root(experiment, architecture)
  runs = sorted(
    (p for p in root.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime
  )
  if not runs:
    raise FileNotFoundError(
      f"No saved run for architecture {architecture} under {root}. Train it first "
      f"with `uv run python -m mjlab.tasks.skills.experiments.{experiment}.train "
      f"--architecture {architecture}`."
    )
  return runs[-1]
