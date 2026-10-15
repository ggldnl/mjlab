"""Record and build the entry points of one robot, see the package docstring."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from mjlab.tasks.bridging.bridges.dataset import dataset
from mjlab.tasks.bridging.selector import build, paths, record
from mjlab.tasks.bridging.selector.specs import Spec, specs_for
from mjlab.tasks.registry import load_rl_cfg


@dataclass
class SelectorCfg:
  """Which robot, which sets, and where the entry points go."""

  robot: str = "g1"
  sets: tuple[str, ...] = ()
  """Entry sets to build. Empty builds every set of the robot that has a checkpoint."""
  checkpoints: dict[str, Path] = field(default_factory=dict)
  """Pinned checkpoint per set. The others use the newest one of their task."""
  window: tuple[str, int, int, int] | None = None
  """One range override: set, start, stop, samples."""
  task: tuple[str, str] | None = None
  """One task override: set, task id. Together with --window it adds a set."""
  runs: int = 64
  """Parallel rollouts per set, the candidates for the medoid."""
  seed: int = 0
  device: str | None = None
  out: Path | None = None
  """Where the entry points go. The rollouts are written next to them."""


def plan(cfg: SelectorCfg) -> dict[str, Spec]:
  """The robot's specs with the overrides applied, restricted to the asked sets."""
  specs = specs_for(cfg.robot)
  if cfg.task is not None:
    name, task = cfg.task
    if name in specs:
      specs[name] = replace(specs[name], task=task)
    elif cfg.window is None or cfg.window[0] != name:
      raise ValueError(f"A new set '{name}' also needs --window {name} START STOP N")
    else:
      specs[name] = Spec(task, *cfg.window[1:])
  if cfg.window is not None:
    name, start, stop, samples = cfg.window
    if name not in specs:
      raise ValueError(f"No set '{name}' for {cfg.robot}, add its task with --task")
    specs[name] = replace(specs[name], start=start, stop=stop, samples=samples)
  unknown = set(cfg.sets) - specs.keys()
  if unknown:
    raise ValueError(
      f"Unknown sets for {cfg.robot}: {', '.join(sorted(unknown))}. "
      f"Available: {', '.join(specs)}"
    )
  if cfg.sets:
    specs = {name: specs[name] for name in cfg.sets}
  return specs


def resolve_checkpoints(
  specs: dict[str, Spec], pinned: dict[str, Path], required: set[str]
) -> tuple[dict[str, Path], dict[str, str]]:
  """Checkpoint per set, and why the others were skipped.

  A pinned path that does not exist, or a set asked for by name without a checkpoint, is an
  error rather than a skip.
  """
  unknown = set(pinned) - specs.keys()
  if unknown:
    raise ValueError(
      f"Pinned checkpoints for unknown sets: {', '.join(sorted(unknown))}"
    )
  found: dict[str, Path] = {}
  skipped: dict[str, str] = {}
  for name, spec in specs.items():
    if name in pinned:
      if not pinned[name].is_file():
        raise FileNotFoundError(f"No checkpoint at {pinned[name]} for {name}")
      found[name] = pinned[name]
      continue
    experiment = load_rl_cfg(spec.task).experiment_name
    try:
      found[name] = dataset.find_checkpoint((experiment,))
    except SystemExit as error:
      if name in required:
        raise
      skipped[name] = str(error)
  return found, skipped


def main(cfg: SelectorCfg) -> Path:
  """Record every set that has a checkpoint, then build its entry points."""
  from mjlab import tasks as _tasks

  del _tasks
  specs = plan(cfg)
  checkpoints, skipped = resolve_checkpoints(specs, cfg.checkpoints, set(cfg.sets))
  specs = {name: spec for name, spec in specs.items() if name in checkpoints}
  if not specs:
    raise SystemExit(f"No entry set of {cfg.robot} has a checkpoint")
  for name, path in checkpoints.items():
    print(f"[selector] {name}: {specs[name].task} from {path}")
  rollouts, states = paths(cfg.robot)
  if cfg.out is not None:
    rollouts, states = cfg.out.with_name(f"{cfg.out.stem}_rollouts.npz"), cfg.out
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  record.collect(specs, cfg.robot, checkpoints, rollouts, cfg.runs, cfg.seed, device)
  out = build.build(rollouts, states, specs, cfg.robot, checkpoints, previous=states)
  for name, reason in skipped.items():
    print(f"[selector] skipped {name}: {reason}")
  return out
