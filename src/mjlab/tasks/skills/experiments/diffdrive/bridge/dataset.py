"""A saved, reusable rollout dataset for bridge training experiments.

For every skill transition (skill1 -> skill2) the dataset stores two families of rollouts:

  end_windows    many rollouts of skill1's end window: the states leading up to the switch,
                 with the switch point at the last index of each rollout.
  start_windows  many rollouts of skill2's start window: the states it passes through just
                 after it starts, from the start at the first index.

Skills are treated as black boxes: each window is just a sequence of reduced states, with
no assumption about how the skill produced them. That is all an experiment needs to learn a
bridge from one to the other; it can reduce a family, sample from it, or use the whole
spread however it likes. Harvesting runs the real skills in MuJoCo, so we do it once and
save it to a single file.

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.bridge.dataset
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mjlab.tasks.skills.experiments.diffdrive import experiment
from mjlab.tasks.skills.experiments.diffdrive.bridge import rollouts
from mjlab.tasks.skills.experiments.diffdrive.gridworld import GridWorld
from mjlab.tasks.skills.experiments.diffdrive.robot import DiffDrive
from mjlab.tasks.skills.experiments.diffdrive.skills import corridor_skills


@dataclass
class RolloutDataset:
  """Per transition, the two window families. Skills are black boxes: just state sequences.

  transitions    list of (src, tgt) ids, length T (which two skills each couple joins).
  end_windows    [T, N, L, 5] N rollouts of skill1's end window, L states each.
  start_windows  [T, P, L, 5] P rollouts of skill2's start window, L states each.
  """

  transitions: list[tuple[int, int]]
  end_windows: np.ndarray
  start_windows: np.ndarray


def harvest_dataset(
  world: GridWorld,
  speeds: dict[int, float],
  mode: str,
  *,
  window_steps: int = experiment.CONFIG.window_steps,
  n_end: int = experiment.CONFIG.n_interrupts,
  n_start: int = experiment.CONFIG.window_samples,
  seed: int = 0,
) -> RolloutDataset:
  """Harvest the end-window and start-window families for every skill transition."""
  robot = DiffDrive()
  model = experiment.build_model(world, robot)
  skills = corridor_skills(world, speeds, mode=mode)
  rng = np.random.default_rng(seed)
  transitions: list[tuple[int, int]] = []
  end_windows: list[np.ndarray] = []
  start_windows: list[np.ndarray] = []
  for src, (cell, tgt) in sorted(world.junction_map().items()):
    transitions.append((src, tgt))
    end_windows.append(
      rollouts.end_window_family(
        model, robot, skills[src], cell, n_end, window_steps, rng
      )
    )
    start_windows.append(
      rollouts.start_window_family(
        model, robot, skills[tgt], n_start, window_steps, rng
      )
    )
  return RolloutDataset(transitions, np.stack(end_windows), np.stack(start_windows))


def save_dataset(dataset: RolloutDataset, path: str) -> None:
  """Write a dataset to a single compressed .npz file."""
  np.savez_compressed(
    path,
    transitions=np.asarray(dataset.transitions, dtype=int),
    end_windows=dataset.end_windows,
    start_windows=dataset.start_windows,
  )


def load_dataset(path: str) -> RolloutDataset:
  """Read a dataset written by save_dataset."""
  data = np.load(path, allow_pickle=False)
  transitions = [(int(a), int(b)) for a, b in data["transitions"]]
  return RolloutDataset(transitions, data["end_windows"], data["start_windows"])


def main() -> None:
  """Harvest a rollout dataset and save it to disk."""
  from dataclasses import dataclass as _dataclass

  import tyro

  @_dataclass
  class Args:
    out: str = "rollouts.npz"  # output .npz path
    cell: float = 1.0  # metres per grid cell
    mode: str = "hold"  # skill mode: "cruise" or "hold"
    slow: float = 0.5  # slow-corridor cruise speed (m/s)
    fast: float = 1.5  # fast-corridor cruise speed (m/s)
    seed: int = 0

  args = tyro.cli(Args)
  world = GridWorld(cell=args.cell)
  speeds = experiment.corridor_speeds(world, slow=args.slow, fast=args.fast)
  dataset = harvest_dataset(world, speeds, args.mode, seed=args.seed)
  save_dataset(dataset, args.out)
  print(f"Saved {len(dataset.transitions)} transitions to {args.out}")
  print(f"  end_windows   {dataset.end_windows.shape}  (T, N, L, 5)")
  print(f"  start_windows {dataset.start_windows.shape}  (T, P, L, 5)")


if __name__ == "__main__":
  main()
