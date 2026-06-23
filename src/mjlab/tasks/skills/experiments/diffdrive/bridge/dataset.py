"""A saved, reusable rollout dataset for bridge training experiments.

For every corridor transition (skill1 -> skill2) the dataset stores a couple:

  end window    a family of skill1 rollouts approaching the junction, i.e. the states
                just before the switch fires. The interrupt (where the bridge must start
                from) is the last state of each rollout.
  start window  a family of skill2 rollouts leaving its initiation set, i.e. the early
                tube the bridge must join. Each starts from rest at the first state.

Each side is a family of rollouts, not a single line, so an experiment can pick a
representative, sample a start, or use the whole spread however it likes. Harvesting runs
the real skills in MuJoCo, so we do it once and save it to a single file; training then
loads the file instead of re-rolling every run.

Run as a script to harvest and save a dataset (see main for the flags):

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.bridge.dataset
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mjlab.tasks.skills.experiments.diffdrive.bridge import config
from mjlab.tasks.skills.experiments.diffdrive.bridge.rollouts import (
  interrupt_tracks,
  window_family,
)
from mjlab.tasks.skills.experiments.diffdrive.controller import junction_map
from mjlab.tasks.skills.experiments.diffdrive.experiment import (
  build_model,
  corridor_speeds,
)
from mjlab.tasks.skills.experiments.diffdrive.gridworld import GridWorld
from mjlab.tasks.skills.experiments.diffdrive.robot import DiffDrive
from mjlab.tasks.skills.experiments.diffdrive.skills import corridor_skills


@dataclass
class RolloutDataset:
  """Per corridor transition: skill1's end-window family and skill2's start-window family.

  transitions    list of (src, tgt) corridor id couples, length T.
  end_windows    [T, N, L, 5] skill1's approach to the junction: N rollouts of L states
                 each, ending at the interrupt (the switch point) at the last index.
  start_windows  [T, P, L, 5] skill2's early tube: P rollouts of L states each, starting
                 from rest in the initiation set at the first index.
  speeds         the per-corridor cruise speed used to harvest.
  mode           the skill mode used: "cruise" or "hold".
  window_steps   control ticks per window (L = window_steps + 1).
  cell           metres per grid cell of the world harvested.
  seed           the rng seed used.
  """

  transitions: list[tuple[int, int]]
  end_windows: np.ndarray
  start_windows: np.ndarray
  speeds: dict[int, float]
  mode: str
  window_steps: int
  cell: float
  seed: int


def _tail(track: np.ndarray, length: int) -> np.ndarray:
  """The last `length` states of a track, front-padded with the first if it is shorter.

  Padding at the front keeps the interrupt (the final state) at the last index.
  """
  if len(track) >= length:
    return track[-length:]
  pad = np.repeat(track[:1], length - len(track), axis=0)
  return np.concatenate([pad, track], axis=0)


def _head(track: np.ndarray, length: int) -> np.ndarray:
  """The first `length` states of a track, back-padded with the last if it is shorter."""
  if len(track) >= length:
    return track[:length]
  pad = np.repeat(track[-1:], length - len(track), axis=0)
  return np.concatenate([track, pad], axis=0)


def _stack(members: list[np.ndarray], count: int) -> np.ndarray:
  """Stack a family to [count, L, 5], repeating the last member or truncating to fit."""
  if len(members) < count:
    members = members + [members[-1]] * (count - len(members))
  return np.stack(members[:count])


def harvest_dataset(
  world: GridWorld,
  speeds: dict[int, float],
  mode: str,
  *,
  window_steps: int = config.WINDOW_STEPS,
  n_end: int = config.N_INTERRUPTS,
  n_start: int = config.WINDOW_SAMPLES,
  seed: int = 0,
) -> RolloutDataset:
  """Harvest the end-window and start-window families for every corridor transition."""
  robot = DiffDrive()
  model = build_model(world, robot)
  skills = corridor_skills(world, speeds, mode=mode)
  rng = np.random.default_rng(seed)
  length = window_steps + 1
  transitions: list[tuple[int, int]] = []
  end_windows: list[np.ndarray] = []
  start_windows: list[np.ndarray] = []
  for src, (cell, tgt) in sorted(junction_map(world).items()):
    transitions.append((src, tgt))
    approaches = interrupt_tracks(model, robot, skills[src], cell, n_end, rng)
    if not approaches:
      raise RuntimeError(f"No skill {src} rollouts reached corner {cell}.")
    end_windows.append(_stack([_tail(t, length) for t in approaches], n_end))
    starts = window_family(model, robot, skills[tgt], window_steps, n_start, rng)
    start_windows.append(_stack([_head(t, length) for t in starts], n_start))
  return RolloutDataset(
    transitions=transitions,
    end_windows=np.stack(end_windows),
    start_windows=np.stack(start_windows),
    speeds=dict(speeds),
    mode=mode,
    window_steps=window_steps,
    cell=world.cell,
    seed=seed,
  )


def save_dataset(dataset: RolloutDataset, path: str) -> None:
  """Write a dataset to a single compressed .npz file."""
  ids = sorted(dataset.speeds)
  np.savez_compressed(
    path,
    transitions=np.asarray(dataset.transitions, dtype=int),
    end_windows=dataset.end_windows,
    start_windows=dataset.start_windows,
    speed_ids=np.asarray(ids, dtype=int),
    speed_vals=np.asarray([dataset.speeds[i] for i in ids], dtype=float),
    mode=dataset.mode,
    window_steps=dataset.window_steps,
    cell=dataset.cell,
    seed=dataset.seed,
  )


def load_dataset(path: str) -> RolloutDataset:
  """Read a dataset written by save_dataset."""
  data = np.load(path, allow_pickle=False)
  transitions = [(int(a), int(b)) for a, b in data["transitions"]]
  speeds = {
    int(i): float(v) for i, v in zip(data["speed_ids"], data["speed_vals"], strict=True)
  }
  return RolloutDataset(
    transitions=transitions,
    end_windows=data["end_windows"],
    start_windows=data["start_windows"],
    speeds=speeds,
    mode=str(data["mode"].item()),
    window_steps=int(data["window_steps"].item()),
    cell=float(data["cell"].item()),
    seed=int(data["seed"].item()),
  )


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
  speeds = corridor_speeds(world, slow=args.slow, fast=args.fast)
  dataset = harvest_dataset(world, speeds, args.mode, seed=args.seed)
  save_dataset(dataset, args.out)
  print(f"Saved {len(dataset.transitions)} transitions to {args.out}")
  print(f"  end_windows   {dataset.end_windows.shape}  (T, N, L, 5)")
  print(f"  start_windows {dataset.start_windows.shape}  (T, P, L, 5)")


if __name__ == "__main__":
  main()
