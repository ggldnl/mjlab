"""Pick one representative rollout per skill and sample states from it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  ROOT_STATE_DIM,
)
from mjlab.tasks.bridging.experiments.humanoid.selector import (
  ROLLOUTS_PATH,
  STATES_PATH,
  WINDOWS,
  Window,
)
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)

SCALES = (0.05, 0.05, 0.15, 0.30, 0.10, 1.50)
"""Root height, tilt, linear velocity, angular velocity, joints, joint rates."""


@dataclass
class BuildCfg:
  """Input, output, and skills to select."""

  path: Path = ROLLOUTS_PATH
  out: Path = STATES_PATH
  skills: tuple[str, ...] = ()
  device: str = "cpu"


def canonical(states: torch.Tensor) -> torch.Tensor:
  """Remove ground position and heading from robot states."""
  heading = yaw_quat(states[:, 3:7])
  out = states.clone()
  out[:, :2] = 0.0
  out[:, 3:7] = quat_mul(quat_conjugate(heading), states[:, 3:7])
  out[:, 7:10] = quat_apply_inverse(heading, states[:, 7:10])
  out[:, 10:ROOT_STATE_DIM] = quat_apply_inverse(heading, states[:, 10:ROOT_STATE_DIM])
  return out


def features(states: torch.Tensor) -> torch.Tensor:
  """Scale canonical states so each physical channel has comparable weight."""
  joints = (states.shape[1] - ROOT_STATE_DIM) // 2
  tilt = states[:, 3:7]
  tilt = tilt * torch.where(tilt[:, :1] < 0, -1, 1)
  blocks = (
    states[:, 2:3],
    2 * tilt[:, 1:4],
    states[:, 7:10],
    states[:, 10:ROOT_STATE_DIM],
    states[:, ROOT_STATE_DIM : ROOT_STATE_DIM + joints],
    states[:, ROOT_STATE_DIM + joints :],
  )
  return torch.cat(
    [
      block / (scale * block.shape[1] ** 0.5)
      for block, scale in zip(blocks, SCALES, strict=True)
    ],
    dim=1,
  )


def complete_rollouts(
  trajectory: np.ndarray, phase: np.ndarray, window: Window
) -> np.ndarray:
  """Row indices for rollouts containing every phase in the window."""
  expected = np.arange(window.start, window.stop)
  inside = np.flatnonzero((phase >= window.start) & (phase < window.stop))
  found: list[np.ndarray] = []
  for rollout in np.unique(trajectory[inside]):
    rows = inside[trajectory[inside] == rollout]
    rows = rows[np.argsort(phase[rows])]
    if np.array_equal(phase[rows], expected):
      found.append(rows)
  if not found:
    raise ValueError(
      f"No rollout contains every phase from {window.start} to {window.stop - 1}"
    )
  return np.stack(found)


def medoid(rollouts: torch.Tensor) -> int:
  """Index of the rollout with the smallest distance to all other rollouts."""
  flat = features(canonical(rollouts.flatten(0, 1))).reshape(rollouts.shape[0], -1)
  return int(torch.cdist(flat, flat).sum(dim=1).argmin())


def sample_phases(window: Window) -> np.ndarray:
  """Center phase of each equal slice of a window."""
  edges = np.linspace(window.start, window.stop, window.samples + 1)
  return np.floor((edges[:-1] + edges[1:]) / 2).astype(np.int64)


def select_rows(
  states: np.ndarray,
  trajectory: np.ndarray,
  phase: np.ndarray,
  window: Window,
  device: str = "cpu",
) -> tuple[np.ndarray, int]:
  """Sample rows from the medoid rollout through one window."""
  candidates = complete_rollouts(trajectory, phase, window)
  batch = torch.from_numpy(states[candidates]).to(device)
  chosen = medoid(batch)
  offsets = sample_phases(window) - window.start
  return candidates[chosen, offsets], int(trajectory[candidates[chosen, 0]])


def build(cfg: BuildCfg) -> Path:
  """Select states and preserve every row-aligned column from the recording."""
  with np.load(cfg.path, allow_pickle=False) as raw:
    required = {"states", "skill", "skill_names", "trajectory", "phase", "fps"}
    missing = required - set(raw.files)
    if missing:
      raise ValueError(f"{cfg.path} is missing: {', '.join(sorted(missing))}")

    names = tuple(str(name) for name in raw["skill_names"])
    wanted = cfg.skills or tuple(name for name in names if name in WINDOWS)
    if not wanted:
      raise ValueError("No recorded skill has a configured window")
    unknown = set(wanted) - WINDOWS.keys()
    absent = set(wanted) - set(names)
    if unknown:
      raise ValueError(f"No window for: {', '.join(sorted(unknown))}")
    if absent:
      raise ValueError(f"Not recorded: {', '.join(sorted(absent))}")

    states = raw["states"]
    skill_index = raw["skill"]
    trajectory = raw["trajectory"]
    phase = raw["phase"]
    picks: list[np.ndarray] = []
    for skill in wanted:
      skill_rows = np.flatnonzero(skill_index == names.index(skill))
      rows, rollout = select_rows(
        states[skill_rows],
        trajectory[skill_rows],
        phase[skill_rows],
        WINDOWS[skill],
        cfg.device,
      )
      picks.append(skill_rows[rows])
      print(f"[selector] {skill}: rollout {rollout}, {len(rows)} states")

    selected = np.concatenate(picks)
    columns = {}
    for name in raw.files:
      value = raw[name]
      columns[name] = (
        value[selected] if value.ndim > 0 and value.shape[0] == len(states) else value
      )
    columns["source_row"] = selected

  cfg.out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(cfg.out, **columns)
  print(f"[selector] wrote {cfg.out} ({len(selected)} states)")
  return cfg.out


if __name__ == "__main__":
  build(tyro.cli(BuildCfg, config=mjlab.TYRO_FLAGS))
