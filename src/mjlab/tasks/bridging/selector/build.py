"""Pick one representative rollout per entry set and sample entry points from it.

The rollout is the medoid of the complete ones, so the entry points of a set are successive
states of one typical execution and differ only in their step. Each entry point keeps the
spread of all complete rollouts around it, in units of a typical error, which says how
much the executions of the skill differ there.

The build refuses a set, and writes nothing, when an entry point lacks its full future,
has both feet off the ground, or when the object in the scene moves inside the range: the
range has to end before the skill touches it.
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mjlab.tasks.bridging.bridges.dataset.dataset import (
  ROOT_STATE_DIM,
)
from mjlab.tasks.bridging.selector import FUTURE_STATES, Window
from mjlab.tasks.bridging.selector.specs import Spec
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)

SCALES = (0.05, 0.05, 0.15, 0.30, 0.10, 1.50)
"""Root height, tilt, linear velocity, angular velocity, joints, joint rates."""

OBJECT_TOLERANCE = 0.01
"""How far, in metres, the object may drift inside the range before it counts as moved."""


@dataclass(frozen=True)
class Selection:
  """The sampled rows of the medoid rollout of one set."""

  rows: np.ndarray
  trajectory: int
  spread: np.ndarray
  """Mean distance of the other complete rollouts to the medoid at each sampled step."""
  runs: int
  """How many rollouts covered the whole range."""


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
    first = int(phase.min()) if len(phase) else window.start
    raise ValueError(
      f"No rollout contains every phase from {window.start} to {window.stop - 1}, "
      f"the first recorded phase is {first}"
    )
  return np.stack(found)


def medoid(rollouts: torch.Tensor) -> int:
  """Index of the rollout with the smallest distance to all other rollouts."""
  flat = features(canonical(rollouts.flatten(0, 1))).reshape(rollouts.shape[0], -1)
  return int(torch.cdist(flat, flat).sum(dim=1).argmin())


def spread(rollouts: torch.Tensor, chosen: int, offsets: np.ndarray) -> np.ndarray:
  """Mean distance of the other rollouts to the chosen one at each offset."""
  count = rollouts.shape[0]
  if count == 1:
    return np.zeros(len(offsets), dtype=np.float32)
  picked = rollouts[:, torch.as_tensor(offsets, device=rollouts.device)]
  flat = features(canonical(picked.flatten(0, 1))).reshape(count, len(offsets), -1)
  distance = (flat - flat[chosen]).norm(dim=-1)
  return (distance.sum(dim=0) / (count - 1)).cpu().numpy().astype(np.float32)


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
) -> Selection:
  """Sample rows from the medoid rollout through one window."""
  candidates = complete_rollouts(trajectory, phase, window)
  batch = torch.from_numpy(states[candidates]).to(device)
  chosen = medoid(batch)
  offsets = sample_phases(window) - window.start
  return Selection(
    rows=candidates[chosen, offsets],
    trajectory=int(trajectory[candidates[chosen, 0]]),
    spread=spread(batch, chosen, offsets),
    runs=len(candidates),
  )


def object_drift(objects: np.ndarray, rows: np.ndarray) -> float:
  """Largest distance the object moves from where it is at the first row. 0 without one."""
  positions = objects[rows, :3]
  if not np.isfinite(positions).all():
    return 0.0
  return float(np.linalg.norm(positions - positions[0], axis=1).max())


def gait_period(contact: np.ndarray, frame: np.ndarray, start: int) -> float | None:
  """Lag, in steps, at which the left foot contact best repeats itself.

  Read from half the range start on, so the transient after the reset counts little.
  Autocorrelation rather than counting touchdowns, which a chattering contact doubles.
  None when the foot never changes contact.
  """
  order = np.argsort(frame)
  signal = (contact[order, 0] > 0.5)[frame[order] >= start // 2].astype(np.float64)
  signal -= signal.mean()
  if len(signal) < 8 or not signal.any():
    return None
  lags = np.arange(2, len(signal) // 2)
  scores = np.array(
    [np.dot(signal[:-lag], signal[lag:]) / (len(signal) - lag) for lag in lags]
  )
  # argmax keeps the first of equal scores, so a multiple of the period never wins a tie
  best = int(np.argmax(scores))
  return float(lags[best]) if scores[best] > 0 else None


PER_SET = (
  "skill_names",
  "goal_dim",
  "skill_tasks",
  "skill_checkpoints",
  "skill_windows",
)
"""Columns with one row per set rather than one per entry point."""


def pad(value: np.ndarray, shape: list[int]) -> np.ndarray:
  """Zero pad every axis after the first up to shape."""
  extra = [(0, size - have) for size, have in zip(shape, value.shape[1:], strict=True)]
  return np.pad(value, [(0, 0), *extra]) if any(after for _, after in extra) else value


def keep_previous(columns: dict[str, Any], previous: Path) -> dict[str, Any]:
  """Add the sets of a previous table that this build did not make.

  Building some sets then never drops the others. A table written before the selector
  kept its provenance has other columns, and is replaced instead.
  """
  with np.load(previous, allow_pickle=False) as old:
    if set(old.files) != set(columns):
      print(f"[selector] {previous} has other columns, replacing it")
      return columns
    names = [str(name) for name in columns["skill_names"]]
    old_names = [str(name) for name in old["skill_names"]]
    keep = [index for index, name in enumerate(old_names) if name not in names]
    if not keep:
      return columns
    if str(old["robot"]) != str(columns["robot"]) or not np.isclose(
      float(old["fps"]), float(columns["fps"])
    ):
      raise ValueError(f"{previous} holds another robot or control rate")
    rows = np.isin(old["skill"], keep)
    remap = np.zeros(len(old_names), dtype=np.int64)
    remap[keep] = np.arange(len(names), len(names) + len(keep))
    merged: dict[str, Any] = {}
    for name, value in columns.items():
      if name in PER_SET:
        merged[name] = np.concatenate([value, old[name][keep]])
        continue
      if value.ndim == 0:
        merged[name] = value
        continue
      before = old[name][rows]
      if name == "skill":
        before = remap[before].astype(value.dtype)
      shape = [
        max(a, b) for a, b in zip(value.shape[1:], before.shape[1:], strict=True)
      ]
      merged[name] = np.concatenate([pad(value, shape), pad(before, shape)])
  print(f"[selector] kept {', '.join(old_names[i] for i in keep)} from {previous}")
  return merged


def build(
  path: Path,
  out: Path,
  specs: dict[str, Spec],
  robot: str,
  checkpoints: dict[str, Path],
  device: str = "cpu",
  previous: Path | None = None,
) -> Path:
  """Select entry points and keep every row-aligned column of the recording.

  With a previous table, its sets that this build does not make are kept.
  """
  with np.load(path, allow_pickle=False) as raw:
    required = {"states", "skill", "skill_names", "trajectory", "phase", "frame", "fps"}
    missing = required - set(raw.files)
    if missing:
      raise ValueError(f"{path} is missing: {', '.join(sorted(missing))}")
    names = tuple(str(name) for name in raw["skill_names"])
    unknown = set(names) - specs.keys()
    if unknown:
      raise ValueError(f"No spec for recorded sets: {', '.join(sorted(unknown))}")

    states = raw["states"]
    skill_index = raw["skill"]
    trajectory = raw["trajectory"]
    phase = raw["phase"]
    frame = raw["frame"]
    contact = raw["foot_contact"] if "foot_contact" in raw.files else None
    objects = raw["object"] if "object" in raw.files else None
    reference = raw["reference"] if "reference" in raw.files else None
    picks: list[np.ndarray] = []
    spreads: list[np.ndarray] = []
    runs: list[np.ndarray] = []
    problems: list[str] = []
    for index, name in enumerate(names):
      window = specs[name].window
      skill_rows = np.flatnonzero(skill_index == index)
      try:
        selection = select_rows(
          states[skill_rows],
          trajectory[skill_rows],
          phase[skill_rows],
          window,
          device,
        )
      except ValueError as error:
        problems.append(f"{name}: {error}")
        continue
      rows = skill_rows[selection.rows]
      picks.append(rows)
      spreads.append(selection.spread)
      runs.append(np.full(len(rows), selection.runs, dtype=np.int32))
      run_rows = skill_rows[trajectory[skill_rows] == selection.trajectory]
      in_range = run_rows[
        (phase[run_rows] >= window.start) & (phase[run_rows] < window.stop)
      ]
      print(
        f"[selector] {name}: rollout {selection.trajectory} of {selection.runs}, "
        f"{len(rows)} entry points, spread {selection.spread.max():.2f}"
      )
      if contact is not None and (contact[rows] < 0.5).all(axis=1).any():
        problems.append(f"{name}: an entry point has both feet off the ground")
      if objects is not None:
        drift = object_drift(objects, in_range)
        if drift > OBJECT_TOLERANCE:
          problems.append(
            f"{name}: the object moves {drift:.3f} m inside the range, end it earlier"
          )
      tracker = reference is not None and np.isfinite(reference[rows]).all()
      if contact is not None and not tracker:
        period = gait_period(contact[run_rows], frame[run_rows], window.start)
        if period is not None:
          print(
            f"[selector] {name}: gait period {period:.0f} steps, "
            f"range spans {window.stop - window.start}"
          )

    if not picks:
      raise ValueError("Entry points rejected:\n  " + "\n  ".join(problems))
    selected = np.concatenate(picks)
    columns: dict[str, Any] = {}
    for column in raw.files:
      value = raw[column]
      columns[column] = (
        value[selected] if value.ndim > 0 and value.shape[0] == len(states) else value
      )
    columns["source_row"] = selected
    columns["spread"] = np.concatenate(spreads)
    columns["runs"] = np.concatenate(runs)
    columns["robot"] = np.asarray(robot)
    columns["skill_tasks"] = np.asarray([specs[name].task for name in names])
    columns["skill_checkpoints"] = np.asarray(
      [str(checkpoints[name]) for name in names]
    )
    columns["skill_windows"] = np.asarray(
      [(specs[n].start, specs[n].stop, specs[n].samples) for n in names], dtype=np.int32
    )
    frame_width = int(frame.max()) + FUTURE_STATES + 1
    trajectory_width = int(trajectory.max()) + 1
    keys = (
      skill_index.astype(np.int64) * trajectory_width + trajectory
    ) * frame_width + frame
    order = np.argsort(keys)
    offsets = np.arange(1, FUTURE_STATES + 1)
    wanted_keys = keys[selected, None] + offsets[None]
    positions = np.searchsorted(keys[order], wanted_keys)
    candidates = order[positions.clip(max=len(order) - 1)]
    future_mask = (positions < len(order)) & (keys[candidates] == wanted_keys)
    for row in np.flatnonzero(~future_mask.all(axis=1)):
      name = names[int(skill_index[selected[row]])]
      problems.append(
        f"{name}: the entry point at step {int(phase[selected[row]])} lacks its "
        f"{FUTURE_STATES} future states"
      )
    if problems:
      raise ValueError("Entry points rejected:\n  " + "\n  ".join(problems))
    columns["future_states"] = states[candidates]
    columns["future_contact"] = (
      contact[candidates]
      if contact is not None
      else np.zeros((len(selected), FUTURE_STATES, 2), dtype=np.float32)
    )
    columns["future_mask"] = future_mask.astype(np.float32)

  if previous is not None and previous.is_file():
    columns = keep_previous(columns, previous)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out, **columns)
  print(f"[selector] wrote {out} ({len(selected)} entry points)")
  return out
