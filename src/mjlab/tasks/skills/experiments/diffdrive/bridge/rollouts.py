"""Harvest the data the bridge is built from, by rolling out the real skills.

We never invent a representative junction. Instead we take the skills and the world
as they are and run them, in the real simulator, to collect two things:

* windows  the early stretch of each skill's tube, i.e. the states it actually
           passes through just after it starts. These are the candidate states the
           bridge may join.
* interrupts  the states the robot is in when a switch fires, i.e. where the
           previous skill leaves it at the junction corner. These are where the
           bridge must start from, and they are spread out by jittering the
           previous skill's start within its initiation set.

The same primitives serve training (which needs both, per corridor transition) and
deployment (which needs only the windows, to pick a goal at the moment of a switch).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.tasks.skills.experiments.diffdrive.bridge import config
from mjlab.tasks.skills.experiments.diffdrive.controller import junction_map
from mjlab.tasks.skills.experiments.diffdrive.experiment import build_model
from mjlab.tasks.skills.experiments.diffdrive.gridworld import GridWorld
from mjlab.tasks.skills.experiments.diffdrive.robot import (
  DiffDrive,
  X,
  Y,
)
from mjlab.tasks.skills.experiments.diffdrive.skills import (
  CorridorSkill,
  corridor_skills,
)


@dataclass
class Harvest:
  """Per corridor transition: the interrupt states and the next tube's window.

  Arrays are stacked to a fixed length so the env can index them as plain tensors:
  interrupts is [T, N, 5], windows is [T, M, 5], aligned with transitions (a list
  of (source, target) corridor ids).
  """

  transitions: list[tuple[int, int]]
  interrupts: np.ndarray
  windows: np.ndarray


def _entry_state(skill: CorridorSkill, speed: float) -> np.ndarray:
  """Reduced state at the skill's geometric start, aligned, at the given speed."""
  ex, ey = skill.world.cell_center(*skill.entry)
  return np.array([float(ex), float(ey), skill.heading, speed, 0.0])


def _rollout(
  model: mujoco.MjModel,
  robot: DiffDrive,
  skill: CorridorSkill,
  start: np.ndarray,
  *,
  max_steps: int,
  stop_cell: tuple[int, int] | None = None,
) -> np.ndarray:
  """Run one skill from `start` and return its reduced-state track.

  Stops early when the robot reaches `stop_cell` (the junction corner) or leaves a
  corridor. The skill is re-armed first, so it re-checks its initiation set against
  `start` and only drives if `start` is a valid place to begin.
  """
  skill.reset()
  data = mujoco.MjData(model)
  robot.reset(model, data, start)
  mujoco.mj_forward(model, data)
  world = skill.world
  track = [robot.sense(model, data).copy()]
  for _ in range(max_steps):
    state = robot.sense(model, data)
    data.ctrl[:] = robot.actuate(model, data, skill(state))
    for _ in range(config.DECIMATION):
      mujoco.mj_step(model, data)
    state = robot.sense(model, data)
    track.append(state.copy())
    x, y = float(state[X]), float(state[Y])
    if not world.is_free(x, y):
      break
    if stop_cell is not None:
      r, c = world.world_to_cell(x, y)
      if (int(r), int(c)) == stop_cell:
        break
  return np.asarray(track)


def harvest_window(
  model: mujoco.MjModel,
  robot: DiffDrive,
  skill: CorridorSkill,
  window_steps: int,
) -> np.ndarray:
  """The early window of a skill's tube: its first `window_steps` states from rest.

  The skill begins at its entry cell at rest and ramps up, exactly as it does in the
  experiment, so the window holds the real low-to-cruise states the bridge can aim at.
  """
  start = _entry_state(skill, speed=0.0)
  track = _rollout(model, robot, skill, start, max_steps=window_steps)
  return track[: window_steps + 1]


def harvest_interrupts(
  model: mujoco.MjModel,
  robot: DiffDrive,
  skill: CorridorSkill,
  junction_cell: tuple[int, int],
  count: int,
  rng: np.random.Generator,
) -> np.ndarray:
  """Interrupt states at the junction corner, spread over the skill's initiation set.

  Each sample jitters the start (lateral offset, heading, speed) within the skill's
  initiation set, runs the skill to the corner, and records the state there. Samples
  that crash or never reach the corner are dropped.
  """
  half = skill.d_tol * 0.7
  phi = skill.phi_tol * 0.7
  v_hi = min(1.2, skill.speed)
  out: list[np.ndarray] = []
  for _ in range(count * 8):
    if len(out) >= count:
      break
    ex, ey = skill.world.cell_center(*skill.entry)
    offset = rng.uniform(-half, half)
    lateral = (-math.sin(skill.heading), math.cos(skill.heading))
    start = np.array(
      [
        float(ex) + offset * lateral[0],
        float(ey) + offset * lateral[1],
        skill.heading + rng.uniform(-phi, phi),
        rng.uniform(0.0, v_hi),
        0.0,
      ]
    )
    track = _rollout(model, robot, skill, start, max_steps=600, stop_cell=junction_cell)
    last = track[-1]
    r, c = skill.world.world_to_cell(float(last[X]), float(last[Y]))
    if (int(r), int(c)) == junction_cell and skill.world.is_free(
      float(last[X]), float(last[Y])
    ):
      out.append(last)
  if not out:
    raise RuntimeError(f"No interrupt states reached corner {junction_cell}.")
  return np.asarray(out)


def _fix_length(arr: np.ndarray, length: int) -> np.ndarray:
  """Pad (by repeating the last row) or truncate `arr` to exactly `length` rows."""
  if len(arr) >= length:
    return arr[:length]
  pad = np.repeat(arr[-1:], length - len(arr), axis=0)
  return np.concatenate([arr, pad], axis=0)


def harvest_transitions(
  world: GridWorld,
  speeds: dict[int, float],
  mode: str,
  *,
  window_steps: int = config.WINDOW_STEPS,
  n_interrupts: int = config.N_INTERRUPTS,
  seed: int = 0,
) -> Harvest:
  """Roll out every corridor transition and stack its interrupts and target window."""
  robot = DiffDrive()
  model = build_model(world, robot)
  skills = corridor_skills(world, speeds, mode=mode)
  rng = np.random.default_rng(seed)
  m = window_steps + 1
  transitions: list[tuple[int, int]] = []
  interrupts: list[np.ndarray] = []
  windows: list[np.ndarray] = []
  for src, (cell, tgt) in sorted(junction_map(world).items()):
    transitions.append((src, tgt))
    window = harvest_window(model, robot, skills[tgt], window_steps)
    windows.append(_fix_length(window, m))
    inter = harvest_interrupts(model, robot, skills[src], cell, n_interrupts, rng)
    interrupts.append(_fix_length(inter, n_interrupts))
  return Harvest(transitions, np.stack(interrupts), np.stack(windows))


def harvest_windows(
  world: GridWorld,
  speeds: dict[int, float],
  mode: str,
  *,
  window_steps: int = config.WINDOW_STEPS,
) -> dict[int, np.ndarray]:
  """The early window of every corridor's skill, keyed by corridor id (for deployment)."""
  robot = DiffDrive()
  model = build_model(world, robot)
  skills = corridor_skills(world, speeds, mode=mode)
  m = window_steps + 1
  return {
    cid: _fix_length(harvest_window(model, robot, skill, window_steps), m)
    for cid, skill in skills.items()
  }
