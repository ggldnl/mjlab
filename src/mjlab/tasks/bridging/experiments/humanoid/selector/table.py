"""Load selected states from the plain NPZ written by build.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mjlab.tasks.bridging.experiments.humanoid.selector import TABLE_PATH


@dataclass(frozen=True)
class Entry:
  """One selected state and the context needed to resume its skill."""

  skill: str
  state: np.ndarray
  frame: int
  seconds: float
  previous_action: np.ndarray
  reference: np.ndarray
  motion_file: str
  motion_scale: float
  source_row: int
  foot_contact: np.ndarray | None = None
  foot_pos_b: np.ndarray | None = None
  foot_quat_b: np.ndarray | None = None
  foot_lin_vel_b: np.ndarray | None = None
  foot_ang_vel_b: np.ndarray | None = None
  future_states: np.ndarray | None = None
  future_contact: np.ndarray | None = None
  future_mask: np.ndarray | None = None

  @property
  def name(self) -> str:
    return f"f{self.frame:03d}"


@dataclass(frozen=True)
class EntryTable:
  """Selected states in skill and phase order."""

  entries: tuple[Entry, ...]
  fps: float

  @property
  def skills(self) -> tuple[str, ...]:
    return tuple(dict.fromkeys(entry.skill for entry in self.entries))

  def of(self, skill: str) -> tuple[Entry, ...]:
    found = tuple(entry for entry in self.entries if entry.skill == skill)
    if not found:
      raise ValueError(f"No states for {skill}. Available: {', '.join(self.skills)}")
    return found

  @staticmethod
  def load(path: Path = TABLE_PATH) -> EntryTable:
    if not path.exists():
      raise FileNotFoundError(f"No selector states at {path}. Run selector.build first")
    with np.load(path, allow_pickle=False) as raw:
      required = {
        "fps",
        "motion_file",
        "motion_scale",
        "phase",
        "previous_action",
        "reference",
        "skill",
        "skill_names",
        "source_row",
        "states",
      }
      missing = required - set(raw.files)
      if missing:
        raise ValueError(f"{path} is missing: {', '.join(sorted(missing))}")
      names = tuple(str(name) for name in raw["skill_names"])
      fps = float(raw["fps"])
      entries = tuple(
        Entry(
          skill=names[int(raw["skill"][index])],
          state=raw["states"][index].copy(),
          frame=int(raw["phase"][index]),
          seconds=int(raw["phase"][index]) / fps,
          previous_action=raw["previous_action"][index].copy(),
          reference=raw["reference"][index].copy(),
          motion_file=str(raw["motion_file"][index]),
          motion_scale=float(raw["motion_scale"][index]),
          source_row=int(raw["source_row"][index]),
          foot_contact=(
            raw["foot_contact"][index].copy() if "foot_contact" in raw.files else None
          ),
          foot_pos_b=(
            raw["foot_pos_b"][index].copy() if "foot_pos_b" in raw.files else None
          ),
          foot_quat_b=(
            raw["foot_quat_b"][index].copy() if "foot_quat_b" in raw.files else None
          ),
          foot_lin_vel_b=(
            raw["foot_lin_vel_b"][index].copy()
            if "foot_lin_vel_b" in raw.files
            else None
          ),
          foot_ang_vel_b=(
            raw["foot_ang_vel_b"][index].copy()
            if "foot_ang_vel_b" in raw.files
            else None
          ),
          future_states=(
            raw["future_states"][index].copy() if "future_states" in raw.files else None
          ),
          future_contact=(
            raw["future_contact"][index].copy()
            if "future_contact" in raw.files
            else None
          ),
          future_mask=(
            raw["future_mask"][index].copy() if "future_mask" in raw.files else None
          ),
        )
        for index in range(len(raw["states"]))
      )
    return EntryTable(entries, fps)
