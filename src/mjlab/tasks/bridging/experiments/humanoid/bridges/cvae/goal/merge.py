"""Combine oracle rollouts and accepted physical branches for DAgger.

Run:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.merge
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.bridges.cvae.goal.collect import (
  DEFAULT_ORACLE_DATASET,
  ORACLE_BRANCH_DATASET,
  ORACLE_ROLLOUT_DATASET,
)

ROW_KEYS = (
  "states",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
  "foot_contact",
  "previous_action",
  "env_id",
  "trajectory",
  "frame",
  "phase",
  "motion_id",
  "skill",
)
META_KEYS = ("skill_names", "body_names", "motion_files", "fps")


@dataclass
class MergeCfg:
  inputs: tuple[Path, ...] = (ORACLE_ROLLOUT_DATASET, ORACLE_BRANCH_DATASET)
  path: Path = DEFAULT_ORACLE_DATASET


def merge(cfg: MergeCfg) -> Path:
  if len(cfg.inputs) < 1:
    raise ValueError("At least one oracle dataset is required")
  rows: dict[str, list[np.ndarray]] = {key: [] for key in ROW_KEYS}
  metadata: dict[str, np.ndarray] = {}
  next_trajectory = 0
  next_env = 0
  for path in cfg.inputs:
    with np.load(path, allow_pickle=False) as raw:
      missing = set(ROW_KEYS + META_KEYS) - set(raw.files)
      if missing:
        raise ValueError(f"{path} is missing {', '.join(sorted(missing))}")
      if not metadata:
        metadata = {key: raw[key].copy() for key in META_KEYS}
      elif any(not np.array_equal(raw[key], metadata[key]) for key in META_KEYS):
        raise ValueError(f"{path} has incompatible oracle metadata")
      for key in ROW_KEYS:
        value = raw[key]
        if key == "trajectory":
          value = value.astype(np.int64) + next_trajectory
        elif key == "env_id":
          value = value.astype(np.int64) + next_env
        rows[key].append(value)
      next_trajectory += int(raw["trajectory"].max()) + 1
      next_env += int(raw["env_id"].max()) + 1
  output = {key: np.concatenate(parts) for key, parts in rows.items()}
  output.update(metadata)
  output["trajectory_ids_global"] = np.asarray(True)
  cfg.path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(cfg.path, **output)  # ty: ignore[invalid-argument-type]
  print(f"[goal] wrote {cfg.path} ({len(output['states'])} rows)")
  return cfg.path


if __name__ == "__main__":
  merge(tyro.cli(MergeCfg, config=mjlab.TYRO_FLAGS))
