"""Extra body trajectories paired with the shared bridge dataset rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class GoalBodies:
  pos: torch.Tensor
  quat: torch.Tensor
  lin_vel: torch.Tensor
  ang_vel: torch.Tensor
  contact: torch.Tensor
  names: tuple[str, ...]


def load_bodies(path: Path, device: str, split: str, holdout: int = 8) -> GoalBodies:
  """Load body arrays in the exact row order of load_dataset."""
  if split not in ("train", "eval"):
    raise ValueError("split must be train or eval")
  with np.load(path, allow_pickle=False) as raw:
    required = {
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
      "foot_contact",
      "body_names",
      "env_id",
    }
    missing = required - set(raw.files)
    if missing:
      raise ValueError(f"{path} is missing {', '.join(sorted(missing))}")
    held = (raw["env_id"] % holdout) == 0
    mask = held if split == "eval" else ~held
    return GoalBodies(
      pos=torch.from_numpy(raw["body_pos_w"][mask]).to(device),
      quat=torch.from_numpy(raw["body_quat_w"][mask]).to(device),
      lin_vel=torch.from_numpy(raw["body_lin_vel_w"][mask]).to(device),
      ang_vel=torch.from_numpy(raw["body_ang_vel_w"][mask]).to(device),
      contact=torch.from_numpy(raw["foot_contact"][mask]).to(device),
      names=tuple(str(name) for name in raw["body_names"]),
    )
