"""A library of jump clips, indexed per environment.

mjlab's tracking task carries one motion, so a single frame index is enough to
address it. Goal conditioning needs a set of clips and, per environment, a pointer
into that set: which jump am I doing, how far into it am I, and how much has the
jump been stretched.

Clips have different lengths, so everything is padded to the longest one and the
true lengths are kept alongside. Reading past the end of a clip is prevented by
clamping the frame index, not by masking, because the environment that got there
is about to be reset anyway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class MotionMetadata:
  """What a clip is, as opposed to what it contains."""

  name: str
  num_frames: int
  fps: float
  goal_xy: tuple[float, float]
  goal_yaw: float
  goal_apex: float
  takeoff_step: int
  land_step: int

  @property
  def distance(self) -> float:
    return float(np.linalg.norm(np.asarray(self.goal_xy)))


class MotionLibrary:
  """Padded tensors for a set of clips, plus their goals.

  Every tensor has a leading motion axis and a padded frame axis, so a pair of
  index tensors `(motion_ids, time_steps)` of shape [num_envs] gathers a batch.
  """

  def __init__(
    self,
    motion_files: tuple[str, ...] | list[str],
    body_indexes: torch.Tensor,
    device: str = "cpu",
  ) -> None:
    if not motion_files:
      raise ValueError("No motion files given.")

    paths = [Path(f) for f in motion_files]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
      raise FileNotFoundError(
        "Missing motion files: "
        + ", ".join(missing)
        + "\nRun: uv run --with joblib python -m mjlab.tasks.skills.experiments"
        ".parkour.jump.dataset"
      )

    self.device = device
    self._body_indexes = body_indexes

    raw = [np.load(p) for p in paths]
    lengths = [int(d["joint_pos"].shape[0]) for d in raw]
    self.time_step_total_per_motion = torch.tensor(
      lengths, dtype=torch.long, device=device
    )
    self.time_step_total = int(max(lengths))
    self.num_motions = len(raw)

    def stack(key: str) -> torch.Tensor:
      return self._stack_padded([d[key] for d in raw], device)

    self.joint_pos = stack("joint_pos")
    self.joint_vel = stack("joint_vel")

    all_body_pos_w = stack("body_pos_w")
    all_body_lin_vel_w = stack("body_lin_vel_w")

    self.body_pos_w = all_body_pos_w[:, :, body_indexes]
    self.body_quat_w = stack("body_quat_w")[:, :, body_indexes]
    self.body_lin_vel_w = all_body_lin_vel_w[:, :, body_indexes]
    self.body_ang_vel_w = stack("body_ang_vel_w")[:, :, body_indexes]

    # The root, kept separately and indexed in the full body list rather than the
    # tracked subset: stretching a jump is defined relative to where the root
    # started, and a config is free not to track the root body at all.
    self.root_pos_w = all_body_pos_w[:, :, 0]
    self.root_lin_vel_w = all_body_lin_vel_w[:, :, 0]

    self.metadata = [
      MotionMetadata(
        name=path.stem,
        num_frames=length,
        fps=float(np.asarray(data["fps"]).reshape(-1)[0]),
        goal_xy=(float(data["goal_xy"][0]), float(data["goal_xy"][1])),
        goal_yaw=float(data["goal_yaw"]),
        goal_apex=float(data["goal_apex"]),
        takeoff_step=int(data["takeoff_step"]),
        land_step=int(data["land_step"]),
      )
      for path, data, length in zip(paths, raw, lengths, strict=True)
    ]

    self.goals = torch.tensor(
      [[m.goal_xy[0], m.goal_xy[1], m.goal_yaw, m.goal_apex] for m in self.metadata],
      dtype=torch.float32,
      device=device,
    )
    self.distances = torch.tensor(
      [m.distance for m in self.metadata], dtype=torch.float32, device=device
    )
    self.land_steps = torch.tensor(
      [m.land_step if m.land_step >= 0 else m.num_frames - 1 for m in self.metadata],
      dtype=torch.long,
      device=device,
    )

  @staticmethod
  def _stack_padded(arrays: list[np.ndarray], device: str) -> torch.Tensor:
    """Pad each clip with its own final frame, then stack."""
    max_len = max(a.shape[0] for a in arrays)
    padded = []
    for a in arrays:
      if a.shape[0] < max_len:
        tail = np.repeat(a[-1:], max_len - a.shape[0], axis=0)
        a = np.concatenate([a, tail], axis=0)
      padded.append(a)
    return torch.tensor(np.stack(padded, axis=0), dtype=torch.float32, device=device)

  def describe(self) -> str:
    lines = [f"Motion library ({self.num_motions} clips):"]
    for i, m in enumerate(self.metadata):
      lines.append(
        f"  [{i}] {m.name:<22} {m.num_frames:>4} frames  "
        f"{m.distance:.2f} m  apex {m.goal_apex:.2f} m  "
        f"flight [{m.takeoff_step}, {m.land_step}]"
      )
    return "\n".join(lines)


def default_motion_dir() -> Path:
  """Where `dataset.py` writes its converted clips.

  Relative to the working directory, like every other dataset in `data/`, rather
  than to this package: motions are data, not code, and are not shipped with it.
  """
  return Path("data") / "asap" / "motions"


def discover_motion_files(
  motion_dir: Path | None = None, pattern: str = "*.npz"
) -> tuple[str, ...]:
  """Find converted clips, ordered by jump distance when a manifest exists.

  Ordering matters: the goal is a continuous distance, and interpolating between
  neighbouring clips is only meaningful if neighbouring means what it sounds like.
  """
  motion_dir = motion_dir or default_motion_dir()
  files = sorted(motion_dir.glob(pattern))

  manifest_path = motion_dir / "manifest.json"
  if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    order = {entry["file"]: entry["distance"] for entry in manifest}
    files.sort(key=lambda p: order.get(p.name, float("inf")))

  return tuple(str(p) for p in files)
