"""Read and write retargeted motion clips (the npz layout of BABEL and LAFAN)."""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mjlab.retargeting.floor import robot_foot_positions
from mjlab.tasks.bridging.config import get_robot

CORPORA = ("BABEL", "LAFAN", "Stitched")

STITCHED_ROOT = Path("data") / "motion_graph"


def corpus_root(corpus: str, robot: str) -> Path:
  """Folder holding {train,val}/<category>/*.npz for one corpus."""
  selected = get_robot(robot)
  if corpus == "BABEL":
    return Path("data") / "babel_retargeted" / selected.babel_dataset
  if corpus == "LAFAN":
    return Path("data") / "lafan_retargeted" / selected.robot_name
  if corpus == "Stitched":
    return STITCHED_ROOT / selected.robot_name
  raise ValueError(f"corpus is one of {CORPORA}, not '{corpus}'")


def corpus_files(corpus: str, robot: str, split: str) -> list[Path]:
  pattern = corpus_root(corpus, robot) / split / "**" / "*.npz"
  return sorted(Path(p) for p in glob.glob(str(pattern), recursive=True))


@dataclass
class Clip:
  name: str
  category: str
  states: np.ndarray
  """(T, 13 + 2J) root pos, quat (wxyz), lin vel, ang vel, joint pos, joint vel."""
  valid: np.ndarray
  """(T,) frames the kinematic filter accepts."""
  feet: np.ndarray
  """(T, 2, 3) left and right sole positions in world."""
  seams: np.ndarray
  """Frames where a stitched clip switches source. Empty for recorded clips."""
  pieces: tuple[str, ...]
  """Source clip of every piece of a stitched clip, as category/name."""


@dataclass
class Corpus:
  clips: list[Clip]
  fps: float
  robot: str
  joint_names: tuple[str, ...]
  body_names: tuple[str, ...]


def load(files: list[Path]) -> Corpus:
  if not files:
    raise SystemExit("No motion clips found.")
  clips: list[Clip] = []
  first: dict[str, np.ndarray] = {}
  for path in files:
    with np.load(path, allow_pickle=False) as raw:
      values = {k: raw[k] for k in raw.files}
    if not first:
      first = values
    elif values["joint_names"].tolist() != first["joint_names"].tolist():
      raise SystemExit(f"{path} uses a different joint order.")
    body_pos, body_quat = values["body_pos_w"], values["body_quat_w"]
    states = np.concatenate(
      [
        body_pos[:, 0],
        body_quat[:, 0],
        values["body_lin_vel_w"][:, 0],
        values["body_ang_vel_w"][:, 0],
        values["joint_pos"],
        values["joint_vel"],
      ],
      axis=-1,
    ).astype(np.float32)
    clips.append(
      Clip(
        name=path.stem,
        category=path.parent.name,
        states=states,
        valid=values["valid"].astype(bool),
        feet=robot_foot_positions(body_pos, body_quat, str(values["robot"])),
        seams=values.get("stitch_seams", np.zeros(0, dtype=np.int64)),
        pieces=tuple(str(p) for p in values.get("stitch_pieces", ())),
      )
    )
  return Corpus(
    clips=clips,
    fps=float(np.asarray(first["fps"]).reshape(-1)[0]),
    robot=str(first["robot"]),
    joint_names=tuple(str(n) for n in first["joint_names"]),
    body_names=tuple(str(n) for n in first["body_names"]),
  )


def save(path: Path, corpus: Corpus, arrays: dict[str, np.ndarray]) -> None:
  """Write one clip with the keys the BABEL and LAFAN loaders read."""
  path.parent.mkdir(parents=True, exist_ok=True)
  columns: dict[str, Any] = {
    "fps": np.asarray([corpus.fps]),
    "robot": np.asarray(corpus.robot),
    "joint_names": np.asarray(corpus.joint_names),
    "body_names": np.asarray(corpus.body_names),
    **arrays,
  }
  np.savez(path, allow_pickle=False, **columns)
