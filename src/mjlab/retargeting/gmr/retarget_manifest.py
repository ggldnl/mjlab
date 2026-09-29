"""Retarget timestamped BABEL safe regions with GMR.

Run:

  uv run python -m mjlab.retargeting.gmr.retarget_manifest --robot unitree_g1
  uv run python -m mjlab.retargeting.gmr.retarget_manifest --robot booster_t1
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tyro

import mjlab
from mjlab.datasets.babel.build_manifest import read_manifest
from mjlab.datasets.babel.materialize import source_path
from mjlab.retargeting.gmr import retarget
from mjlab.scripts import csv_to_npz
from mjlab.tasks.bridging.motion_filter import align_g1_floor


def _token(value: Any) -> str:
  return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_") or "unknown"


def _segment_path(output_dir: Path, entry: dict[str, Any]) -> Path:
  categories = entry.get("categories")
  category = categories[0] if isinstance(categories, list) and categories else "other"
  segment = entry.get("segment_id")
  if not segment:
    segment = f"{float(entry['start_s']):.3f}_{float(entry['end_s']):.3f}"
  name = f"{_token(entry['babel_sid'])}_{_token(segment)}.npz"
  return output_dir / _token(entry["split"]) / _token(category) / name


def _line_range(entry: dict[str, Any], fps: float, frames: int) -> tuple[int, int]:
  start = float(entry["start_s"])
  end = float(entry["end_s"])
  if start < 0 or end <= start:
    raise ValueError(f"Invalid BABEL interval: {start:g} to {end:g}")
  first = max(0, math.ceil(start * fps - 1e-9))
  stop = min(frames, math.ceil(end * fps - 1e-9))
  if stop - first < 3:
    raise ValueError(f"BABEL interval {start:g} to {end:g} has fewer than 3 frames")
  return first + 1, stop


def _add_metadata(path: Path, entry: dict[str, Any]) -> dict[str, Any]:
  with np.load(path, allow_pickle=False) as motion:
    values = {name: motion[name] for name in motion.files}
  qa: dict[str, Any] = {
    "path": path.as_posix(),
    "source": entry["source"],
    "split": entry["split"],
    "start_s": float(entry["start_s"]),
    "end_s": float(entry["end_s"]),
    "status": "accepted",
    "reason": "",
  }
  if (
    str(values.get("robot", "")) == "unitree_g1"
    and {
      "body_pos_w",
      "body_quat_w",
    }
    <= values.keys()
  ):
    values["body_pos_w"], floor = align_g1_floor(
      values["body_pos_w"], values["body_quat_w"]
    )
    qa.update(floor)
    if floor["status"] == "rejected":
      path.unlink(missing_ok=True)
      return qa
    values["ground_z_offset"] = np.asarray(floor["offset"], dtype=np.float32)
    values["ground_floor_z"] = np.asarray(floor["floor_z"], dtype=np.float32)
    values["ground_minimum_sole_z"] = np.asarray(
      floor["minimum_sole_z"], dtype=np.float32
    )
    values["ground_contact_fraction"] = np.asarray(
      floor["contact_fraction"], dtype=np.float32
    )
  values.update(
    babel_sid=np.asarray(entry["babel_sid"]),
    babel_segment_id=np.asarray(entry.get("segment_id") or ""),
    babel_segment_ids=np.asarray(entry.get("segment_ids", [])),
    babel_split=np.asarray(entry["split"]),
    babel_source=np.asarray(entry["source"]),
    babel_start_s=np.asarray(float(entry["start_s"])),
    babel_end_s=np.asarray(float(entry["end_s"])),
    babel_categories=np.asarray(entry.get("categories", [])),
    babel_label=np.asarray(entry.get("label") or ""),
    babel_labels=np.asarray(entry.get("labels", [])),
  )
  temporary = path.with_suffix(".tmp.npz")
  np.savez(temporary, **values)
  temporary.replace(path)
  return qa


def retarget_entries(
  manifest_path: Path,
  input_dir: Path,
  output_dir: Path,
  smplx_dir: Path,
  robot: csv_to_npz.RobotName,
  retarget_fps: float = retarget.RETARGET_FPS,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  keep_csv: bool = True,
  overwrite: bool = False,
  limit: int | None = None,
  verbose: bool = False,
  qa_report: Path | None = None,
) -> tuple[int, int, int]:
  """Retarget manifest entries. Returns accepted, skipped, and rejected counts."""
  if limit is not None and limit < 1:
    raise ValueError("limit must be positive")
  entries = read_manifest(manifest_path)
  if limit is not None:
    entries = entries[:limit]

  pending = []
  skipped = 0
  for entry in entries:
    destination = _segment_path(output_dir, entry)
    if destination.is_file() and not overwrite:
      skipped += 1
    else:
      pending.append(entry)

  by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
  for entry in pending:
    by_source[str(entry["source"])].append(entry)

  joint_names = csv_to_npz.robot_joint_names(robot)
  converted = rejected = 0
  qa_rows: list[dict[str, Any]] = []
  for source, source_entries in sorted(by_source.items()):
    smplx_file = input_dir / source_path(source)
    if not smplx_file.is_file():
      raise FileNotFoundError(
        f"Missing materialized AMASS source: {smplx_file}. "
        "Run mjlab.datasets.babel.materialize first."
      )

    csv_path = output_dir / "csv" / source_path(source).with_suffix(".csv")
    fps_path = csv_path.with_suffix(".fps")
    if csv_path.is_file() and fps_path.is_file() and not overwrite:
      fps = float(fps_path.read_text(encoding="utf-8"))
    else:
      fps = retarget.retarget_clip(
        smplx_file,
        csv_path,
        smplx_dir,
        robot,
        joint_names,
        retarget_fps,
        verbose,
      )
      fps_path.write_text(f"{fps:g}\n", encoding="utf-8")

    with csv_path.open(encoding="utf-8") as csv_file:
      frames = sum(1 for line in csv_file if line.strip())
    for entry in source_entries:
      destination = _segment_path(output_dir, entry)
      destination.parent.mkdir(parents=True, exist_ok=True)
      csv_to_npz.main(
        input_file=str(csv_path),
        output_name=destination.stem,
        robot=robot,
        output_dir=destination.parent,
        input_fps=fps,
        output_fps=output_fps,
        device=device,
        render=render,
        upload_to_wandb=False,
        line_range=_line_range(entry, fps, frames),
      )
      qa = _add_metadata(destination, entry)
      qa_rows.append(qa)
      if qa["status"] == "accepted":
        converted += 1
      else:
        rejected += 1

    if not keep_csv:
      csv_path.unlink(missing_ok=True)
      fps_path.unlink(missing_ok=True)

  if qa_report is not None:
    qa_report.parent.mkdir(parents=True, exist_ok=True)
    qa_report.write_text(
      "".join(json.dumps(row, sort_keys=True) + "\n" for row in qa_rows),
      encoding="utf-8",
    )
  return converted, skipped, rejected


def main(
  manifest_path: Path = Path("data/babel/manifest.jsonl"),
  input_dir: Path = Path("data/amass_babel"),
  output_dir: Path | None = None,
  smplx_dir: Path = retarget.SMPLX_DIR,
  robot: csv_to_npz.RobotName = "unitree_g1",
  retarget_fps: float = retarget.RETARGET_FPS,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  keep_csv: bool = True,
  overwrite: bool = False,
  limit: int | None = None,
  verbose: bool = False,
  qa_report: Path | None = None,
) -> None:
  """Retarget all selected BABEL regions, reusing one GMR solve per source."""
  output_dir = output_dir or Path("data/babel_retargeted") / robot
  qa_report = qa_report or output_dir / "qa.jsonl"
  converted, skipped, rejected = retarget_entries(
    manifest_path,
    input_dir,
    output_dir,
    smplx_dir,
    robot,
    retarget_fps,
    output_fps,
    device,
    render,
    keep_csv,
    overwrite,
    limit,
    verbose,
    qa_report,
  )
  print(
    f"Retargeted {converted} BABEL segments to {robot}; "
    f"rejected {rejected}; skipped {skipped}; QA: {qa_report}"
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
