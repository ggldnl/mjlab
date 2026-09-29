"""Retarget the selected original LAFAN BVH performances with GMR.

Run:

  uv run python -m mjlab.retargeting.gmr.retarget_lafan --robot g1
  uv run python -m mjlab.retargeting.gmr.retarget_lafan --robot t1
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tyro

import mjlab
from mjlab.datasets.lafan.build_manifest import read_manifest
from mjlab.retargeting.gmr import retarget
from mjlab.scripts import csv_to_npz
from mjlab.tasks.bridging.config import RobotAlias, get_robot


def _destination(output_dir: Path, entry: dict[str, Any]) -> Path:
  return (
    output_dir
    / str(entry["split"])
    / str(entry["category"])
    / Path(str(entry["source"])).with_suffix(".npz").name
  )


def retarget_entries(
  manifest_path: Path,
  input_dir: Path,
  output_dir: Path,
  robot: RobotAlias,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  keep_csv: bool = True,
  overwrite: bool = False,
  limit: int | None = None,
  verbose: bool = False,
  qa_report: Path | None = None,
) -> tuple[int, int, int]:
  """Retarget LAFAN manifest entries. Returns accepted, skipped, rejected."""
  entries = read_manifest(manifest_path)
  if limit is not None:
    if limit < 1:
      raise ValueError("limit must be positive")
    entries = entries[:limit]
  target_robot = get_robot(robot).robot_name
  joint_names = csv_to_npz.robot_joint_names(target_robot)
  accepted = skipped = rejected = 0
  qa_rows = []
  for entry in entries:
    source = input_dir / str(entry["source"])
    if not source.is_file():
      raise FileNotFoundError(f"Missing LAFAN source: {source}")
    destination = _destination(output_dir, entry)
    if destination.is_file() and not overwrite:
      skipped += 1
      continue
    csv_path = output_dir / "csv" / source.with_suffix(".csv").name
    fps = retarget.retarget_bvh_clip(
      source, csv_path, target_robot, joint_names, verbose
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    csv_to_npz.main(
      input_file=str(csv_path),
      output_name=destination.stem,
      robot=target_robot,
      output_dir=destination.parent,
      input_fps=fps,
      output_fps=output_fps,
      device=device,
      render=render,
      upload_to_wandb=False,
      line_range=None,
    )
    qa: dict[str, Any] = {
      "path": destination.as_posix(),
      "source": entry["source"],
      "split": entry["split"],
      "status": "accepted",
      "reason": "",
    }
    floor = retarget.finalize_motion(
      destination,
      {
        "lafan_source": np.asarray(entry["source"]),
        "lafan_split": np.asarray(entry["split"]),
        "lafan_category": np.asarray(entry["category"]),
        "lafan_take": np.asarray(entry["take"]),
        "lafan_subject": np.asarray(entry["subject"]),
      },
    )
    qa.update(floor)
    qa_rows.append(qa)
    if floor["status"] == "accepted":
      accepted += 1
    else:
      rejected += 1
    if not keep_csv:
      csv_path.unlink(missing_ok=True)
  if qa_report is not None:
    qa_report.parent.mkdir(parents=True, exist_ok=True)
    qa_report.write_text(
      "".join(json.dumps(row, sort_keys=True) + "\n" for row in qa_rows),
      encoding="utf-8",
    )
  return accepted, skipped, rejected


def main(
  manifest_path: Path = Path("data/lafan/manifest.jsonl"),
  input_dir: Path = Path("data/lafan/raw"),
  output_dir: Path | None = None,
  robot: RobotAlias = "g1",
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  keep_csv: bool = True,
  overwrite: bool = False,
  limit: int | None = None,
  verbose: bool = False,
  qa_report: Path | None = None,
) -> None:
  """Retarget every selected LAFAN performance and apply floor QA."""
  target_robot = get_robot(robot).robot_name
  output_dir = output_dir or Path("data/lafan_retargeted") / target_robot
  qa_report = qa_report or output_dir / "qa.jsonl"
  accepted, skipped, rejected = retarget_entries(
    manifest_path,
    input_dir,
    output_dir,
    robot,
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
    f"Retargeted {accepted} LAFAN clips to {robot} ({target_robot}); "
    f"rejected {rejected}; "
    f"skipped {skipped}; QA: {qa_report}"
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
