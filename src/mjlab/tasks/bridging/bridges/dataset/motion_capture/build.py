"""Build the retargeted motion capture corpus in one command.

1. Select BABEL intervals and LAFAN performances, see filters.py.
2. Extract the selected AMASS takes from their tarballs.
3. Retarget with GMR, ground every frame and reject clips failing floor QA.
4. Mark frames that fail the kinematic filter as invalid.

Needs, once:

    data/babel/babel_v1.0_release/{train,val,test}.json  https://babel.is.tue.mpg.de
    data/amass_atomic/tarballs/{CMU,KIT}.tar.bz2          mjlab.datasets.amass.download
    data/body_models/smplx/SMPLX_NEUTRAL.npz              https://smpl-x.is.tue.mpg.de

LAFAN is downloaded when missing. Each source lands in data/<source>_retargeted/<robot>/
as {train,val}/<category>/*.npz, with selection.jsonl (what was taken and why) and
qa.jsonl (floor QA per clip). A rerun keeps existing clips and refreshes the masks.

Run

1. Build the G1 corpus.

    uv run python -m mjlab.tasks.bridging.bridges.dataset.motion_capture.build --robot g1
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import tyro

import mjlab
from mjlab.datasets.amass.download import extract
from mjlab.datasets.lafan import download as lafan_download
from mjlab.retargeting.gmr.retarget import Clip, retarget_clips
from mjlab.tasks.bridging.bridges.dataset.motion_capture.filters import (
  FilterProfile,
  motion_bad_frames,
  motion_filter,
  select_babel,
  select_lafan,
)
from mjlab.tasks.bridging.config import RobotAlias, RobotAssetName, get_robot

BABEL_LABELS = Path("data/babel/babel_v1.0_release")
AMASS_TARBALLS = Path("data/amass_atomic/tarballs")
AMASS_SOURCES = Path("data/amass_babel")
LAFAN_RAW = Path("data/lafan/raw")


def _token(value: Any) -> str:
  return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_") or "unknown"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _babel_clips(entries: list[dict[str, Any]]) -> list[Clip]:
  clips = []
  for entry in entries:
    category = entry["categories"][0] if entry["categories"] else "other"
    name = f"{_token(entry['babel_sid'])}_{_token(entry['segment_id'])}.npz"
    metadata = {f"babel_{key.removeprefix('babel_')}": v for key, v in entry.items()}
    clips.append(
      Clip(
        Path(entry["source"]),
        Path(entry["split"], _token(category), name),
        float(entry["start_s"]),
        float(entry["end_s"]),
        metadata,
      )
    )
  return clips


def _lafan_clips(entries: list[dict[str, Any]]) -> list[Clip]:
  return [
    Clip(
      Path(entry["source"]),
      Path(entry["split"], entry["category"], Path(entry["source"]).stem + ".npz"),
      metadata={f"lafan_{key}": value for key, value in entry.items()},
    )
    for entry in entries
  ]


def _retarget(
  clips: list[Clip],
  selection: list[dict[str, Any]],
  robot: RobotAssetName,
  source_root: Path,
  output_root: Path,
  device: str,
) -> None:
  _write_jsonl(output_root / "selection.jsonl", selection)
  rows, skipped = retarget_clips(clips, robot, source_root, output_root, device=device)
  qa_path = output_root / "qa.jsonl"
  old = (
    [json.loads(line) for line in qa_path.read_text().splitlines()]
    if qa_path.is_file()
    else []
  )
  merged = {row["path"]: row for row in old + rows}
  _write_jsonl(qa_path, list(merged.values()))
  rejected = sum(row["status"] == "rejected" for row in rows)
  print(f"{output_root}: {len(rows)} retargeted, {rejected} rejected, {skipped} kept")


def _mark_valid(output_root: Path, profile: FilterProfile, robot: str) -> None:
  """Store the kinematic filter's verdict as a per-frame valid mask in every clip."""
  cfg = motion_filter(profile)
  seconds = valid_seconds = 0.0
  for path in sorted(output_root.glob("*/*/*.npz")):
    with np.load(path, allow_pickle=False) as motion:
      values = {name: motion[name] for name in motion.files}
    fps = float(np.asarray(values["fps"]).reshape(-1)[0])
    valid = np.ones(len(values["joint_pos"]), dtype=bool)
    if cfg is not None:
      state = np.concatenate(
        [values[f"body_{k}_w"][:, 0] for k in ("pos", "quat", "lin_vel", "ang_vel")]
        + [values["joint_pos"], values["joint_vel"]],
        axis=-1,
      )
      bad = motion_bad_frames(
        state, values["body_pos_w"], values["body_quat_w"], fps, cfg, robot
      )
      valid = ~np.logical_or.reduce(tuple(bad.values()))
    values["valid"] = valid
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **values)
    temporary.replace(path)
    seconds += len(valid) / fps
    valid_seconds += valid.sum() / fps
  print(f"{output_root}: {seconds / 60:.1f} min, {valid_seconds / 60:.1f} min valid")


def main(
  robot: RobotAlias = "g1",
  profile: FilterProfile = "bridge",
  device: str = "cuda:0",
) -> None:
  """Select, retarget, ground, QA and filter every BABEL and LAFAN clip."""
  selected = get_robot(robot)
  name = selected.robot_name

  babel = select_babel(BABEL_LABELS)
  missing = extract([entry["source"] for entry in babel], AMASS_TARBALLS, AMASS_SOURCES)
  if missing:
    print(f"Not in the AMASS archives, dropped: {missing}")
    babel = [entry for entry in babel if entry["source"] not in missing]
  babel_root = Path("data/babel_retargeted") / selected.babel_dataset
  _retarget(_babel_clips(babel), babel, name, AMASS_SOURCES, babel_root, device)
  _mark_valid(babel_root, profile, name)

  if not any(LAFAN_RAW.glob("*.bvh")):
    lafan_download.main()
  lafan = select_lafan(LAFAN_RAW)
  lafan_root = Path("data/lafan_retargeted") / name
  _retarget(_lafan_clips(lafan), lafan, name, LAFAN_RAW, lafan_root, device)
  _mark_valid(lafan_root, profile, name)


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
