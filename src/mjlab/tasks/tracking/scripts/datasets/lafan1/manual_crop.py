"""Slice clean, atomic single-skill clips out of the long LAFAN1 G1 motions.

The Unitree-retargeted LAFAN1 CSVs are long, continuous performances (e.g.
``walk1_subject1`` is ~260 s of a subject walking around, turning, idling).
For the latent-skill work we want *fundamental* skills as clean, isolated
clips: a steady forward walk, a run, a jump, a crouch. This script carves
those out by frame range and converts each to the NPZ format the
``Mjlab-Tracking-Flat-Unitree-G1`` task trains on, reusing
``mjlab.scripts.csv_to_npz`` so the body ordering stays correct.

Workflow:
  1. Download the source clips (only the fundamental categories needed):
       uv run python src/mjlab/tasks/tracking/scripts/datasets/download.py \
           --output-dir data/lafan1_g1 \
           --motions walk1_subject1 run1_subject2 sprint1_subject1 jumps1_subject1
  2. Find clean windows. Render a candidate window straight from this script
     (writes only an MP4, no NPZ) and iterate on the frame range until the
     stretch is a single steady skill:
       MUJOCO_GL=egl uv run python \
           src/mjlab/tasks/tracking/scripts/datasets/manual_crop.py \
           --preview-source walk1_subject1 --preview-range 1000 1300
     Or preview the windows already recorded in ``SKILLS`` with ``--preview``.
  3. Record the windows in ``SKILLS`` below, then run this script to emit one
     atomic NPZ per skill:
       MUJOCO_GL=egl uv run python \
           src/mjlab/tasks/tracking/scripts/datasets/manual_crop.py \
           --data-dir data/lafan1_g1 --output-dir data/lafan1_g1/skills

The frame ranges below are reasonable *starting guesses* into mid-clip
steady regions; verify and adjust them against the rendered video before
trusting them. ``line_range`` is 1-indexed and inclusive, matching
``csv_to_npz``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.scripts import csv_to_npz


@dataclass(frozen=True)
class Skill:
  """One atomic skill carved from a source LAFAN1 clip."""

  source: str  # CSV file name in ``data_dir`` (with or without ``.csv``).
  start: int  # First frame, 1-indexed inclusive.
  end: int  # Last frame, 1-indexed inclusive.


# output_name -> slice. Keep each window to a single, repeating skill; a few
# seconds (≈100-300 frames at 30 fps) is plenty for a periodic gait.
SKILLS: dict[str, Skill] = {
  "walk_forward": Skill("walk1_subject1", start=1000, end=1300),
  "run_forward": Skill("run2_subject1", start=400, end=650),
  "sprint_forward": Skill("sprint1_subject1", start=200, end=400),
  "jump": Skill("jumps1_subject1", start=300, end=500),
}


def main(
  data_dir: Path = Path("data/lafan1_g1"),
  output_dir: Path = Path("data/lafan1_g1/skills"),
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  skills: tuple[str, ...] = (),
  preview: bool = False,
  preview_source: str | None = None,
  preview_range: tuple[int, int] | None = None,
) -> None:
  """Convert each entry in ``SKILLS`` to an atomic tracking NPZ.

  Args:
    data_dir: Directory holding the source LAFAN1 CSVs.
    output_dir: Directory to write the per-skill NPZs (and videos) into.
    input_fps: Frame rate of the source CSVs (LAFAN1 is 30).
    output_fps: Output frame rate (tracking task default is 50).
    device: Torch device for the MuJoCo replay.
    render: Also write an MP4 of each sliced skill for a quick sanity check.
    skills: Subset of ``SKILLS`` keys to build. Empty builds all of them.
    preview: Render the selected window(s) to MP4 only and delete the NPZ
      afterwards. Use this to eyeball candidate frame ranges without
      committing them; forces rendering on.
    preview_source: For ad-hoc preview, the source clip name to slice. Pair
      with ``preview_range`` to try a window that is not yet in ``SKILLS``
      (writes a single ``preview.mp4``). Implies ``preview``.
    preview_range: ``(start, end)`` frames (1-indexed, inclusive) for the
      ad-hoc preview. Requires ``preview_source``.
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA unavailable; falling back to CPU (slow).")
    device = "cpu"

  ad_hoc = preview_source is not None or preview_range is not None
  if ad_hoc:
    if preview_source is None or preview_range is None:
      raise SystemExit(
        "Ad-hoc preview needs both --preview-source and --preview-range."
      )
    preview = True
    worklist = [("preview", Skill(preview_source, *preview_range))]
  else:
    wanted = skills or tuple(SKILLS)
    unknown = [s for s in wanted if s not in SKILLS]
    if unknown:
      raise SystemExit(
        f"Unknown skill(s): {', '.join(unknown)}. Available: {', '.join(SKILLS)}"
      )
    worklist = [(name, SKILLS[name]) for name in wanted]

  for name, skill in worklist:
    source = skill.source if skill.source.endswith(".csv") else f"{skill.source}.csv"
    input_file = data_dir / source
    if not input_file.exists():
      raise SystemExit(
        f"Missing source clip {input_file}. Download it first with download.py."
      )
    print(f"\n=== {name}: {source} frames {skill.start}-{skill.end} ===")
    csv_to_npz.main(
      input_file=str(input_file),
      output_name=name,
      output_dir=output_dir,
      input_fps=input_fps,
      output_fps=output_fps,
      device=device,
      render=render or preview,
      upload_to_wandb=False,
      line_range=(skill.start, skill.end),
    )
    if preview:
      # Keep only the MP4 so previewing never pollutes the skill set.
      (output_dir / f"{name}.npz").unlink(missing_ok=True)

  if preview:
    print(f"\nPreview video(s) written to {output_dir}/ (no NPZ kept).")
  else:
    print(f"\nDone. Atomic skill NPZs written to {output_dir}/")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
