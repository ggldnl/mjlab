"""Retarget and render one local AMASS clip on a supported robot.

The AMASS and SMPL-X licenses do not allow bundling their data. Download AMASS clips with
``mjlab.datasets.amass.download`` and place the SMPL-X model at
``data/body_models/smplx/SMPLX_NEUTRAL.npz`` before running this script.

Run:

    uv run python -m mjlab.tasks.bridging.tests.retarget_amass --robot unitree_g1
    uv run python -m mjlab.tasks.bridging.tests.retarget_amass --robot booster_t1
"""

from pathlib import Path

import tyro

import mjlab
from mjlab.retargeting.gmr import retarget
from mjlab.scripts import csv_to_npz


def select_clip(amass_dir: Path, skill: str, clip_index: int) -> Path:
  """Select one AMASS clip deterministically from a curated skill directory."""
  skill_dir = amass_dir / skill
  clips = sorted(path for path in skill_dir.rglob("*.npz") if path.is_file())
  if not clips:
    raise SystemExit(
      f"No AMASS clips found under {skill_dir}. Run the AMASS downloader first, or pass "
      "--clip with a SMPL-X stageii NPZ."
    )
  if not 0 <= clip_index < len(clips):
    raise SystemExit(
      f"--clip-index {clip_index} is outside the {len(clips)} clips in {skill_dir}"
    )
  return clips[clip_index]


def main(
  robot: csv_to_npz.RobotName = "unitree_g1",
  clip: Path | None = None,
  amass_dir: Path = Path("data/amass_atomic"),
  skill: str = "walk",
  clip_index: int = 0,
  output_dir: Path | None = None,
  smplx_dir: Path = retarget.SMPLX_DIR,
  device: str = "cuda:0",
) -> None:
  """Retarget one clip and save an MP4 rendered with the selected mjlab robot."""
  input_path = clip or select_clip(amass_dir, skill, clip_index)
  if not input_path.is_file():
    raise SystemExit(f"No such AMASS clip: {input_path}")
  output_dir = output_dir or Path("data/retarget_preview") / robot

  print(f"Retargeting {input_path} to {robot}")
  retarget.main(
    input_path=input_path,
    output_dir=output_dir,
    smplx_dir=smplx_dir,
    robot=robot,
    device=device,
    render=True,
  )
  print(f"Rendered preview: {output_dir / f'{input_path.stem}.mp4'}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
