"""Adapter: PHUMA G1 retargeted motions -> mjlab tracking pipeline.

PHUMA (Lim et al., "PHUMA: Physically-Grounded Humanoid Locomotion Dataset",
arXiv:2510.26236) is a 73-hour locomotion corpus that has been *physics-aware
curated* (implausible foot contact corrected, corrupted clips dropped) and
*physics-constrained retargeted* to humanoid robots -- including the Unitree
G1. So unlike raw AMASS, the data already ships as G1 joint trajectories: no
SMPL retargeting needed on our side. Clips are named by motion (``walk_*``,
``run_*``, ``jump_*``, ...), so curating atomic skills is a filename match.

Each clip is a ``.npy`` dict with ``root_trans`` (N,3), ``root_ori`` (N,4) quat
in xyzw, ``dof_pos`` (N,29), and ``fps``. The ``dof_pos`` is already the full
**29-DOF** Unitree G1 layout (3-DOF waist + 7-DOF arms incl. wrists) -- the same
order ``csv_to_npz`` expects -- so no joint remap is needed (unlike the 23-DOF
openhe adapter). The only mismatch is the container: a ``.npy`` dict rather than
the Unitree generalized-coordinate CSV ``csv_to_npz`` reads.

This script bridges the gap. For each clip it:

  1. (once) downloads and extracts the PHUMA ``data.zip`` (the ``g1/`` set),
  2. writes a Unitree-convention CSV -- base position, base quaternion in xyzw,
     then the 29 joint angles passed straight through,
  3. replays that CSV through ``csv_to_npz`` so MuJoCo forward kinematics fill
     in per-body poses with the correct (MuJoCo) body ordering.

The 29-DOF ``dof_pos`` is assumed to follow the canonical Unitree G1 joint order
(the order ``csv_to_npz`` feeds to ``find_joints(..., preserve_order=True)``);
it is validated to be 29-wide. Render a converted clip (``--render True``) to
sanity-check the joint mapping. Use ``--list`` to inspect the available clips
(grouped by skill) before converting.

Note: the public HF release *excludes* LAFAN1- and LocoMuJoCo-derived motions
for licensing reasons; the AMASS/video-derived walk/run/jump/crouch clips are
present. Cite PHUMA (arXiv:2510.26236) and the upstream sources accordingly.

``--skills`` defaults to all of walk/run/sprint/jump/crouch. To override it,
note that mjlab's tyro config uses Python-literal syntax for collections, so
pass a quoted tuple of quoted strings, e.g. ``--skills "('walk','run')"`` (plain
space-separated ``walk run`` will not parse).

Run with:
  # 1. (large, ~3.4 GB) download + convert one clip per skill, with a video:
  uv run python src/mjlab/tasks/tracking/scripts/datasets/download.py \
      --output-dir data/phuma_g1 --max-per-skill 1 --render True
  # inspect what's available first, without converting:
  uv run python src/mjlab/tasks/tracking/scripts/datasets/download.py --list True
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

import mjlab
from mjlab.scripts import csv_to_npz

HF_DATA_ZIP = (
  "https://huggingface.co/datasets/DAVIAN-Robotics/PHUMA/resolve/main/data.zip"
)

# PHUMA retargets to the full 29-DOF Unitree G1 (3-DOF waist + 7-DOF arms incl.
# wrists), the same layout csv_to_npz expects -- so dof_pos passes straight
# through, no remap. Verified against the data: clips store dof_pos as (N, 29).
G1_NUM_DOF = 29

# Atomic skill -> filename keywords (matched case-insensitively against each
# clip's ``.npy`` name). Conservative on purpose.
SKILL_KEYWORDS: dict[str, tuple[str, ...]] = {
  "walk": ("walk",),
  "run": ("run", "jog"),
  "sprint": ("sprint", "dash"),
  "jump": ("jump", "hop", "leap"),
  "crouch": ("crouch", "squat", "kneel"),
}


def _download_and_extract(cache_dir: Path) -> Path:
  """Download (once) and extract PHUMA's data.zip; return the ``g1/`` dir."""
  g1_dir = cache_dir / "data" / "g1"
  if g1_dir.is_dir():
    return g1_dir
  cache_dir.mkdir(parents=True, exist_ok=True)
  zip_path = cache_dir / "data.zip"
  if not zip_path.exists():
    tmp = zip_path.with_suffix(".zip.part")
    print(f"Downloading PHUMA data.zip (~3.4 GB) -> {zip_path}")
    urllib.request.urlretrieve(HF_DATA_ZIP, tmp)
    tmp.replace(zip_path)
  print(f"Extracting {zip_path} -> {cache_dir}")
  with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(cache_dir)
  if not g1_dir.is_dir():
    raise SystemExit(
      f"Expected a g1/ folder after extraction; not found in {cache_dir}."
    )
  return g1_dir


def _classify(filename: str, skills: tuple[str, ...]) -> str | None:
  name = filename.lower()
  for skill in skills:
    if any(kw in name for kw in SKILL_KEYWORDS[skill]):
      return skill
  return None


def _discover(g1_dir: Path, skills: tuple[str, ...]) -> dict[str, list[Path]]:
  """Group ``g1/`` clips by atomic skill (sorted for determinism)."""
  grouped: dict[str, list[Path]] = {skill: [] for skill in skills}
  for path in sorted(g1_dir.rglob("*.npy")):
    skill = _classify(path.name, skills)
    if skill is not None:
      grouped[skill].append(path)
  return grouped


def _load_npy(path: Path) -> dict[str, Any]:
  data = np.load(path, allow_pickle=True)
  return data.item() if data.dtype == object else dict(data)


def _to_unitree_csv(clip: dict[str, Any]) -> np.ndarray:
  """Build the (N, 36) Unitree CSV matrix: pos(3) + quat_xyzw(4) + dof(29)."""
  pos = np.asarray(clip["root_trans"], dtype=np.float32)  # (N, 3)
  quat = np.asarray(clip["root_ori"], dtype=np.float32)  # (N, 4) xyzw
  dof = np.asarray(clip["dof_pos"], dtype=np.float32)  # (N, 29)
  if dof.shape[1] != G1_NUM_DOF:
    raise SystemExit(
      f"Expected {G1_NUM_DOF}-DOF G1 dof_pos, got {dof.shape[1]}. PHUMA's G1 "
      "joint layout may have changed; check the clip against csv_to_npz's "
      "expected 29-DOF order."
    )
  return np.concatenate([pos, quat, dof], axis=1)


def main(
  output_dir: Path = Path("data/phuma_g1"),
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  skills: tuple[str, ...] = tuple(SKILL_KEYWORDS),
  max_per_skill: int = 5,
  keep_csv: bool = False,
  list: bool = False,
) -> None:
  """Convert curated PHUMA G1 clips to tracking NPZs.

  Args:
    output_dir: Directory for the NPZs (PHUMA data is cached under
      ``output_dir/phuma_cache``; intermediate CSVs/videos alongside NPZs).
    output_fps: Output frame rate for the NPZ (tracking default is 50).
    device: Torch device for the MuJoCo replay in ``csv_to_npz``.
    render: Also write an MP4 of each converted clip.
    skills: Atomic skills to build (walk/run/sprint/jump/crouch).
    max_per_skill: Cap on clips converted per skill (PHUMA has many; keep this
      small for a first pass).
    keep_csv: Keep the intermediate Unitree CSVs (otherwise removed).
    list: Only download/extract and print the clips grouped by skill, then exit
      (no conversion). Use this to verify naming before committing GPU time.
  """
  unknown = [s for s in skills if s not in SKILL_KEYWORDS]
  if unknown:
    raise SystemExit(
      f"Unknown skill(s): {', '.join(unknown)}. Available: {', '.join(SKILL_KEYWORDS)}"
    )

  cache_dir = output_dir / "phuma_cache"
  g1_dir = _download_and_extract(cache_dir)
  grouped = _discover(g1_dir, skills)

  if list:
    for skill in skills:
      clips = grouped[skill]
      print(f"\n{skill}: {len(clips)} clip(s)")
      for p in clips[:20]:
        print(f"  {p.relative_to(g1_dir)}")
      if len(clips) > 20:
        print(f"  ... (+{len(clips) - 20} more)")
    return

  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA unavailable; falling back to CPU (slow).")
    device = "cpu"

  csv_dir = output_dir / "csv"
  csv_dir.mkdir(parents=True, exist_ok=True)

  for skill in skills:
    clips = grouped[skill][:max_per_skill]
    if not clips:
      print(f"\n=== {skill}: no clips matched; skipping ===")
      continue
    for i, clip_path in enumerate(clips):
      name = f"{skill}_{i}" if len(clips) > 1 else skill
      print(f"\n=== {name}: {clip_path.relative_to(g1_dir)} ===")
      clip = _load_npy(clip_path)
      matrix = _to_unitree_csv(clip)
      clip_fps = float(clip.get("fps", 30.0))

      csv_path = csv_dir / f"{name}.csv"
      np.savetxt(csv_path, matrix, delimiter=",")
      print(f"  wrote {matrix.shape[0]} frames @ {clip_fps} fps -> {csv_path}")

      csv_to_npz.main(
        input_file=str(csv_path),
        output_name=name,
        output_dir=output_dir,
        input_fps=clip_fps,
        output_fps=output_fps,
        device=device,
        render=render,
        upload_to_wandb=False,
        line_range=None,
      )
      if not keep_csv:
        csv_path.unlink(missing_ok=True)

  print(
    f"\nDone. Atomic skill NPZs in {output_dir}/. Train one with:\n"
    "  uv run train Mjlab-Tracking-Flat-Unitree-G1 "
    f"--env.commands.motion.motion-file {output_dir}/walk.npz --env.scene.num-envs 4096"
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
