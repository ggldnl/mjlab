"""Download OmniRetarget's G1 robot-terrain clips and the obstacles they were solved against.

OmniRetarget (Yang et al., arXiv:2509.26633) retargets human motion to the 29 DOF G1 while
preserving the contacts, so a climb clip and the box it climbs are solved together. That is
the property this dataset is here for: a clip retargeted without the obstacle has no defined
relationship to one placed in simulation, and the hands land a few centimetres above or
below the face they are supposed to push off.

Two things are fetched, and a climb needs both:

    clips      robot-terrain.zip, 145 npz files, one per scene and height scale. Each is
               qpos [T, 36]: root quaternion in wxyz, root position, then 29 joint angles
    obstacle   models/terrain/<scene>/multi_boxes_z_scale_<scale>.urdf and the OBJ meshes
               it references. The box the clip was solved against, in the clip's own
               world frame

There are 29 climb scenes, each shipping five height scales. The scale is applied to Z
alone, and the clip is re-solved against the scaled box rather than stretched, so a scale is
a different climb rather than the same one played taller.

The clips are 30 fps and their qpos is quaternion first in wxyz, which is neither of the
conventions the other adapters here read. Converting is the caller's job; see
skills/climb/dataset.py for the one that does it.

Run

1. Fetch the clips and the obstacle for the default scene.

    uv run python src/mjlab/tasks/tracking/scripts/datasets/omniretarget/download.py

2. Fetch a different scene's obstacle.

    uv run python src/mjlab/tasks/tracking/scripts/datasets/omniretarget/download.py \
      --scenes "('climb_14','climb_21')"

The dataset is MIT licensed. Cite OmniRetarget (Yang et al., arXiv:2509.26633) in any
publication.
"""

from __future__ import annotations

import re
import urllib.request
import zipfile
from pathlib import Path

import tyro

import mjlab

REPO = "omniretarget/OmniRetarget_Dataset"
RAW_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/{{path}}"

TERRAIN_ZIP = "robot-terrain.zip"
"""The climbing subset. 38 MB, and the only one of the three this task wants: robot-object
is loco-manipulation and robot-object-terrain drags a chair up the box."""

DEFAULT_DIR = Path("data") / "omniretarget"
"""Where the archive, the extracted clips and the obstacle models are cached."""

Z_SCALES: tuple[str, ...] = ("0.8", "0.9", "1.0", "1.1", "1.2")
"""Height scales every scene ships. Written as strings because they are part of a filename
and 1.0 has to stay "1.0"."""

CLIMB_SCENES: tuple[str, ...] = tuple(f"climb_{i:02d}" for i in range(29))
"""Every climbing scene in the release."""

NOT_CLIMBS: tuple[str, ...] = (
  "climb_08",
  "climb_09",
  "climb_16",
  "climb_19",
  "climb_27",
)
"""Scenes whose root never rises far enough to be on the box, measured over the whole set.

Named here rather than left to be discovered, because the file name says climb and the
motion does not: whatever the subject was doing, converting one of these gives a tracker
with nothing to climb and no error to read."""


def clip_name(scene: str, z_scale: str = "1.0") -> str:
  """Name of one clip, which is also its stem inside the archive."""
  return f"{scene}_z_scale_{z_scale}"


def _download(path: str, destination: Path) -> Path:
  """Fetch one repository path, unless it is already cached."""
  if destination.exists():
    return destination

  destination.parent.mkdir(parents=True, exist_ok=True)
  print(f"  downloading {path}")
  # Written under a temporary name first, so an interrupted download cannot leave a
  # truncated file behind that later runs treat as cached
  partial = destination.with_suffix(destination.suffix + ".part")
  try:
    urllib.request.urlretrieve(RAW_URL.format(path=path), partial)
    partial.replace(destination)
  except Exception as exc:
    partial.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {path}: {exc}") from exc
  return destination


def fetch(name: str, output_dir: Path = DEFAULT_DIR) -> Path:
  """Extract one clip's npz, downloading the archive first if it is not cached.

  Args:
    name: Clip name, as ``clip_name`` builds it.
    output_dir: Directory to cache the archive and the extracted clips in.
  """
  destination = output_dir / "robot-terrain" / f"{name}.npz"
  if destination.exists():
    return destination

  archive = _download(TERRAIN_ZIP, output_dir / TERRAIN_ZIP)
  member = f"robot-terrain/{name}.npz"
  with zipfile.ZipFile(archive) as bundle:
    if member not in bundle.namelist():
      raise ValueError(f"'{name}' is not in {TERRAIN_ZIP}")
    bundle.extract(member, output_dir)
  return destination


def fetch_terrain(
  scene: str, z_scale: str = "1.0", output_dir: Path = DEFAULT_DIR
) -> Path:
  """Download one scene's obstacle description and return the URDF.

  The URDF names its meshes by a path relative to itself, so they are fetched alongside it
  and the file can be read where it lands. Only the meshes this scale references are
  downloaded, which is all of them: a scale changes the mesh scale attribute, not the mesh.

  Args:
    scene: Scene name, one of ``CLIMB_SCENES``.
    z_scale: Which height variant's URDF to read.
    output_dir: Directory to cache the models under.
  """
  relative = f"models/terrain/{scene}/multi_boxes_z_scale_{z_scale}.urdf"
  urdf = _download(relative, output_dir / relative)

  for mesh in sorted(set(re.findall(r'mesh filename="([^"]+)"', urdf.read_text()))):
    _download(f"models/terrain/{scene}/{mesh}", urdf.parent / mesh)
  return urdf


def main(
  output_dir: Path = DEFAULT_DIR,
  scenes: tuple[str, ...] = ("climb_24",),
  z_scales: tuple[str, ...] = ("1.0",),
) -> None:
  """Download the terrain archive and the obstacles for the selected scenes.

  Args:
    output_dir: Directory to cache everything under.
    scenes: Scenes to fetch obstacle models for. The archive holds every clip either way,
      so this only decides which boxes come with it.
    z_scales: Which height variants to fetch models for.
  """
  unknown = [scene for scene in scenes if scene not in CLIMB_SCENES]
  if unknown:
    raise ValueError(f"Unknown scenes: {', '.join(unknown)}")

  _download(TERRAIN_ZIP, output_dir / TERRAIN_ZIP)
  for scene in scenes:
    for z_scale in z_scales:
      fetch(clip_name(scene, z_scale), output_dir)
      fetch_terrain(scene, z_scale, output_dir)
      print(f"  {clip_name(scene, z_scale)} ready")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
