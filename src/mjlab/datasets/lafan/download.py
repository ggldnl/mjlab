"""Download and extract the original Ubisoft LAFAN BVH archive.

Run:

  uv run python -m mjlab.datasets.lafan.download
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

import tyro

import mjlab

ARCHIVE_URL = (
  "https://github.com/ubisoft/ubisoft-laforge-animation-dataset/"
  "raw/master/lafan1/lafan1.zip"
)
DEFAULT_DIR = Path("data") / "lafan"


def extract(archive: Path, output_dir: Path, overwrite: bool = False) -> int:
  """Extract BVH files without trusting paths stored in the archive."""
  count = 0
  with zipfile.ZipFile(archive) as source:
    for member in source.infolist():
      if member.is_dir() or Path(member.filename).suffix.lower() != ".bvh":
        continue
      destination = output_dir / "raw" / Path(member.filename).name
      if destination.exists() and not overwrite:
        continue
      destination.parent.mkdir(parents=True, exist_ok=True)
      with source.open(member) as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst)
      count += 1
  return count


def main(
  output_dir: Path = DEFAULT_DIR,
  url: str = ARCHIVE_URL,
  overwrite: bool = False,
) -> None:
  """Download the official archive and extract its BVH performances."""
  archive = output_dir / "lafan1.zip"
  if overwrite or not archive.is_file():
    output_dir.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(".zip.part")
    try:
      with urllib.request.urlopen(url) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
      partial.replace(archive)
    except Exception as error:
      partial.unlink(missing_ok=True)
      raise RuntimeError(f"Could not download LAFAN from {url}: {error}") from error
  extracted = extract(archive, output_dir, overwrite)
  print(f"LAFAN: extracted {extracted} BVH files under {output_dir / 'raw'}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
