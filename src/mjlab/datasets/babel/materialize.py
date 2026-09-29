"""Extract the AMASS files selected by a BABEL manifest.

Run:

  uv run python -m mjlab.datasets.babel.materialize
"""

from __future__ import annotations

import shutil
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import tyro

import mjlab
from mjlab.datasets.amass.download import ARCHIVE_NAMES
from mjlab.datasets.babel.build_manifest import read_manifest


def source_path(source: str) -> Path:
  """Convert a safe BABEL source path to a local relative path."""
  path = PurePosixPath(source.replace("\\", "/"))
  if path.is_absolute() or not path.parts or ".." in path.parts:
    raise ValueError(f"Unsafe BABEL source path: {source!r}")
  return Path(*path.parts)


def _member_candidates(source: str, subset: str) -> tuple[str, ...]:
  path = PurePosixPath(source.replace("\\", "/"))
  relative = PurePosixPath(*path.parts[1:]) if len(path.parts) > 1 else path
  archive = ARCHIVE_NAMES.get(subset, subset)
  variants: set[str] = {path.as_posix(), relative.as_posix()}
  if relative.parts:
    variants.add(PurePosixPath(archive, *relative.parts[1:]).as_posix())
  variants |= {value.replace("_poses.npz", "_stageii.npz") for value in variants}
  variants |= {value.replace(" ", "_") for value in variants}
  return tuple(sorted(variants))


def _member_index(
  members: list[tarfile.TarInfo],
) -> dict[str, set[tarfile.TarInfo]]:
  """Index every path suffix once for fast manifest lookup."""
  index: defaultdict[str, set[tarfile.TarInfo]] = defaultdict(set)
  for member in members:
    if not member.isfile():
      continue
    parts = PurePosixPath(member.name.replace("\\", "/").lstrip("./")).parts
    for start in range(len(parts)):
      index[PurePosixPath(*parts[start:]).as_posix()].add(member)
  return dict(index)


def _find_member(
  members: dict[str, set[tarfile.TarInfo]], source: str, subset: str
) -> tarfile.TarInfo:
  candidates = _member_candidates(source, subset)
  matches = {
    member for candidate in candidates for member in members.get(candidate, ())
  }
  if len(matches) != 1:
    detail = "not found" if not matches else f"matched {len(matches)} archive members"
    raise FileNotFoundError(f"BABEL source {source!r} {detail} in the {subset} archive")
  return matches.pop()


def materialize(
  manifest_path: Path,
  archive_dir: Path,
  output_dir: Path,
  overwrite: bool = False,
) -> tuple[int, int]:
  """Extract unique manifest sources. Returns extracted and skipped counts."""
  sources_by_subset: defaultdict[str, set[str]] = defaultdict(set)
  for entry in read_manifest(manifest_path):
    source = str(entry.get("source", ""))
    subset = str(entry.get("subset", ""))
    path = source_path(source)
    if not subset or path.parts[0] != subset:
      raise ValueError(f"Manifest subset does not match source: {entry}")
    sources_by_subset[subset].add(source)

  extracted = skipped = 0
  archive_paths: dict[str, Path] = {}
  for subset in sources_by_subset:
    archive_name = ARCHIVE_NAMES.get(subset, subset)
    archives = (
      archive_dir / f"{subset}.tar.bz2",
      archive_dir / f"{archive_name}.tar.bz2",
    )
    archive_path = next((path for path in archives if path.is_file()), None)
    if archive_path is None:
      raise FileNotFoundError(
        f"Missing AMASS archive for {subset}; expected one of: "
        + ", ".join(str(path) for path in archives)
      )
    archive_paths[subset] = archive_path

  for subset, sources in sorted(sources_by_subset.items()):
    archive_path = archive_paths[subset]
    with tarfile.open(archive_path, "r:bz2") as archive:
      members = _member_index(archive.getmembers())
      pending: list[tuple[tarfile.TarInfo, Path]] = []
      for source in sorted(sources):
        destination = output_dir / source_path(source)
        if destination.is_file() and not overwrite:
          skipped += 1
          continue
        member = _find_member(members, source, subset)
        pending.append((member, destination))
      for member, destination in sorted(pending, key=lambda item: item[0].offset_data):
        stream = archive.extractfile(member)
        if stream is None:
          raise FileNotFoundError(f"Cannot read {member.name} from {archive_path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        with stream, temporary.open("wb") as output:
          shutil.copyfileobj(stream, output)
        temporary.replace(destination)
        extracted += 1

  return extracted, skipped


def main(
  manifest_path: Path = Path("data/babel/manifest.jsonl"),
  archive_dir: Path = Path("data/amass_atomic/tarballs"),
  output_dir: Path = Path("data/amass_babel"),
  overwrite: bool = False,
) -> None:
  """Extract manifest sources from retained AMASS tarballs."""
  extracted, skipped = materialize(manifest_path, archive_dir, output_dir, overwrite)
  print(f"Materialized {extracted} AMASS files in {output_dir}; skipped {skipped}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
