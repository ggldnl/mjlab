"""Select original LAFAN locomotion performances by filename theme.

LAFAN has no frame-level action labels. Its filenames carry only a broad theme,
so the manifest keeps walk, run, and sprint takes and leaves frame-level quality
checks to the shared post-retargeting kinematic filter.

Run:

  uv run python -m mjlab.datasets.lafan.build_manifest
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import tyro

import mjlab

DEFAULT_THEMES = ("walk", "run", "sprint")
_NAME = re.compile(
  r"^(?P<theme>[A-Za-z]+)(?P<take>\d+)_subject(?P<subject>\d+)$",
  re.IGNORECASE,
)


def build_manifest(
  input_dir: Path,
  themes: tuple[str, ...] = DEFAULT_THEMES,
  validation_subject: int = 5,
) -> list[dict[str, Any]]:
  """Select broad locomotion themes and preserve LAFAN's subject split."""
  allowed = {theme.lower() for theme in themes}
  entries = []
  for path in sorted(input_dir.rglob("*.bvh")):
    match = _NAME.fullmatch(path.stem)
    if match is None:
      continue
    theme = match.group("theme").lower()
    if theme not in allowed:
      continue
    subject = int(match.group("subject"))
    entries.append(
      {
        "source": path.relative_to(input_dir).as_posix(),
        "split": "val" if subject == validation_subject else "train",
        "category": theme,
        "take": int(match.group("take")),
        "subject": subject,
        "fps": 30.0,
      }
    )
  if not entries:
    raise ValueError(f"No LAFAN {sorted(allowed)} BVH files found under {input_dir}")
  return entries


def read_manifest(path: Path) -> list[dict[str, Any]]:
  entries = []
  for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
      continue
    entry = json.loads(line)
    if not isinstance(entry, dict):
      raise ValueError(f"Manifest line {line_number} is not an object: {path}")
    entries.append(entry)
  return entries


def main(
  input_dir: Path = Path("data/lafan/raw"),
  output_path: Path = Path("data/lafan/manifest.jsonl"),
  themes: tuple[str, ...] = DEFAULT_THEMES,
  validation_subject: int = 5,
) -> None:
  """Write the selected LAFAN source manifest."""
  entries = build_manifest(input_dir, themes, validation_subject)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(
    "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries),
    encoding="utf-8",
  )
  counts = Counter(entry["category"] for entry in entries)
  splits = Counter(entry["split"] for entry in entries)
  print(f"Selected {len(entries)} LAFAN clips: {dict(counts)}; {dict(splits)}")
  print(f"Manifest: {output_path}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
