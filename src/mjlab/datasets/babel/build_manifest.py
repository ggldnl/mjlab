"""Select contiguous safe AMASS regions from BABEL labels.

Download BABEL v1.0 from https://babel.is.tue.mpg.de and extract its JSON files
under ``data/babel/babel_v1.0_release``.

Run:

  uv run python -m mjlab.datasets.babel.build_manifest
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import tyro

import mjlab

DEFAULT_SUBSETS = ("BMLrub", "CMU", "KIT", "ACCAD")
DEFAULT_ALLOW = (
  "walk",
  "run",
  "jog",
  "turn",
  "step",
)
DEFAULT_CONTEXT = (
  "transition",
  "stand",
  "stop",
  "forward movement",
  "backwards movement",
  "sideways movement",
  "circular movement",
  "face direction",
  "look",
  "arm movements",
  "hand movements",
  "head movements",
  "leg movements",
  "foot movements",
  "feet movements",
  "waist movements",
)
DEFAULT_DENY = (
  "unknown",
  "noisy labels",
  "misc. action",
  "misc. activities",
  "misc. abstract action",
  "interact with/use object",
  "interact with object",
  "use object",
  "touch object",
  "take/pick something up",
  "pick something up",
  "place something",
  "grasp object",
  "move something",
  "lift something",
  "clean something",
  "open something",
  "close something",
  "press something",
  "give something",
  "release something",
  "object",
  "chair",
  "couch",
  "bench",
  "table",
  "shelf",
  "box",
  "ball",
  "tool",
  "rope",
  "rail",
  "wall",
  "support",
  "sit",
  "lie",
  "kneel",
  "crouch",
  "squat",
  "lunge",
  "yoga",
  "exercise",
  "training",
  "sport",
  "pose",
  "t pose",
  "a pose",
  "balance",
  "stretch",
  "bend",
  "lean",
  "jump",
  "hop",
  "leap",
  "cartwheel",
  "flip",
  "headstand",
  "handstand",
  "fight",
  "martial art",
  "kick",
  "punch",
  "hit",
  "fall",
  "crawl",
  "dance",
  "trip",
  "stumble",
  "get injured",
)
SPLITS = ("train", "val", "test")
BABEL_SUBSET_NAMES = {
  "BMLrub": "BioMotionLab_NTroje",
  "DFaust67": "DFaust_67",
  "EyesJapanDataset": "Eyes_Japan_Dataset",
  "MPIHDM05": "MPI_HDM05",
  "MPILimits": "MPI_Limits",
  "MPImosh": "MPI_mosh",
  "SSMsynced": "SSM_synced",
  "TCDhandMocap": "TCD_handMocap",
  "Transitionsmocap": "Transitions_mocap",
}


def _normalized(values: tuple[str, ...]) -> set[str]:
  return {value.strip().lower() for value in values if value.strip()}


def _categories(label: dict[str, Any]) -> tuple[str, ...]:
  raw = label.get("act_cat")
  values = raw if isinstance(raw, list) else []
  categories = {str(value).strip().lower() for value in values if str(value).strip()}
  if not categories and label.get("proc_label"):
    categories.add(str(label["proc_label"]).strip().lower())
  return tuple(sorted(categories))


def _contains_denied_motion(text: str, denied: set[str]) -> bool:
  normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
  for term in denied:
    normalized_term = re.sub(r"[^a-z0-9]+", " ", term).strip()
    pattern = re.escape(normalized_term).replace(r"\ ", r"\s+")
    if re.search(rf"\b{pattern}\w*\b", normalized):
      return True
  return False


def _labels(record: dict[str, Any]) -> list[tuple[dict[str, Any], float, float]]:
  duration = float(record.get("dur", 0.0))
  frame_ann = record.get("frame_ann")
  if isinstance(frame_ann, dict) and isinstance(frame_ann.get("labels"), list):
    labels = []
    for label in frame_ann["labels"]:
      if not isinstance(label, dict):
        continue
      start = max(0.0, float(label.get("start_t", 0.0)))
      end = min(duration, float(label.get("end_t", duration)))
      if end > start:
        labels.append((label, start, end))
    return labels

  seq_ann = record.get("seq_ann")
  if not isinstance(seq_ann, dict) or seq_ann.get("mul_act"):
    return []
  labels = seq_ann.get("labels")
  if duration <= 0 or not isinstance(labels, list):
    return []
  return [(label, 0.0, duration) for label in labels if isinstance(label, dict)]


def _safe_regions(
  labels: list[tuple[dict[str, Any], float, float]],
  allowed: set[str] | None,
  context: set[str],
  denied: set[str],
  min_duration_s: float,
) -> list[tuple[float, float, list[dict[str, Any]]]]:
  """Merge locomotion intervals after removing blocked and unknown motion."""
  safe = []
  blocked = []
  for label, start, end in labels:
    categories = set(_categories(label))
    description = " ".join(
      (
        str(label.get("raw_label", "")),
        str(label.get("proc_label", "")),
        *categories,
      )
    )
    if _contains_denied_motion(description, denied):
      blocked.append((start, end))
    elif allowed is None:
      safe.append((label, start, end))
    elif not allowed.isdisjoint(categories) and categories.issubset(allowed | context):
      safe.append((label, start, end))
    elif not categories or not categories.issubset(context):
      blocked.append((start, end))

  merged: list[list[float]] = []
  for _, start, end in sorted(safe, key=lambda item: (item[1], item[2])):
    if merged and start <= merged[-1][1] + 1e-9:
      merged[-1][1] = max(merged[-1][1], end)
    else:
      merged.append([start, end])

  regions = []
  for start, end in merged:
    pieces = [(start, end)]
    for blocked_start, blocked_end in blocked:
      next_pieces = []
      for piece_start, piece_end in pieces:
        if blocked_end <= piece_start or blocked_start >= piece_end:
          next_pieces.append((piece_start, piece_end))
          continue
        if piece_start < blocked_start:
          next_pieces.append((piece_start, blocked_start))
        if blocked_end < piece_end:
          next_pieces.append((blocked_end, piece_end))
      pieces = next_pieces
    for piece_start, piece_end in pieces:
      if piece_end - piece_start < min_duration_s:
        continue
      contributors = [
        label
        for label, label_start, label_end in safe
        if label_end > piece_start and label_start < piece_end
      ]
      regions.append((piece_start, piece_end, contributors))
  return regions


def build_manifest(
  annotations_dir: Path,
  subsets: tuple[str, ...] = DEFAULT_SUBSETS,
  allow: tuple[str, ...] = DEFAULT_ALLOW,
  context: tuple[str, ...] = DEFAULT_CONTEXT,
  deny: tuple[str, ...] = DEFAULT_DENY,
  min_duration_s: float = 1.0,
  allow_all: bool = False,
) -> list[dict[str, Any]]:
  """Return maximal safe regions from dense or unambiguous sequence labels.

  Set allow_all to keep every labeled motion except denied intervals.
  """
  if min_duration_s < 0:
    raise ValueError("min_duration_s must be nonnegative")
  selected_subsets = {BABEL_SUBSET_NAMES.get(value, value) for value in subsets}
  allowed = None if allow_all else _normalized(allow)
  safe_context = _normalized(context)
  denied = _normalized(deny)
  if allowed is not None and not allowed:
    raise ValueError("allow must contain at least one category")

  entries: list[dict[str, Any]] = []
  for split in SPLITS:
    path = annotations_dir / f"{split}.json"
    if not path.is_file():
      raise FileNotFoundError(f"Missing BABEL split: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, dict):
      raise ValueError(f"BABEL split must contain a JSON object: {path}")

    for sid, raw_record in records.items():
      if not isinstance(raw_record, dict):
        continue
      source = str(raw_record.get("feat_p", "")).replace("\\", "/")
      raw_subset, separator, relative = source.partition("/")
      subset = BABEL_SUBSET_NAMES.get(raw_subset, raw_subset)
      if subset not in selected_subsets:
        continue
      if subset != raw_subset:
        source = subset + separator + relative
      labels = _labels(raw_record)
      source_description = re.sub(r"_poses(?=\.npz$)", "", source)
      if _contains_denied_motion(source_description, denied):
        continue
      for index, (start, end, contributors) in enumerate(
        _safe_regions(labels, allowed, safe_context, denied, min_duration_s)
      ):
        categories = sorted(
          {category for label in contributors for category in _categories(label)}
        )
        descriptions = list(
          dict.fromkeys(
            str(label.get("proc_label") or label.get("raw_label") or "")
            for label in contributors
          )
        )
        segment_ids = [
          str(label["seg_id"])
          for label in contributors
          if label.get("seg_id") is not None
        ]
        entries.append(
          {
            "babel_sid": raw_record.get("babel_sid", sid),
            "segment_id": f"safe_{index:04d}",
            "segment_ids": segment_ids,
            "split": split,
            "subset": subset,
            "source": source,
            "start_s": start,
            "end_s": end,
            "categories": categories,
            "label": " | ".join(filter(None, descriptions)),
            "labels": descriptions,
          }
        )

  return sorted(
    entries,
    key=lambda entry: (
      str(entry["split"]),
      str(entry["source"]),
      float(entry["start_s"]),
      float(entry["end_s"]),
    ),
  )


def write_manifest(entries: list[dict[str, Any]], output_path: Path) -> None:
  """Write one selected BABEL segment per JSON line."""
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(
    "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries),
    encoding="utf-8",
  )


def read_manifest(path: Path) -> list[dict[str, Any]]:
  """Read a JSONL manifest produced by this module."""
  entries = []
  for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
      continue
    entry = json.loads(line)
    if not isinstance(entry, dict):
      raise ValueError(f"Manifest line {line_number} is not a JSON object: {path}")
    entries.append(entry)
  return entries


def _print_summary(entries: list[dict[str, Any]], output_path: Path) -> None:
  sources = {str(entry["source"]) for entry in entries}
  seconds_by_subset: defaultdict[str, float] = defaultdict(float)
  category_counts: Counter[str] = Counter()
  for entry in entries:
    duration = float(entry["end_s"]) - float(entry["start_s"])
    seconds_by_subset[str(entry["subset"])] += duration
    category_counts.update(str(category) for category in entry["categories"])

  print(
    f"Selected {len(entries)} safe regions from {len(sources)} AMASS files "
    f"({sum(seconds_by_subset.values()) / 3600:.2f} h)"
  )
  for subset, seconds in sorted(seconds_by_subset.items()):
    print(f"  {subset}: {seconds / 60:.1f} min")
  print(
    "  categories: " + ", ".join(f"{k}={v}" for k, v in category_counts.most_common())
  )
  print(f"Manifest: {output_path}")


def main(
  annotations_dir: Path = Path("data/babel/babel_v1.0_release"),
  output_path: Path = Path("data/babel/manifest.jsonl"),
  subsets: tuple[str, ...] = DEFAULT_SUBSETS,
  allow: tuple[str, ...] = DEFAULT_ALLOW,
  context: tuple[str, ...] = DEFAULT_CONTEXT,
  deny: tuple[str, ...] = DEFAULT_DENY,
  min_duration_s: float = 1.0,
  allow_all: bool = False,
) -> None:
  """Build a BABEL manifest without downloading or retargeting motion files."""
  entries = build_manifest(
    annotations_dir, subsets, allow, context, deny, min_duration_s, allow_all
  )
  write_manifest(entries, output_path)
  _print_summary(entries, output_path)


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
