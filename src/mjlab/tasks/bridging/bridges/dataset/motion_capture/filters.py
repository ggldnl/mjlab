"""Every criterion that decides which motion capture reaches the bridge.

BABEL      AMASS takes kept by their action labels, cut to the labeled intervals
LAFAN      whole performances kept by the theme in their file name
kinematic  frames rejected after retargeting, marked invalid in each clip

Floor QA and grounding are part of retargeting, see mjlab.retargeting.floor.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mjlab.retargeting.floor import robot_foot_positions

# BABEL

# BMLrub and ACCAD are left out: many of their takes interact with objects or
# hold poses, and the labels do not always say so
BABEL_SUBSETS = ("CMU", "KIT")
BABEL_ALLOW = (
  "walk",
  "run",
  "jog",
  "turn",
  "step",
  # Lowered and propulsive states: the loader cuts long flights, so jump and leap
  # contribute their crouch, push off and landing, not the flight itself
  "crouch",
  "squat",
  "hop",
  "jump",
  "leap",
)
BABEL_CONTEXT = (
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
BABEL_DENY = (
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
  "lift",
  "carry",
  "catch",
  "throw",
  "knock",
  "wash",
  # Recovery from a shove the simulator never applies
  "push",
  "recovery",
  # Props and terrain the flat simulated floor does not have
  "parkour",
  "seesaw",
  "slope",
  "stair",
  "beam",
  "stone",
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


def _all_labels(record: dict[str, Any]) -> list[dict[str, Any]]:
  """Every frame and sequence label of a record, whether or not it is usable."""
  labels = []
  for key in ("frame_ann", "seq_ann"):
    annotation = record.get(key)
    if isinstance(annotation, dict) and isinstance(annotation.get("labels"), list):
      labels += [label for label in annotation["labels"] if isinstance(label, dict)]
  return labels


def _describe(label: dict[str, Any]) -> str:
  return " ".join(
    (
      str(label.get("raw_label", "")),
      str(label.get("proc_label", "")),
      *_categories(label),
    )
  )


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
    if _contains_denied_motion(_describe(label), denied):
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


def select_babel(
  annotations_dir: Path,
  subsets: tuple[str, ...] = BABEL_SUBSETS,
  allow: tuple[str, ...] = BABEL_ALLOW,
  context: tuple[str, ...] = BABEL_CONTEXT,
  deny: tuple[str, ...] = BABEL_DENY,
  min_duration_s: float = 1.0,
  allow_all: bool = False,
  whole_take: bool = True,
) -> list[dict[str, Any]]:
  """BABEL regions to retarget, from the labels under annotations_dir.

  Set allow_all to keep every labeled motion except denied intervals.
  With whole_take, one denied label anywhere drops the whole take: an object or a
  pose usually shapes the unlabeled time around it too. Without it, only the denied
  time is cut.
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
      if whole_take and any(
        _contains_denied_motion(_describe(label), denied)
        for label in _all_labels(raw_record)
      ):
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


# LAFAN

LAFAN_THEMES = ("walk", "run", "sprint", "jumps", "multipleActions")
KNOWN_THEMES = (
  "aiming",
  "dance",
  "fallAndGetUp",
  "fight",
  "fightAndSports",
  "ground",
  "jumps",
  "multipleActions",
  "obstacles",
  "push",
  "pushAndFall",
  "pushAndStumble",
  "run",
  "sprint",
  "walk",
)
_NAME = re.compile(
  r"^(?P<theme>[A-Za-z]+)(?P<take>\d+)_subject(?P<subject>\d+)$",
  re.IGNORECASE,
)


def select_lafan(
  input_dir: Path,
  themes: tuple[str, ...] = LAFAN_THEMES,
  validation_subject: int = 5,
) -> list[dict[str, Any]]:
  """Select broad locomotion themes and preserve LAFAN's subject split."""
  allowed = {theme.lower() for theme in themes}
  unknown = allowed - {theme.lower() for theme in KNOWN_THEMES}
  if unknown:
    raise ValueError(f"Unknown LAFAN themes {sorted(unknown)}; known: {KNOWN_THEMES}")
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


# Kinematic


@dataclass(frozen=True)
class MotionFilterCfg:
  """Optional limits applied to retargeted G1 kinematics."""

  min_root_up: float | None = None
  min_root_height: float | None = None
  max_root_xy_speed: float | None = None
  max_root_z_speed: float | None = None
  max_root_angular_speed: float | None = None
  max_airborne_foot_height: float | None = None
  max_foot_penetration: float | None = 0.06
  min_airborne_time_s: float = 0.04
  """Only flights at least this long are rejected, and then all of their frames."""


QUALITY_MOTION_FILTER = MotionFilterCfg(
  max_root_xy_speed=8.0,
  max_root_z_speed=4.0,
  max_root_angular_speed=12.0,
)
LOCOMOTION_MOTION_FILTER = MotionFilterCfg(
  min_root_up=0.7,
  min_root_height=0.4,
  max_root_xy_speed=2.5,
  max_root_z_speed=1.25,
  max_root_angular_speed=5.0,
  max_airborne_foot_height=0.05,
)
# Kinematic windows for the bridge: crouch, squat, hop, fast gaits and the
# accelerating and braking phases around a jump are all kept.
# Flights of 0.25 s or longer (jumps, leaps) are cut; running strides
# (p90 0.16 to 0.21 s) and hops or skips (p90 0.24 s) are not.
# Tilt stays: crouch and squat tilt at most about 27 deg, while past 45 deg are
# lying, getting up and retargeting glitches, and the tracker's fell_over uses 45 deg
BRIDGE_MOTION_FILTER = MotionFilterCfg(
  min_root_up=0.7,
  max_airborne_foot_height=0.05,
  min_airborne_time_s=0.25,
)
DEFAULT_MOTION_FILTER = BRIDGE_MOTION_FILTER
FilterProfile = Literal["bridge", "quality", "locomotion", "none"]


def motion_filter(profile: FilterProfile) -> MotionFilterCfg | None:
  """Return the named filter profile."""
  if profile == "bridge":
    return BRIDGE_MOTION_FILTER
  if profile == "quality":
    return QUALITY_MOTION_FILTER
  if profile == "locomotion":
    return LOCOMOTION_MOTION_FILTER
  if profile == "none":
    return None
  raise ValueError(f"Unknown motion filter profile: {profile}")


def _long_runs(mask: np.ndarray, length: int) -> np.ndarray:
  """Keep only the runs of True that last at least length frames, whole."""
  edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
  starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
  out = np.zeros(len(mask), dtype=bool)
  for start, end in zip(starts, ends, strict=True):
    if end - start >= length:
      out[start:end] = True
  return out


def motion_bad_frames(
  state: np.ndarray,
  body_pos: np.ndarray,
  body_quat: np.ndarray,
  fps: float,
  cfg: MotionFilterCfg = DEFAULT_MOTION_FILTER,
  robot: str = "unitree_g1",
) -> dict[str, np.ndarray]:
  """Mark bad frames once so every kinematic consumer uses the same cuts."""
  if fps <= 0 or cfg.min_airborne_time_s <= 0:
    raise ValueError("fps and airborne time must be positive")
  if len(state) != len(body_pos) or len(state) != len(body_quat):
    raise ValueError("state and body poses must have the same frame count")

  feet = robot_foot_positions(body_pos, body_quat, robot)
  floor = float(np.quantile(feet[..., 2].min(axis=1), 0.02))
  foot_height = feet[..., 2] - floor
  bad_frames: dict[str, np.ndarray] = {
    "nonfinite": np.asarray(~np.isfinite(state).all(axis=1), dtype=bool),
  }
  if cfg.min_root_up is not None:
    root_quat = state[:, 3:7]
    root_up = 1 - 2 * (root_quat[:, 1] ** 2 + root_quat[:, 2] ** 2)
    bad_frames["tilted"] = root_up < cfg.min_root_up
  if cfg.min_root_height is not None:
    bad_frames["low_root"] = state[:, 2] - floor < cfg.min_root_height
  if (
    cfg.max_root_xy_speed is not None
    or cfg.max_root_z_speed is not None
    or cfg.max_root_angular_speed is not None
  ):
    fast = np.zeros(len(state), dtype=bool)
    if cfg.max_root_xy_speed is not None:
      fast |= np.linalg.vector_norm(state[:, 7:9], axis=-1) > cfg.max_root_xy_speed
    if cfg.max_root_z_speed is not None:
      fast |= np.abs(state[:, 9]) > cfg.max_root_z_speed
    if cfg.max_root_angular_speed is not None:
      fast |= (
        np.linalg.vector_norm(state[:, 10:13], axis=-1) > cfg.max_root_angular_speed
      )
    bad_frames["fast_root"] = fast
  if cfg.max_airborne_foot_height is not None:
    event_frames = max(1, round(cfg.min_airborne_time_s * fps))
    airborne = foot_height.min(axis=1) > cfg.max_airborne_foot_height
    bad_frames["airborne"] = _long_runs(airborne, event_frames)
  if cfg.max_foot_penetration is not None:
    bad_frames["penetration"] = foot_height.min(axis=1) < -cfg.max_foot_penetration
  return bad_frames


def filter_window_starts(
  state: np.ndarray,
  body_pos: np.ndarray,
  body_quat: np.ndarray,
  columns: int,
  fps: float,
  cfg: MotionFilterCfg = DEFAULT_MOTION_FILTER,
  robot: str = "unitree_g1",
) -> tuple[np.ndarray, dict[str, int]]:
  """Return window starts accepted by the selected kinematic limits."""
  if columns < 2:
    raise ValueError("window length must exceed one")
  if len(state) < columns:
    return np.empty(0, dtype=np.int64), {}

  bad_frames = motion_bad_frames(state, body_pos, body_quat, fps, cfg, robot)
  rejected = {}
  for name, bad in bad_frames.items():
    counts = np.concatenate(([0], np.cumsum(bad, dtype=np.int64)))
    rejected[name] = (counts[columns:] - counts[:-columns]) > 0
  valid = ~np.logical_or.reduce(tuple(rejected.values()))
  return np.flatnonzero(valid), {
    name: int(mask.sum()) for name, mask in rejected.items()
  }
