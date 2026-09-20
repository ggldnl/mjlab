"""Load the clip and tracker checkpoint pairs used for CVAE training."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.tracker import (
  _trained_on,
)

DEFAULT_MANIFEST = Path("configs/cvae_teachers.yaml")


@dataclass(frozen=True)
class Teacher:
  source: str
  motion: Path
  checkpoint: Path


def load_teachers(
  path: Path = DEFAULT_MANIFEST, validate: bool = False
) -> tuple[Teacher, ...]:
  if not path.is_file():
    raise FileNotFoundError(f"No CVAE teacher manifest at {path}")
  raw = yaml.safe_load(path.read_text(encoding="utf-8"))
  entries = raw.get("teachers") if isinstance(raw, dict) else None
  if not isinstance(entries, list) or not entries:
    raise ValueError(f"{path} needs a nonempty teachers list")
  teachers = []
  for entry in entries:
    if not isinstance(entry, dict) or set(entry) != {"source", "motion", "checkpoint"}:
      raise ValueError(f"Each teacher in {path} needs source, motion, checkpoint")
    teacher = Teacher(
      source=str(entry["source"]),
      motion=Path(entry["motion"]),
      checkpoint=Path(entry["checkpoint"]),
    )
    if teacher.motion.stem != teacher.source:
      raise ValueError(f"{teacher.source} does not match {teacher.motion}")
    if teacher.source in {item.source for item in teachers}:
      raise ValueError(f"Duplicate teacher source {teacher.source}")
    if validate:
      if not teacher.motion.is_file() or not teacher.checkpoint.is_file():
        raise FileNotFoundError(f"Missing motion or checkpoint for {teacher.source}")
      trained_on = _trained_on(teacher.checkpoint.parent)
      if trained_on is None or trained_on.resolve() != teacher.motion.resolve():
        raise ValueError(f"{teacher.checkpoint} was not trained on {teacher.motion}")
    teachers.append(teacher)
  return tuple(teachers)
