"""Robot-specific names and motion paths for the shared diffusion bridge."""

from __future__ import annotations

from pathlib import Path

from mjlab.tasks.bridging.config import get_robot


def robot_name(robot: str) -> str:
  return get_robot(robot).robot_name


def motion_patterns(robot: str, split: str) -> tuple[str, ...]:
  """Return every retargeted locomotion source for one robot and split."""
  if split not in ("train", "val"):
    raise ValueError("split must be train or val")
  selected = get_robot(robot)
  name = selected.robot_name
  return (
    str(
      Path("data")
      / "babel_retargeted"
      / selected.babel_dataset
      / split
      / "**"
      / "*.npz"
    ),
    str(Path("data") / "lafan_retargeted" / name / split / "**" / "*.npz"),
  )


def tracker_task_id(robot: str) -> str:
  get_robot(robot)
  return f"Mjlab-{robot.upper()}-Diffusion-Universal-Tracker"


def tracker_experiment(robot: str) -> str:
  get_robot(robot)
  return f"{robot}_diffusion_universal_tracker"


def planner_experiment(robot: str) -> str:
  get_robot(robot)
  return f"{robot}_kinematic_diffusion_planner"


def improvement_task_id(robot: str) -> str:
  get_robot(robot)
  return f"Mjlab-{robot.upper()}-Diffusion-Planner-Improvement"


def improvement_experiment(robot: str) -> str:
  get_robot(robot)
  return f"{robot}_diffusion_planner_improvement"
