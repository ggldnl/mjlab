"""Collect the policy rollouts consumed by build.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

import mjlab
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.bridging.bridges.dataset import dataset
from mjlab.tasks.bridging.bridges.dataset.dataset import RolloutCfg
from mjlab.tasks.bridging.config import get_robot
from mjlab.tasks.bridging.selector import (
  WINDOWS,
  paths,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


@dataclass
class RecordCfg(RolloutCfg):
  """Skills, checkpoints, and output for rollout collection."""

  robot: str = "g1"
  num_envs: int = 256
  steps: int = 800
  settle: int = 0
  skills: tuple[str, ...] = ()
  checkpoints: tuple[str, ...] = ()
  path: Path | None = None


def collect(cfg: RecordCfg) -> Path:
  """Drive each skill and write one shared rollout file."""
  available = get_robot(cfg.robot).skills
  skills = cfg.skills or tuple(name for name in WINDOWS if name in available)
  unknown = set(skills) - available.keys()
  if unknown:
    raise ValueError(f"Unknown skills: {', '.join(sorted(unknown))}")
  if cfg.checkpoints and len(cfg.checkpoints) != len(skills):
    raise ValueError("checkpoints must be empty or contain one path per skill")

  states: list[np.ndarray] = []
  env_ids: list[np.ndarray] = []
  trajectories: list[np.ndarray] = []
  frames: list[np.ndarray] = []
  phases: list[np.ndarray] = []
  sources: list[np.ndarray] = []
  commands: list[np.ndarray] = []
  metadata: list[dict[str, np.ndarray]] = []
  fps: float | None = None

  for source, skill in enumerate(skills):
    task = available[skill]
    env_cfg = load_env_cfg(task)
    if not any(
      sensor.name == "feet_ground_contact" for sensor in env_cfg.scene.sensors
    ):
      env_cfg.scene.sensors = (
        *env_cfg.scene.sensors,
        ContactSensorCfg(
          name="feet_ground_contact",
          primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
          ),
          secondary=ContactMatch(mode="body", pattern="terrain"),
          fields=("found",),
          reduce="netforce",
          num_slots=1,
        ),
      )
    rate = dataset.control_rate(env_cfg)
    if fps is not None and abs(rate - fps) > 1e-6:
      raise ValueError("All recorded skills must use the same control rate")
    fps = rate

    explicit = cfg.checkpoints[source] if cfg.checkpoints else None
    experiment = load_rl_cfg(task).experiment_name
    checkpoint = dataset.find_checkpoint(
      (experiment,),
      explicit,
      hint=f" Train it with `uv run train {task}`.",
    )
    context: dict[str, np.ndarray] = {}
    result = dataset.record(task, env_cfg, checkpoint, cfg, skill, metadata=context)
    if "foot_contact" not in context:
      raise ValueError(f"{skill} has no recorded feet_ground_contact sensor")
    state, env_id, trajectory, frame, phase, command = result
    states.append(state)
    env_ids.append(env_id)
    trajectories.append(trajectory)
    frames.append(frame)
    phases.append(phase)
    sources.append(np.full(len(state), source, dtype=np.int16))
    commands.append(command)
    metadata.append(context)

  if fps is None:
    raise ValueError("At least one skill is required")
  return dataset.write(
    cfg.path or paths(cfg.robot)[0],
    states,
    env_ids,
    trajectories,
    frames,
    sources,
    skills,
    fps,
    commands,
    phases,
    metadata,
  )


if __name__ == "__main__":
  collect(tyro.cli(RecordCfg, config=mjlab.TYRO_FLAGS))
