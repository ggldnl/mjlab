"""Collect the policy rollouts consumed by build.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

import mjlab
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset import dataset
from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import RolloutCfg
from mjlab.tasks.bridging.experiments.humanoid.selector import (
  ROLLOUTS_PATH,
  WINDOWS,
)
from mjlab.tasks.bridging.experiments.humanoid.skills import SKILLS
from mjlab.tasks.registry import load_env_cfg


@dataclass
class RecordCfg(RolloutCfg):
  """Skills, checkpoints, and output for rollout collection."""

  num_envs: int = 256
  steps: int = 800
  skills: tuple[str, ...] = tuple(WINDOWS)
  checkpoints: tuple[str, ...] = ()
  path: Path = ROLLOUTS_PATH


def collect(cfg: RecordCfg) -> Path:
  """Drive each skill and write one shared rollout file."""
  unknown = set(cfg.skills) - SKILLS.keys()
  if unknown:
    raise ValueError(f"Unknown skills: {', '.join(sorted(unknown))}")
  if cfg.checkpoints and len(cfg.checkpoints) != len(cfg.skills):
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

  for source, skill in enumerate(cfg.skills):
    task = SKILLS[skill]
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
    checkpoint = dataset.find_checkpoint(
      (f"g1_{skill}",),
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
    cfg.path,
    states,
    env_ids,
    trajectories,
    frames,
    sources,
    cfg.skills,
    fps,
    commands,
    phases,
    metadata,
  )


if __name__ == "__main__":
  collect(tyro.cli(RecordCfg, config=mjlab.TYRO_FLAGS))
