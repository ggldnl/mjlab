"""A dataset built from the trained skills' own rollouts.

Run:

    1. Train the skills. Any subset works; `skills` below selects which to record.

       uv run train Mjlab-G1-Walk --env.scene.num-envs 4096
       uv run train Mjlab-G1-Run --env.scene.num-envs 4096
       uv run train Mjlab-G1-Jump --env.scene.num-envs 4096

    2. Collect. Checkpoints are found under each skill's own log directory.

       uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.skills
       uv run python -m ...datasets.skills --skills "('walk','run')" --steps 800

    3. Calibrate the tolerances against it.

       uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.evaluate \
         --calibrate True --dataset data/bridge/rollouts.npz

    4. Train the bridge on it. This is the default dataset, so no flag is needed.

       uv run train Mjlab-G1-Bridge --env.scene.num-envs 4096

Every state here came out of a skill the bridge will be asked to serve, under the physics
that has to reproduce it. Both ends of a window are reachable by construction.

What it cannot answer: whether the bridge learned bridging or learned these five skills.
Training states and service states come from the same policies, so memorizing the pool
and generalizing look identical. tracker.py is the other half of that experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets import dataset
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.dataset import (
  DEFAULT_DATASET,
  RolloutCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.jump import JUMP_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.kick import KICK_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.push import PUSH_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.run import RUN_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.walk import WALK_TASK_ID
from mjlab.tasks.registry import load_env_cfg


@dataclass(frozen=True)
class SkillSpec:
  """One skill, and where its trained weights might be."""

  task: str
  experiments: tuple[str, ...]
  """Log directories to search, best first.

  Several names per skill because the experiments were renamed from parkour_* to g1_* and
  the log directories did not follow at the time. They match now, so the first entry hits;
  the rest cost nothing and keep an older copy of logs/ loadable."""


SKILLS: dict[str, SkillSpec] = {
  "walk": SkillSpec(WALK_TASK_ID, ("g1_walk", "parkour_walk")),
  "run": SkillSpec(RUN_TASK_ID, ("g1_run", "parkour_run")),
  "jump": SkillSpec(JUMP_TASK_ID, ("g1_jump", "parkour_jump_rsi", "humanoid_jump")),
  "kick": SkillSpec(KICK_TASK_ID, ("g1_kick", "parkour_kick")),
  "push": SkillSpec(PUSH_TASK_ID, ("g1_push", "parkour_push")),
}


@dataclass
class SkillsCfg(RolloutCfg):
  """How the skills dataset is collected."""

  skills: tuple[str, ...] = ("walk", "run", "jump")
  """Which skills to record. The default three need nothing on the floor. Kick and push
  work too, and are left out because a state whose meaning depends on where a crate was is
  not a useful thing for the bridge to aim at."""

  path: Path = DEFAULT_DATASET
  checkpoints: tuple[str, ...] = ()
  """Explicit checkpoint paths, one per entry of `skills`, for when the automatic search
  picks the wrong run. Empty means search."""


def collect(cfg: SkillsCfg) -> Path:
  """Record every configured skill and write one npz."""
  states: list[np.ndarray] = []
  env_ids: list[np.ndarray] = []
  frames: list[np.ndarray] = []
  sources: list[np.ndarray] = []
  goals: list[np.ndarray] = []
  fps = 0.0

  for index, name in enumerate(cfg.skills):
    if name not in SKILLS:
      raise SystemExit(f"Unknown skill '{name}'. Known: {', '.join(SKILLS)}.")
    spec = SKILLS[name]
    explicit = cfg.checkpoints[index] if index < len(cfg.checkpoints) else None

    env_cfg = load_env_cfg(spec.task)
    rate = dataset.control_rate(env_cfg)
    if fps and abs(rate - fps) > 1e-6:
      raise SystemExit(
        f"'{name}' runs at {rate:.1f} Hz and the skills before it at {fps:.1f} Hz. A dataset "
        f"mixing control rates has no single meaning for a deadline counted in steps."
      )
    fps = rate

    checkpoint = dataset.find_checkpoint(
      spec.experiments,
      explicit,
      hint=f" Train it with `uv run train {spec.task}`, or name one in `checkpoints`.",
    )
    rows, envs, ages, commands = dataset.record(
      spec.task, env_cfg, checkpoint, cfg, name
    )
    states.append(rows)
    env_ids.append(envs)
    frames.append(ages)
    sources.append(np.full(len(rows), index, dtype=np.int16))
    goals.append(commands)
    print(f"[dataset] {name}: {len(rows)} states, {commands.shape[1]} command numbers")

  return dataset.write(
    cfg.path, states, env_ids, frames, sources, cfg.skills, fps, goals
  )


if __name__ == "__main__":
  collect(tyro.cli(SkillsCfg, config=mjlab.TYRO_FLAGS))
