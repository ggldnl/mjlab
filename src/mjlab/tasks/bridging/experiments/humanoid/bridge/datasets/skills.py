"""DEPRECATED. A corpus built from the trained skills' own rollouts.

Superseded by datasets/tracker.py, which is the default. This module still runs and the
datasets it wrote still load; it is kept so an older run can be reproduced and so a posture
family the human motion corpus is missing can be filled in deliberately. Nothing here is
maintained against new work on the bridge.

Why it was set aside:

  It cannot answer the question the bridge exists to answer. Every state here comes from a
  policy the bridge will be asked to serve, so training and service states come from the
  same five distributions and memorising the pool looks exactly like generalising. A bridge
  that only ever worked on this corpus would be evidence of nothing.

  It offers weaker feasibility guarantees per posture family. A skill rollout covers the
  states that skill occupies and no others, so coverage is whatever the pool happens to be,
  and it moves every time a skill is retrained.

  It made the bridge a function of the skill pool, which defeats the point. The whole
  premise is a policy independent of the skills it connects.

Run, if you still need it:

    1. Train the skills. Any subset works; `skills` below selects which to record.

       uv run train Mjlab-G1-Walk --env.scene.num-envs 4096
       uv run train Mjlab-G1-Run --env.scene.num-envs 4096
       uv run train Mjlab-G1-Jump --env.scene.num-envs 4096

    2. Collect. Checkpoints are found under each skill's own log directory.

       uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.skills

    3. Train the bridge against it, which now takes an explicit path.

       uv run train Mjlab-G1-Bridge --env.scene.num-envs 4096 \
         --env.commands.bridge.dataset-path data/bridge/rollouts.npz

Every state here did come out of a skill under the physics that has to reproduce it, so
both ends of a window are reachable by construction. That was always true and was never the
problem.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets import dataset
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.dataset import (
  SKILLS_DATASET,
  RolloutCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.jump import JUMP_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.passing import PASS_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.punch_combo import (
  PUNCH_COMBO_TASK_ID,
)
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
  "walk": SkillSpec(WALK_TASK_ID, ("g1_walk",)),
  "run": SkillSpec(RUN_TASK_ID, ("g1_run",)),
  "jump": SkillSpec(JUMP_TASK_ID, ("g1_jump",)),
  "pass": SkillSpec(PASS_TASK_ID, ("g1_pass",)),
  # "kick": SkillSpec(PASS_TASK_ID, ("g1_kick",)),
  "push": SkillSpec(PUSH_TASK_ID, ("g1_push",)),
  "punch_combo": SkillSpec(PUNCH_COMBO_TASK_ID, ("g1_punch_combo",)),
}


@dataclass
class SkillsCfg(RolloutCfg):
  """How the skills dataset is collected."""

  skills: tuple[str, ...] = ("walk", "run", "jump")
  """Which skills to record. The default three need nothing on the floor. Kick and push
  work too, and are left out because a state whose meaning depends on where a crate was is
  not a useful thing for the bridge to aim at."""

  path: Path = SKILLS_DATASET
  checkpoints: tuple[str, ...] = ()
  """Explicit checkpoint paths, one per entry of `skills`, for when the automatic search
  picks the wrong run. Empty means search."""


def collect(cfg: SkillsCfg) -> Path:
  """Record every configured skill and write one npz."""
  warnings.warn(
    "The skills corpus is deprecated. It cannot show whether a bridge learned bridging or "
    "learned these skills, since both its training and its service states come from the "
    "same policies. Prefer datasets/tracker.py, which is the default.",
    DeprecationWarning,
    stacklevel=2,
  )
  states: list[np.ndarray] = []
  env_ids: list[np.ndarray] = []
  trajectory_ids: list[np.ndarray] = []
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
    rows, envs, trajectories, ages, commands = dataset.record(
      spec.task, env_cfg, checkpoint, cfg, name
    )
    states.append(rows)
    env_ids.append(envs)
    trajectory_ids.append(trajectories)
    frames.append(ages)
    sources.append(np.full(len(rows), index, dtype=np.int16))
    goals.append(commands)
    print(f"[dataset] {name}: {len(rows)} states, {commands.shape[1]} command numbers")

  return dataset.write(
    cfg.path, states, env_ids, trajectory_ids, frames, sources, cfg.skills, fps, goals
  )


if __name__ == "__main__":
  collect(tyro.cli(SkillsCfg, config=mjlab.TYRO_FLAGS))
