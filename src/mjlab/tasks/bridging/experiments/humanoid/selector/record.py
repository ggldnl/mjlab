"""Record each skill's rollouts. The input build.py reads.

Drives one trained policy per skill and writes down every control step. One npz, one
source per skill, in the layout bridge/datasets/dataset.py defines.

Needs a trained checkpoint per skill. They are found under logs/rsl_rl/g1_<skill>, and
the path picked is printed before each recording starts.

Run
---

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.record
    uv run python -m ...selector.record --skills "('walk','jump','kick')"
    uv run python -m ...selector.record --num-envs 128 --steps 800

Then build the table:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.build

Notes
-----

Separate from bridge/datasets/, which records for a different purpose: the bridge wants
states to cross between, this wants the states one skill occupies. Same file layout, so
bridge/datasets/view.py can play these back, and no shared entry point.

Training configs, not play configs. Play narrows command ranges and drops the noise the
policy trained under, which gives a tidier demo and a narrower set of states. Entry
points should cover what a skill does in service, edges of its command range included.

The first `settle` steps after every reset are dropped, so progress 0 is half a second
into an episode rather than its first frame. Without that the dataset fills with one
identical standing pose per failure: mjlab resets the instant an environment terminates,
and the step after a fall is a robot at its default pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets import dataset
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.dataset import RolloutCfg
from mjlab.tasks.bridging.experiments.humanoid.selector.table import ROLLOUTS_PATH
from mjlab.tasks.bridging.experiments.humanoid.skills.backflip import BACKFLIP_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.front_kick import (
  FRONT_KICK_TASK_ID,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.jump import JUMP_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.kick import KICK_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.passing import PASS_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.punch_combo import (
  PUNCH_COMBO_TASK_ID,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.push import PUSH_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.run import RUN_TASK_ID
from mjlab.tasks.bridging.experiments.humanoid.skills.walk import WALK_TASK_ID
from mjlab.tasks.registry import load_env_cfg

SKILLS: dict[str, str] = {
  "walk": WALK_TASK_ID,
  "run": RUN_TASK_ID,
  "jump": JUMP_TASK_ID,
  "kick": KICK_TASK_ID,
  "front_kick": FRONT_KICK_TASK_ID,
  "pass": PASS_TASK_ID,
  "push": PUSH_TASK_ID,
  "punch_combo": PUNCH_COMBO_TASK_ID,
  "backflip": BACKFLIP_TASK_ID,
}
"""Skill name to task id. Every one logs to g1_<name>, so nothing else has to be said."""


def experiment(skill: str) -> str:
  """Log directory holding that skill's checkpoints."""
  return f"g1_{skill}"


@dataclass
class RecordCfg(RolloutCfg):
  """How much of each skill to record."""

  skills: tuple[str, ...] = ("walk", "run", "jump")
  """Which skills to record. The default three need nothing on the floor.

  Kick, push and pass work too. Their entry states mean less on their own, since where
  the ball or the crate was is part of what the skill was doing and none of that is in
  a robot state."""

  path: Path = ROLLOUTS_PATH

  checkpoints: tuple[str, ...] = ()
  """Explicit checkpoint paths, one per entry of `skills`, for when the newest run under
  a log directory is not the one meant. Empty means search."""


def collect(cfg: RecordCfg) -> Path:
  """Record every configured skill and write one npz."""
  unknown = [name for name in cfg.skills if name not in SKILLS]
  if unknown:
    raise SystemExit(
      f"Unknown skills {', '.join(unknown)}. Known: {', '.join(sorted(SKILLS))}."
    )

  states: list[np.ndarray] = []
  env_ids: list[np.ndarray] = []
  trajectory_ids: list[np.ndarray] = []
  frames: list[np.ndarray] = []
  sources: list[np.ndarray] = []
  goals: list[np.ndarray] = []
  fps = 0.0

  for index, name in enumerate(cfg.skills):
    task = SKILLS[name]
    env_cfg = load_env_cfg(task)

    # A frame count only means a duration if every source ran at the same rate, and
    # dwell_s divides by one number for the whole table
    rate = dataset.control_rate(env_cfg)
    if fps and abs(rate - fps) > 1e-6:
      raise SystemExit(
        f"'{name}' runs at {rate:.1f} Hz and the skills before it at {fps:.1f} Hz. "
        f"One dataset cannot hold both: dwell is counted in steps."
      )
    fps = rate

    checkpoint = dataset.find_checkpoint(
      (experiment(name),),
      cfg.checkpoints[index] if index < len(cfg.checkpoints) else None,
      hint=f" Train it with `uv run train {task}`, or name one in `checkpoints`.",
    )
    rows, envs, trajectories, ages, commands = dataset.record(
      task, env_cfg, checkpoint, cfg, name
    )
    states.append(rows)
    env_ids.append(envs)
    trajectory_ids.append(trajectories)
    frames.append(ages)
    sources.append(np.full(len(rows), index, dtype=np.int16))
    goals.append(commands)
    rollouts = len(np.unique(trajectories))
    print(f"[selector] {name}: {len(rows)} states over {rollouts} rollouts")

  return dataset.write(
    cfg.path, states, env_ids, trajectory_ids, frames, sources, cfg.skills, fps, goals
  )


if __name__ == "__main__":
  collect(tyro.cli(RecordCfg, config=mjlab.TYRO_FLAGS))
