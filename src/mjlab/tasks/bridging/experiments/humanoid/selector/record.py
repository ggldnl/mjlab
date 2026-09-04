"""Record each skill's rollouts. The input build.py reads.

First step of the pipeline. Drives one trained policy per skill and writes down every
control step: one npz, one source per skill, in the layout bridge/datasets/dataset.py
defines. Needs a trained checkpoint per skill, found under logs/rsl_rl/g1_<skill>, and the
path picked is printed before each recording starts.

Separate from bridge/datasets/, which records for a different purpose: the bridge wants
states to cross between, this wants the states one skill occupies. Same file layout, so
bridge/datasets/view.py can play these back, and no shared entry point.

Training configs, not play configs. Play narrows command ranges and drops the noise the
policy trained under, which gives a tidier demo and a narrower set of states. Entry points
should cover what a skill does in service, edges of its command range included.

The first settle steps after every reset are dropped, so progress 0 is half a second into
an episode rather than its first frame. Without that the dataset fills with one identical
standing pose per failure: mjlab resets the instant an environment terminates, and the step
after a fall is a robot at its default pose.

Run

1. Record the default skills.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.record

2. Choose which skills, or how much of each.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.record --skills "('walk','jump','push')"
    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.record --num-envs 128 --steps 800

3. Then find the candidates.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.build
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
from mjlab.tasks.bridging.experiments.humanoid.skills.front_kick import (
  FRONT_KICK_TASK_ID,
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

SKILLS: dict[str, str] = {
  "walk": WALK_TASK_ID,
  "run": RUN_TASK_ID,
  "jump": JUMP_TASK_ID,
  "front_kick": FRONT_KICK_TASK_ID,
  "pass": PASS_TASK_ID,
  "push": PUSH_TASK_ID,
  "punch_combo": PUNCH_COMBO_TASK_ID,
}
"""Skill name to task id. Every one logs to g1_<name>, so nothing else has to be said.

Add a skill by adding a line. It needs a trained checkpoint under that log directory and
nothing else: build.py does not know what any of these are."""


def experiment(skill: str) -> str:
  """Log directory holding that skill's checkpoints."""
  return f"g1_{skill}"


@dataclass
class RecordCfg(RolloutCfg):
  """How much of each skill to record."""

  num_envs: int = 256
  """More than the shared default. Each environment draws its own command, so this is how
  many command values the recording contains, and a node has to be visited under all of
  them to look reliable rather than rare."""

  skills: tuple[str, ...] = ("walk", "run", "jump")
  """Which skills to record. The default three need nothing on the floor.

  Front kick, push and pass work too. Their entry states mean less on their own, since
  where the ball or the crate was is part of what the skill was doing, and none of that
  is in a robot state."""

  path: Path = ROLLOUTS_PATH

  checkpoints: tuple[str, ...] = ()
  """Explicit checkpoint paths, one per entry of skills, for when the newest run under
  a log directory is not the one meant. Empty means search."""


def spread_of(
  commands: np.ndarray, trajectories: np.ndarray, detail: int = 6
) -> list[str]:
  """How much of each command's range this recording actually covered.

  A node is a moment the skill goes through whatever it was asked for, so the recording
  has to contain every "whatever". Random sampling per environment usually gets there,
  and this is how you see that it did rather than assume it: `distinct` well below the
  rollout count means most episodes drew the same command and the corpus is narrower
  than the skill.

  A tracking skill's command is mostly the reference it is chasing, which is dozens of
  numbers changing every frame, so only the widest `detail` of them are printed. The
  count on the first line is the part worth reading.
  """
  if commands.size == 0:
    return ["command: none"]
  first = np.array(
    [commands[trajectories == t][0] for t in np.unique(trajectories)], dtype=np.float64
  )
  distinct = len(np.unique(first.round(3), axis=0))
  low, high = commands.min(axis=0), commands.max(axis=0)
  span = high - low
  moving = np.nonzero(span > 1e-9)[0]
  out = [
    f"command: {distinct} distinct over {len(first)} rollouts, "
    f"{moving.size} of {commands.shape[1]} values vary"
  ]
  widest = moving[np.argsort(-span[moving])][:detail]
  out += [f"  [{i}] {low[i]:+.2f} to {high[i]:+.2f}" for i in sorted(widest)]
  if moving.size > detail:
    out.append(f"  and {moving.size - detail} more")
  return out


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
    for line in spread_of(commands, trajectories):
      print(f"[selector]   {line}")

  return dataset.write(
    cfg.path, states, env_ids, trajectory_ids, frames, sources, cfg.skills, fps, goals
  )


if __name__ == "__main__":
  collect(tyro.cli(RecordCfg, config=mjlab.TYRO_FLAGS))
