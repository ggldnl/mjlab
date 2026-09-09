"""Record each skill's rollouts. The input build.py reads.

First step of the pipeline. Drives one trained policy per skill and writes down every
control step: one npz, one source per skill, in the layout bridge/datasets/dataset.py
defines. Needs a trained checkpoint per skill, found under logs/rsl_rl/g1_<skill>.

One file holds every skill, so recording is all or nothing: running this with a single
skill replaces the rollouts of all the others.

The first settle steps after every reset are dropped, so progress 0 is half a second into
an episode rather than its first frame. Without that the dataset fills with one identical
standing pose per failure: mjlab resets the instant an environment terminates, and the step
after a fall is a robot at its default pose.

Every row carries two clocks. `frame` is control steps since its episode reset, which orders
a rollout and measures a dwell. `phase` is which frame of its own reference the policy was
reading, which is what a tracker is resumed at. For a skill with no reference they are the
same number. For a tracker they are unrelated, because training resets into a sampled frame
of the clip: a state recorded 25 steps into an episode can be anywhere in the motion, and
this recording has them at frames 27 to 183.

Run

1. Record the default skills.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.record

2. Choose which skills, or how much of each.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.record --skills "('walk','jump','kick')"
    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.record --num-envs 128 --steps 800

3. Then pick the entry states.

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
from mjlab.tasks.bridging.experiments.humanoid.selector import WINDOWS
from mjlab.tasks.bridging.experiments.humanoid.selector.table import ROLLOUTS_PATH
from mjlab.tasks.bridging.experiments.humanoid.skills import SKILLS
from mjlab.tasks.registry import load_env_cfg


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

  # TODO we should extend this to work with whatever object the skill expects in the environment
  #   In this case the view script should change and account for the object.
  #   We could make it so each skill declares the objects it needs along with its observation
  #   of the objects
  skills: tuple[str, ...] = tuple(WINDOWS)
  """Which skills to record. Everything with a window by default, since anything else
  is recorded for nothing: build.py only takes states from inside a window.

  A skill that touches an object is recorded like any other, and its entry states mean
  less on their own: where the ball or the box was is part of what the skill was doing,
  and none of that is in a robot state. Its window is what keeps that from mattering, by
  covering only the part of the skill before the object is involved."""

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
  phases: list[np.ndarray] = []
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

    # TODO how many trajectories do we collect per-skill? As of my understanding, we do the following:
    #   1. we create an environment, that implicitly consists of many parallel independent rollouts
    #   2. we step the environment and we record the outcomes
    #   3. we store them
    #   So the number of trajectories depend on the environment, it will be something like 1024, right?
    checkpoint = dataset.find_checkpoint(
      (experiment(name),),
      cfg.checkpoints[index] if index < len(cfg.checkpoints) else None,
      hint=f" Train it with `uv run train {task}`, or name one in `checkpoints`.",
    )
    rows, envs, trajectories, ages, clip_frames, commands = dataset.record(
      task, env_cfg, checkpoint, cfg, name
    )
    states.append(rows)
    env_ids.append(envs)
    trajectory_ids.append(trajectories)
    frames.append(ages)
    phases.append(clip_frames)
    sources.append(np.full(len(rows), index, dtype=np.int16))
    goals.append(commands)
    rollouts = len(np.unique(trajectories))
    print(f"[selector] {name}: {len(rows)} states over {rollouts} rollouts")
    for line in spread_of(commands, trajectories):
      print(f"[selector]   {line}")

  return dataset.write(
    cfg.path,
    states,
    env_ids,
    trajectory_ids,
    frames,
    sources,
    cfg.skills,
    fps,
    goals,
    phases,
  )


if __name__ == "__main__":
  collect(tyro.cli(RecordCfg, config=mjlab.TYRO_FLAGS))
