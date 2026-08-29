"""A dataset built from a motion tracker following LAFAN1, rather than from the skill pool.

    1. Train a trajectory tracking policy on LAFAN1 clips. Each is ~4 minutes of continuous
        motion, so one clip is a lot of material, but it covers only some kind of motions:
        `walk1_subject1` is four solid minutes of human walking, turning, stopping etc., a
        tracker trained on just that sees a lot but will be useless on `jumps1_subject1`,
        because it never saw a jump.

        uv run train Mjlab-Tracking-Flat-Unitree-G1 --env.scene.num-envs 4096
          --env.commands.motion.motion-file data/lafan1/motions/walk1_subject1.npz

        Each motion tracking task will produce a checkpoint under the `g1_tracking` experiment.

        To see how a tracking task is going:

        uv run play Mjlab-Tracking-Flat-Unitree-G1 \
          --checkpoint-file logs/rsl_rl/g1_tracking/<run>/model_3000.pt
          --motion-file data/lafan1/motions/walk1_subject1.npz

    2. Collect the dataset. Pass nothing, however many trackers you trained. Every run
        wrote down the clip it was pointed at, so this reads them back and asks each
        tracker only for the motion it learned.

        # Every run under `g1_tracking`, each paired with its own clip
        uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.tracker

        Name clips yourself only to ask a tracker for motion it was *not* trained on, when
        the question is how far it generalizes. `survival` answers that, per clip.

        uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.tracker \
          --motions "('data/lafan1/motions/walk2_subject1.npz','data/lafan1/motions/walk3_subject1.npz')" \
          --checkpoints "('logs/rsl_rl/g1_tracking/<walk-run>/model_3000.pt',)"

    3. Calibrate the tolerances against the new dataset. The gaps differ from the skills one,
        and stale tolerances are how the statue-vs-policy trap happens.

        uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.evaluate
          --calibrate True --dataset data/bridge/tracker.npz

    4. Train the bridge on it.

        uv run train Mjlab-G1-Bridge --env.scene.num-envs 4096
          --env.commands.bridge.dataset-path data/bridge/tracker.npz

##
# Why this dataset exists
##

skills.py draws its states from the same policies the bridge will be asked to serve, so a
bridge trained and tested on it might have learned bridging or might have learned those
five skills. Nothing in that experiment separates the two.

A tracker trained on LAFAN1 knows nothing about walking to a crate or kicking a ball. If a
bridge trained only on its states still hands over cleanly between the five skills, what it
learned is a property of the robot and the physics rather than of the pool.

##
# Why the tracker and not the clips
##

A retargeted human clip is a description, not a state a G1 is ever in: joint angles fitted
to a different body, heights off by centimetres, nothing accounting for what these
actuators hold. Two such frames are usually an impossible pair, and an unsolvable episode
teaches hedging.

Running a clip through a tracker fixes that. What comes out is what a real G1 did while
following the clip, under physics, with a policy keeping it upright.

##
# One tracker per clip
##

MotionCommandCfg.motion_file names one file, so a tracker is trained against one clip. It
is only trustworthy on clips resembling that one.

So do not train on one clip and collect from all twenty-one. The tracker would fall over
on the twenty it never learned, and the recorded states would be real physics but mostly
recoveries from stumbling.

Instead each clip is measured as it is recorded. A tracker that stays up produces one row
per environment per step after the settle window; one that keeps falling produces far
fewer, because every reset costs `settle` steps. That ratio is `survival`, printed per
clip, and a clip below `min_survival` is dropped with its number said out loud.

Pairing a checkpoint with its clip needs no arguments: `uv run train` dumps the whole env
config into `params/env.yaml` beside the checkpoints, and for a tracking run that names
`motion_file`. So this walks every run under `g1_tracking` and reads the pairing back.
Naming `motions` explicitly overrides it, which is what to do when testing generalization.

##
# What this dataset makes possible
##

Two rows of one episode are a start and target one robot actually got between, and the
frames between them are a deadline it met. dataset.py records the frame index so the
option is there. Nothing reads it yet; using it would mean teaching the command term to
draw both ends of a pair from one episode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets import dataset
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.dataset import (
  TRACKER_DATASET,
  RolloutCfg,
)
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

TRACKING_TASK = "Mjlab-Tracking-Flat-Unitree-G1"
TRACKING_EXPERIMENTS = ("g1_tracking",)
MOTION_DIR = Path("data") / "lafan1" / "motions"


@dataclass
class TrackerCfg(RolloutCfg):
  """How the tracker dataset is collected."""

  motions: tuple[str, ...] = ()
  """Clips to drive the tracker through.

  Empty is the usual case. It does not mean every clip on disk, it means every clip some
  tracking run was trained on, read back from the config each run wrote beside its own
  checkpoints. One tracker gives one clip, four give four, and nothing has to be typed.

  Naming clips here overrides that, and then `checkpoints` says what drives them.

  These are mjlab tracking files, per-body world poses and velocities, not the retargeted
  joint-angle files. See this module's header for why that matters."""

  task: str = TRACKING_TASK
  checkpoints: tuple[str, ...] = ()
  """Only read when `motions` is given. One checkpoint per clip in the same order, or a
  single one used for all of them. Empty falls back to the newest run's checkpoint, which
  is right only when every named clip is one that tracker can hold."""

  path: Path = TRACKER_DATASET
  steps: int = 300
  """Shorter per clip than the skills dataset, because there are many more clips."""

  min_survival: float = 0.6
  """How much of a clip the tracker has to get through before its states are kept.

  One if the robot never falls, near zero if it falls constantly, since every reset costs
  `settle` steps of recording. 0.6 keeps a clip the tracker merely finds hard and drops one
  it cannot follow at all. Set it to zero to keep everything and read the printed numbers
  yourself."""


_MOTION_LINE = re.compile(r"^\s*motion_file:\s*(.+?)\s*$", re.MULTILINE)


def _trained_on(run: Path) -> Path | None:
  """Which clip this training run was pointed at, read from the config it wrote down.

  `uv run train` dumps the whole env config beside the checkpoints, so a run already
  records the clip it learned.

  Matched as text, not parsed as YAML. The dump carries `!!python/object/apply` tags for
  things like `Path`; a safe loader refuses them and an unsafe one would execute them.
  """
  params = run / "params" / "env.yaml"
  if not params.exists():
    return None
  match = _MOTION_LINE.search(params.read_text(encoding="utf-8"))
  if match is None:
    return None
  clip = Path(match.group(1).strip().strip("'\""))
  return clip if clip.exists() else None


def _discover() -> tuple[tuple[Path, Path], ...]:
  """Every tracking run that has a checkpoint, paired with the clip it was trained on.

  This is what makes running with no arguments correct rather than merely safe. A
  checkpoint on its own says nothing about which motion it can hold, so the obvious default
  of newest checkpoint against every clip in the folder drives a walking tracker through
  the jumps and leaves the survival filter to discard twenty wasted rollouts.
  """
  root = dataset.LOG_ROOT / TRACKING_EXPERIMENTS[0]
  if not root.exists():
    return ()
  pairs: list[tuple[Path, Path]] = []
  for run in sorted(root.iterdir()):
    if not run.is_dir():
      continue
    found = sorted(run.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    clip = _trained_on(run)
    if found and clip is not None:
      pairs.append((clip, found[-1]))
  return tuple(pairs)


def _pairs(cfg: TrackerCfg) -> tuple[tuple[Path, Path], ...]:
  """(clip, checkpoint) for everything about to be recorded."""
  if not cfg.motions:
    found = _discover()
    if not found:
      raise SystemExit(
        f"No usable tracking run under {dataset.LOG_ROOT / TRACKING_EXPERIMENTS[0]}. "
        f"Train one with `uv run train {cfg.task} --env.commands.motion.motion-file "
        f"{MOTION_DIR / 'walk1_subject1.npz'}`, or name clips and checkpoints yourself "
        f"with `motions` and `checkpoints`."
      )
    return found

  clips = tuple(Path(m) for m in cfg.motions)
  missing = [str(p) for p in clips if not p.exists()]
  if missing:
    raise SystemExit(f"No clip at {', '.join(missing)}.")

  out: list[tuple[Path, Path]] = []
  for index, clip in enumerate(clips):
    # One checkpoint per clip if that many were given, otherwise a single one for all of
    # them, which is what a tracker covering a set of clips wants
    explicit = (
      cfg.checkpoints[index]
      if index < len(cfg.checkpoints)
      else (cfg.checkpoints[-1] if cfg.checkpoints else None)
    )
    out.append(
      (
        clip,
        dataset.find_checkpoint(
          TRACKING_EXPERIMENTS,
          explicit,
          hint=(
            f" Train a tracker on this clip: `uv run train {cfg.task} "
            f"--env.commands.motion.motion-file {clip}`, then re-run this."
          ),
        ),
      )
    )
  return tuple(out)


def collect(cfg: TrackerCfg) -> Path:
  """Drive each tracker through the clip it was trained on and write one npz."""
  if cfg.steps <= cfg.settle:
    raise SystemExit(
      f"steps is {cfg.steps} and settle is {cfg.settle}, so every row would be inside the "
      f"settle window and every clip would be dropped for zero survival, which looks "
      f"exactly like a tracker that cannot hold its clip and is not."
    )
  pairs = _pairs(cfg)
  print(f"[dataset] {len(pairs)} clip/checkpoint pairs")
  for clip, checkpoint in pairs:
    print(f"[dataset]   {clip.stem} <- {checkpoint}")

  states: list[np.ndarray] = []
  env_ids: list[np.ndarray] = []
  frames: list[np.ndarray] = []
  sources: list[np.ndarray] = []
  goals: list[np.ndarray] = []
  names: list[str] = []
  fps = 0.0

  for clip, checkpoint in pairs:
    env_cfg = load_env_cfg(cfg.task)
    motion = env_cfg.commands["motion"]
    if not isinstance(motion, MotionCommandCfg):
      raise SystemExit(
        f"'{cfg.task}' has no motion command, so it is not a tracking task and there is "
        f"no clip to put in it."
      )
    motion.motion_file = str(clip)

    rate = dataset.control_rate(env_cfg)
    if fps and abs(rate - fps) > 1e-6:
      raise SystemExit(
        f"'{clip.stem}' runs at {rate:.1f} Hz and the clips before it at {fps:.1f} Hz."
      )
    fps = rate

    rows, envs, ages, commands = dataset.record(
      cfg.task, env_cfg, checkpoint, cfg, clip.stem
    )

    # What a tracker that never fell would have produced. Each fall costs `settle` steps
    # of recording, so the shortfall is how much of the clip it lost
    ceiling = cfg.num_envs * max(cfg.steps - cfg.settle, 1)
    survival = len(rows) / ceiling
    if survival < cfg.min_survival:
      print(
        f"[dataset] {clip.stem}: survival {survival:.2f} below {cfg.min_survival:.2f}, "
        f"dropped, this tracker cannot hold this clip"
      )
      continue

    states.append(rows)
    env_ids.append(envs)
    frames.append(ages)
    sources.append(np.full(len(rows), len(names), dtype=np.int16))
    goals.append(commands)
    names.append(clip.stem)
    print(f"[dataset] {clip.stem}: {len(rows)} states, survival {survival:.2f}")

  if not names:
    raise SystemExit(
      f"Every clip came in under min_survival {cfg.min_survival:.2f}, so this tracker "
      f"cannot follow any of them. Train it on one of these clips, or point `motions` at "
      f"the one it was trained on."
    )
  return dataset.write(
    cfg.path, states, env_ids, frames, sources, tuple(names), fps, goals
  )


if __name__ == "__main__":
  collect(tyro.cli(TrackerCfg, config=mjlab.TYRO_FLAGS))
