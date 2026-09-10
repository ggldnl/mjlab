"""Pick the entry states of each skill, one per slice of its window.

Second step of the pipeline: reads the rollouts record.py wrote, writes the entry table.
Nothing is judged and nothing is dropped. What a skill's entry states are is decided by
its window in selector/__init__.py, and this only carries that decision out.

Per skill:

    1. Keep the rows whose frame falls inside the window.
    2. Cut the window into as many equal slices as the window asks for states.
    3. Canonicalize. Drop ground position, drop yaw, rotate the velocities into the
       heading frame. See state.py.
    4. Take the medoid of each slice: the recorded state closest to every other state in
       that slice. Never an average, because an averaged quaternion is not a pose, and
       never a cluster center, because the slices are already fixed by the window.

The medoid is the whole robustness story. Every rollout in a slice was at the same frame
of the same skill, so they should agree; the ones that do not are the rollouts that
drifted or fell, they sit on the outside of the cloud, and the middle of the cloud is a
state the skill really was in. No threshold decides that, so there is none to tune.

Then, once over every skill together, how fast the robot can change each channel. See
achievable. query.py divides by it.

Run

1. Pick the entry states, having recorded the skills first.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.build

2. Only one skill, or a different rollout file.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.build --skills "('jump',)"

3. Then look at them.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.view
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.dataset import (
  Dataset,
  load_dataset,
)
from mjlab.tasks.bridging.experiments.humanoid.selector import WINDOWS, Window
from mjlab.tasks.bridging.experiments.humanoid.selector.ground import Ground
from mjlab.tasks.bridging.experiments.humanoid.selector.state import (
  canonical,
  channel_gap,
  features,
  reference_in_heading,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.table import (
  ROLLOUTS_PATH,
  TABLE_PATH,
  Entry,
  EntryTable,
)

##
# The window.
##


def resolve(skill: str, window: Window, phase: torch.Tensor) -> tuple[int, int]:
  """The window clamped to the frames that were actually recorded.

  Recording drops the first steps after every reset, so the earliest frame in a rollout
  file is the settle count and never zero. A window asking for the true opening of a
  clip is not wrong, it just gets the earliest thing there is, and the clamp is printed
  so the difference is visible rather than silent.
  """
  first, last = int(phase.min()), int(phase.max()) + 1
  low, high = max(window.phase[0], first), min(window.phase[1], last)
  if high - low < window.states:
    raise SystemExit(
      f"'{skill}' asks for {window.states} states over frames {window.phase[0]} to "
      f"{window.phase[1]}, and only frames {first} to {last} were recorded, leaving "
      f"{max(high - low, 0)} frames. Widen the window in selector/__init__.py, ask for "
      f"fewer states, or record more of this skill."
    )
  if (low, high) != window.phase:
    print(
      f"[selector] {skill}: window {window.phase[0]}-{window.phase[1]} clamped to "
      f"{low}-{high}, the recorded range being {first}-{last}"
    )
  return low, high


def edges(window: tuple[int, int], states: int) -> list[tuple[int, int]]:
  """The window as that many half open frame slices, as equal as whole frames allow."""
  low, high = window
  cuts = np.linspace(low, high, states + 1).round().astype(int)
  return [(int(cuts[i]), int(cuts[i + 1])) for i in range(states)]


##
# Choosing one state per slice.
##


def medoid(feat: torch.Tensor, sample: int = 1024, seed: int = 0) -> int:
  """Row closest to all the others. An index into feat.

  Subsampled, which makes it linear in the slice size instead of quadratic. The exact
  medoid is not worth an N x N matrix: these rows are one frame of one skill, so the
  cloud is tight and any of its middle is the same pose.
  """
  rows = torch.arange(feat.shape[0], device=feat.device)
  if rows.numel() > sample:
    draw = torch.Generator(device=feat.device)
    draw.manual_seed(seed)
    rows = rows[
      torch.randperm(rows.numel(), device=feat.device, generator=draw)[:sample]
    ]
  cost = torch.cdist(feat[rows], feat[rows]).sum(dim=1)
  return int(rows[int(cost.argmin())])


##
# What the robot can do.
##


def runs(trajectory: torch.Tensor, frame: torch.Tensor) -> tuple[torch.Tensor, ...]:
  """Rows in time order, and how many more rows each one is followed by in its rollout.

  Returns (order, available). Sorting is what makes "the row k ticks later" an index
  offset: the recording is time-major, so rows of one rollout are strided rather than
  adjacent.
  """
  width = int(frame.max().item()) + 1
  order = torch.argsort(trajectory * width + frame)
  path, step = trajectory[order], frame[order]
  steps_by_one = (path[1:] == path[:-1]) & (step[1:] == step[:-1] + 1)
  opens = torch.cat(
    [torch.ones(1, dtype=torch.bool, device=steps_by_one.device), ~steps_by_one]
  )
  run = opens.long().cumsum(0) - 1
  last = torch.bincount(run).cumsum(0) - 1
  positions = torch.arange(order.numel(), device=order.device)
  return order, last[run] - positions


def achievable(
  states: torch.Tensor,
  trajectory: torch.Tensor,
  frame: torch.Tensor,
  fps: float,
  spans_s: tuple[float, ...] = (0.3, 0.5, 0.7, 1.0, 1.2),
  quantile: float = 0.99,
  cap: int = 200_000,
  seed: int = 0,
) -> np.ndarray:
  """Fastest per-channel change per second the corpus achieved. (6,)

  Take every pair of states a fixed time apart inside one rollout, measure how far apart
  they are per channel, divide by the time between them, and keep the 99th percentile.

  This is what turns a gap into a difficulty. Whether shedding 1 m/s in 0.7 s is a lot
  depends on what a G1 can do, which the rollouts already answer, so an effort above 1
  means the hand-over needs a faster change than any recorded skill performed.

  Measured over every skill together: the bound is a property of the robot, and the
  bridge that has to meet it is the same policy whichever skill it hands to.

  Rate is assumed to scale with time, so one number per channel covers every window
  length. Close enough over the range a bridge window spans, wrong over long ones where
  a body runs out of room rather than out of acceleration.

  spans_s is the bridge's window range. cap subsamples before the quantile, which is a
  cost bound and not a statistical one at these sizes. seed makes that subsample
  repeatable: query.py divides by what comes out of here, so an unseeded draw moves
  every effort score.
  """
  order, available = runs(trajectory, frame)
  rows: list[torch.Tensor] = []
  for seconds in spans_s:
    steps = max(1, int(round(seconds * fps)))
    usable = (available >= steps).nonzero().flatten()
    if usable.numel() == 0:
      continue
    here, there = order[usable], order[usable + steps]
    rows.append(channel_gap(states[here], states[there]) / (steps / fps))
  if not rows:
    raise SystemExit(
      "No rollout is long enough to measure a rate over. Record more steps per episode."
    )

  gaps = torch.cat(rows)
  if gaps.shape[0] > cap:
    draw = torch.Generator(device=gaps.device)
    draw.manual_seed(seed)
    keep = torch.randperm(gaps.shape[0], device=gaps.device, generator=draw)[:cap]
    gaps = gaps[keep]
  return torch.quantile(gaps, quantile, dim=0).cpu().numpy().astype(np.float64)


CHANNEL_REPORT = (
  "root_z m/s",
  "tilt rad/s",
  "lin_vel m/s2",
  "ang_vel rad/s2",
  "joint rad/s",
  "joint rad/s2",
)
"""What each achievable rate is, in the units it comes out in. A change in a velocity
per second is an acceleration, which the channel names do not say."""


##
# Building the table.
##


@dataclass
class BuildCfg:
  """Where to read and write. The windows themselves live in selector/__init__.py."""

  path: Path = ROLLOUTS_PATH
  """Rollouts to read. One source per skill. Written by record.py."""
  out: Path = TABLE_PATH

  skills: tuple[str, ...] = ()
  """Which skills to build. Empty means every skill that has both a window and
  rollouts."""

  split: str = "train"
  """Which side of the dataset holdout to measure. The holdout exists for the bridge;
  this is a measurement and 7/8 of the rollouts is plenty. Pass 'eval' to cross-check."""

  device: str = "cuda:0"

  sample: int = 1024
  """Rows a slice's medoid is searched over. See medoid."""

  seed: int = 0
  """Every draw this file makes: the medoid subsample, and the one achievable takes
  before its quantile. Nothing else here is random."""


def build(cfg: BuildCfg) -> EntryTable:
  """Every entry state of every requested skill."""
  data = load_dataset(cfg.path, cfg.device, cfg.split)
  if any(
    value is None
    for value in (
      data.previous_action,
      data.reference,
      data.motion_file,
      data.motion_scale,
    )
  ):
    raise SystemExit(
      "Rollouts lack resumption context. Rerun selector.record, then selector.build"
    )
  ground = Ground()
  if ground.num_joints != data.num_joints:
    raise SystemExit(
      f"The rollouts hold {data.num_joints}-joint states and this G1 has "
      f"{ground.num_joints}, so they were recorded against a different robot."
    )

  everything = canonical(data.states)
  rates = achievable(everything, data.trajectory, data.frame, data.fps, seed=cfg.seed)
  print("[selector] achievable rates per second:")
  for name, rate in zip(CHANNEL_REPORT, rates, strict=True):
    print(f"[selector]   {name:<15} {rate:.3f}")

  wanted = cfg.skills or tuple(n for n in data.names if n in WINDOWS)
  unknown = [name for name in wanted if name not in WINDOWS]
  if unknown:
    raise SystemExit(
      f"No window for {', '.join(unknown)}. Add one to WINDOWS in "
      f"selector/__init__.py, or leave the skill out: a skill with no window gets no "
      f"entry states and simply cannot be handed over to."
    )
  skipped = [name for name in data.names if name not in wanted]
  if skipped:
    print(f"[selector] recorded but given no window: {', '.join(skipped)}")

  entries: list[Entry] = []
  for skill in wanted:
    entries += for_skill(data, everything, skill, WINDOWS[skill], ground, cfg)
  if not entries:
    raise SystemExit(
      "No skill had both a window and rollouts, so there is nothing to write."
    )
  return EntryTable(entries=tuple(entries), fps=data.fps, rates=rates)


def for_skill(
  data: Dataset,
  everything: torch.Tensor,
  skill: str,
  window: Window,
  ground: Ground,
  cfg: BuildCfg,
) -> list[Entry]:
  """One skill's entry states, earliest frame first."""
  rows = data.of((skill,))
  assert data.reference is not None and data.previous_action is not None
  assert data.motion_file is not None and data.motion_scale is not None
  references = reference_in_heading(data.states[rows], data.reference[rows])
  states = everything[rows]
  trajectory = data.trajectory[rows]
  # The skill's own clock. For a tracker it is the frame of the reference the policy was
  # reading, which is the only thing it can be resumed at; for a skill with no reference
  # record.py writes the step count here, which resumes nothing and still orders the
  # window
  phase = data.phase[rows] if data.phase is not None else data.frame[rows]
  # What the skill was being asked for. Carried onto the entry and never used to pick
  # one: a crouch is a crouch whatever distance it is crouching for
  commands = data.commands_of(skill)
  commands = None if commands is None else commands[rows]

  feat = features(states)
  rollouts = int(torch.unique(trajectory).numel())
  found: list[Entry] = []

  for index, (low, high) in enumerate(
    edges(resolve(skill, window, phase), window.states)
  ):
    members = ((phase >= low) & (phase < high)).nonzero().flatten()
    if members.numel() == 0:
      print(f"[selector] {skill}: no rollout was at frames {low}-{high}, slice dropped")
      continue

    here = feat[members]
    center = int(members[medoid(here, cfg.sample, cfg.seed + index)])
    pose = states[center].cpu().numpy().astype(np.float32)
    found.append(
      Entry(
        skill=skill,
        name=f"f{int(phase[center]):03d}",
        state=pose,
        command=(
          np.zeros(0, dtype=np.float32)
          if commands is None
          else commands[center].cpu().numpy().astype(np.float32)
        ),
        frame=int(phase[center]),
        seconds=int(phase[center]) / data.fps,
        coverage=int(torch.unique(trajectory[members]).numel()) / rollouts,
        spread=float(torch.cdist(here, feat[center : center + 1]).median()),
        clearance=ground.clearance(pose.astype(np.float64)),
        previous_action=data.previous_action[rows[center]].cpu().numpy().copy(),
        reference=references[center].cpu().numpy().copy(),
        motion_file=str(data.motion_file[int(rows[center])]),
        motion_scale=float(data.motion_scale[rows[center]]),
      )
    )

  found.sort(key=lambda entry: entry.frame)
  print(f"[selector] {skill}: {len(found)} entries over {rollouts} rollouts")
  return found


def main(cfg: BuildCfg) -> None:
  table = build(cfg)
  print("\n".join(table.lines()))
  table.save(cfg.out)


if __name__ == "__main__":
  main(tyro.cli(BuildCfg, config=mjlab.TYRO_FLAGS))
