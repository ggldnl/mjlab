"""Find the candidate entry points of each skill, and measure them.

Second step of the pipeline: reads the rollouts record.py wrote, writes the candidate table
filter.py judges. Measures, decides nothing. Every cluster it finds is written out,
including the ones no bridge could use, because filter.py can only report why a candidate
failed if the numbers behind the failure are in the file.

    1. Canonicalize. Drop ground position, drop yaw, rotate the velocities into the
       heading frame. See state.py.
    2. Scale each channel so one unit means the same thing everywhere.
    3. Cluster with k-medoids. Cluster centers are recorded states, never averages: an
       averaged quaternion is not a pose the robot can hold.
    4. Measure each cluster: how many rollouts pass through it, how long they stay, how
       wide it is, where in the skill it sits, and how far off the floor it is.
    5. Measure what the robot can do, once, over every skill. See achievable.

A cluster many rollouts pass through is a spot the skill has to go through to do its job,
so it is a spot another skill could join it at. That is the whole idea. Nothing here knows
which skill it is looking at. What the columns mean is in table.py.

Run

1. Find the candidates, having recorded the skills first.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.build

2. Ask for more spots per skill, or only one skill.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.build --clusters 48
    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.build --skills "('jump',)"

3. Then accept or reject them.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.filter
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.dataset import (
  Dataset,
  load_dataset,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.ground import Ground
from mjlab.tasks.bridging.experiments.humanoid.selector.state import (
  canonical,
  channel_gap,
  features,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.table import (
  CANDIDATES_PATH,
  ROLLOUTS_PATH,
  Entry,
  EntryTable,
)

##
# Clustering.
##


def kmedoids(
  feat: torch.Tensor, clusters: int, rounds: int = 5, sample: int = 512, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
  """Cluster the rows. Returns the row index of each center, and every row's label.

  Centers are rows, not averages: the bridge has to aim at one, and an averaged quaternion
  is not a pose.

  Seeded by farthest point sampling, which is deterministic and does not spend every seed
  in the dense part of the cloud the way random seeding does. For a jump that dense part is
  standing still, and a seed spent there is an entry point not found anywhere else.

  The update step picks, out of a random sample of each cluster, the row with the smallest
  total distance to that sample. Sampling makes the step linear in cluster size instead of
  quadratic, and the exact medoid is not worth a 10000 x 10000 matrix.

  An empty cluster keeps its previous center rather than being reseeded. It means the cloud
  holds fewer distinct spots than asked for, which is an answer.
  """
  rng = np.random.default_rng(seed)
  count = min(clusters, feat.shape[0])

  middle = feat.mean(dim=0, keepdim=True)
  picked = [int(torch.cdist(middle, feat).argmin())]
  far = torch.cdist(feat[picked[-1] : picked[-1] + 1], feat).squeeze(0)
  while len(picked) < count:
    picked.append(int(far.argmax()))
    here = torch.cdist(feat[picked[-1] : picked[-1] + 1], feat).squeeze(0)
    far = torch.minimum(far, here)

  centers = torch.tensor(picked, device=feat.device)
  labels = torch.cdist(feat, feat[centers]).argmin(dim=1)
  for _ in range(rounds):
    for index in range(count):
      members = (labels == index).nonzero().flatten()
      if members.numel() == 0:
        continue
      if members.numel() > sample:
        draw = rng.integers(0, members.numel(), sample)
        members = members[torch.from_numpy(draw).to(feat.device)]
      cost = torch.cdist(feat[members], feat[members]).sum(dim=1)
      centers[index] = members[int(cost.argmin())]
    labels = torch.cdist(feat, feat[centers]).argmin(dim=1)
  return centers, labels


##
# Measuring.
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
) -> np.ndarray:
  """Fastest per-channel change per second the corpus achieved. (6,)

  Take every pair of states a fixed time apart inside one rollout, measure how far apart
  they are per channel, divide by the time between them, and keep the 99th percentile.

  This is what turns a gap into a difficulty. Whether shedding 1 m/s in 0.7 s is a lot
  depends on what a G1 can do, which the rollouts already answer, so an effort above 1
  means the hand-over needs a faster change than any recorded skill performed.

  Measured over every skill together: the bound is a property of the robot, and the bridge
  that has to meet it is the same policy whichever skill it hands to.

  Rate is assumed to scale with time, so one number per channel covers every window length.
  Close enough over the range a bridge window spans, wrong over long ones where a body runs
  out of room rather than out of acceleration.

  spans_s is the bridge's window range. cap subsamples before the quantile, which is a cost
  bound and not a statistical one at these sizes.
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
    keep = torch.randperm(gaps.shape[0], device=gaps.device)[:cap]
    gaps = gaps[keep]
  return torch.quantile(gaps, quantile, dim=0).cpu().numpy().astype(np.float64)


def progress_of(trajectory: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
  """How far into its own rollout each row is. (N,) in [0, 1].

  Normalized per rollout, because rollouts have different lengths and because the
  recording drops the first steps after every reset, so frames do not start at zero.

  Only means something for a skill with episodes. Walk and run do not terminate inside
  a recording, so each environment is one rollout the length of the whole run, and this
  reads as time since recording started rather than phase within a skill. Coverage still
  says something there, since it counts environments that visit a spot; progress does
  not. Segmenting locomotion by gait cycle rather than by episode is what would fix it.
  """
  _, inverse = torch.unique(trajectory, return_inverse=True)
  width = int(inverse.max().item()) + 1
  value = frame.float()
  low = torch.zeros(width, device=frame.device).scatter_reduce(
    0, inverse, value, "amin", include_self=False
  )
  high = torch.zeros(width, device=frame.device).scatter_reduce(
    0, inverse, value, "amax", include_self=False
  )
  span = (high - low).clamp(min=1.0)
  return ((value - low[inverse]) / span[inverse]).clamp(0.0, 1.0)


def dwell_steps(
  labels: torch.Tensor, trajectory: torch.Tensor, frame: torch.Tensor, clusters: int
) -> torch.Tensor:
  """Median consecutive steps a rollout spends in each cluster. (clusters,) in steps.

  A cluster that only fast-moving rollouts clip through scores one or two, and it is a
  bad thing to aim at whatever else it scores: the bridge would have to arrive on an
  exact tick.
  """
  width = int(frame.max().item()) + 1
  order = torch.argsort(trajectory * width + frame)
  label, path, step = labels[order], trajectory[order], frame[order]

  # A run is a stretch of one rollout that stays in one cluster and steps by one frame
  same = (
    (label[1:] == label[:-1]) & (path[1:] == path[:-1]) & (step[1:] == step[:-1] + 1)
  )
  opens = torch.cat([torch.ones(1, dtype=torch.bool, device=same.device), ~same])
  lengths = torch.bincount(opens.long().cumsum(0) - 1).float()

  out = torch.zeros(clusters, device=labels.device)
  for index in range(clusters):
    here = lengths[label[opens] == index]
    if here.numel():
      out[index] = here.median()
  return out


##
# Building the candidate table.
##


@dataclass
class BuildCfg:
  """How candidates are found."""

  path: Path = ROLLOUTS_PATH
  """Rollouts to read. One source per skill. Written by record.py."""
  out: Path = CANDIDATES_PATH

  skills: tuple[str, ...] = ()
  """Which sources to look at. Empty means all of them."""

  clusters: int = 32
  """Candidate spots per skill. filter.py drops whatever does not survive, so this is an
  upper bound on the final table, not its size.

  Measured on walk, run and jump. At 16 the jump's opening stand was not a cluster of
  its own and the candidates that survived had a spread of 5 to 6, which is a region
  rather than a place. At 32 the stand comes out at 0.96 coverage and the spreads halve.
  Read spread after changing it: a wide cluster holding a large share means raise
  this."""

  split: str = "train"
  """Which side of the dataset holdout to measure. The holdout exists for the bridge;
  this is a measurement and 7/8 of the rollouts is plenty. Pass 'eval' to cross-check."""

  device: str = "cuda:0"
  seed: int = 0


def build(cfg: BuildCfg) -> EntryTable:
  """Every candidate of every requested skill, measured and unjudged."""
  data = load_dataset(cfg.path, cfg.device, cfg.split)
  ground = Ground()
  if ground.num_joints != data.num_joints:
    raise SystemExit(
      f"The rollouts hold {data.num_joints}-joint states and this G1 has "
      f"{ground.num_joints}, so they were recorded against a different robot."
    )

  everything = canonical(data.states)
  rates = achievable(everything, data.trajectory, data.frame, data.fps)
  print("[selector] achievable rates per second:")
  for name, rate in zip(CHANNEL_REPORT, rates, strict=True):
    print(f"[selector]   {name:<12} {rate:.3f}")

  wanted = cfg.skills or data.names
  candidates: list[Entry] = []
  for skill in wanted:
    candidates += for_skill(data, everything, skill, ground, cfg)
  return EntryTable(entries=tuple(candidates), fps=data.fps, rates=rates)


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


def for_skill(
  data: Dataset, everything: torch.Tensor, skill: str, ground: Ground, cfg: BuildCfg
) -> list[Entry]:
  """One skill's candidates, most held first."""
  rows = data.of((skill,))
  states = everything[rows]
  trajectory, frame = data.trajectory[rows], data.frame[rows]
  # What the skill was being asked for at each of these states. Carried onto the entry
  # and never clustered on: a crouch is a crouch whatever distance it is crouching for,
  # and splitting by command would make one node look like five rare ones
  commands = data.commands_of(skill)
  commands = None if commands is None else commands[rows]

  feat = features(states)
  centers, labels = kmedoids(feat, cfg.clusters, seed=cfg.seed)
  count = int(centers.numel())

  progress = progress_of(trajectory, frame)
  dwell = dwell_steps(labels, trajectory, frame, count) / data.fps
  rollouts = int(torch.unique(trajectory).numel())

  found: list[Entry] = []
  for index in range(count):
    members = (labels == index).nonzero().flatten()
    if members.numel() == 0:
      continue
    center = int(centers[index])
    pose = states[center].cpu().numpy().astype(np.float32)
    # The center's own progress, not the median over its members. A skill that comes
    # back to a pose puts both visits in one cluster, and the median of those is a
    # moment nothing was ever at. frame and progress have to name the same tick
    here = float(progress[center])
    seconds = float(dwell[index])
    coverage = int(torch.unique(trajectory[members]).numel()) / rollouts
    found.append(
      Entry(
        skill=skill,
        name=f"p{round(here * 100):02d}",
        state=pose,
        command=(
          np.zeros(0, dtype=np.float32)
          if commands is None
          else commands[center].cpu().numpy().astype(np.float32)
        ),
        frame=int(frame[center]),
        progress=here,
        coverage=coverage,
        spread=float(torch.cdist(feat[members], feat[center : center + 1]).median()),
        clearance=ground.clearance(pose.astype(np.float64)),
        dwell_s=seconds,
        hold_s=coverage * seconds,
        share=int(members.numel()) / int(rows.numel()),
      )
    )

  found.sort(key=lambda entry: entry.hold_s, reverse=True)
  print(f"[selector] {skill}: {len(found)} candidates over {rollouts} rollouts")
  return rename(found)


def rename(entries: list[Entry]) -> list[Entry]:
  """Make the progress names unique, keeping the order they came in.

  Two clusters can land on the same percentage. Suffixing beats renumbering: the name
  still says where in the skill the entry is, which is the only thing it is for.

  Done here rather than in filter.py so a name means the same thing in both files, and
  a rejection report can be read against the candidate table.
  """
  seen: dict[str, int] = {}
  out: list[Entry] = []
  for entry in entries:
    count = seen.get(entry.name, 0)
    seen[entry.name] = count + 1
    suffix = "" if count == 0 else chr(ord("a") + count - 1)
    out.append(replace(entry, name=entry.name + suffix))
  return out


def main(cfg: BuildCfg) -> None:
  table = build(cfg)
  table.save(cfg.out)
  print(
    "[selector] nothing has been judged yet. Run "
    "`python -m ...selector.filter` to accept or reject these."
  )


if __name__ == "__main__":
  main(tyro.cli(BuildCfg, config=mjlab.TYRO_FLAGS))
