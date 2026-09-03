"""Find the entry points of each skill from its own recorded rollouts.

Method
------

    1. Canonicalize. Drop ground position, drop yaw, rotate the velocities into the
       heading frame. Two states that differ only by where on the floor they happened
       become the same state.
    2. Scale each channel by SCALE, so one unit means the same thing everywhere.
    3. Cluster with k-medoids. Cluster centers are recorded states, never averages:
       an averaged quaternion is not a pose the robot can hold.
    4. Score each cluster and drop the ones rollouts do not reliably pass through, or
       that no bridge could deliver the robot to.

A cluster many rollouts pass through is a spot the skill has to go through to do its
job. That is the whole criterion. Nothing here knows which skill it is looking at: no
contacts, no gait, no clip landmarks, no per-skill thresholds.

The one thing that criterion cannot see is whether the robot can be put there at all.
Mid-flight states are real waists of a jump and no controller can deliver one, so a
second filter measures each candidate against the floor. See ground.py.

What the columns mean is in table.py.

Run
---

    1. Record the skills.

        uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.record

    2. Build the table.

        uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.build
        uv run python -m ...selector.build --clusters 48 --min-coverage 0.1
        uv run python -m ...selector.build --skills "('jump',)"

    3. Look at what came out.

        uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.view
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import tyro

import mjlab
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.dataset import (
  ROOT_STATE_DIM,
  Dataset,
  load_dataset,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.ground import Ground
from mjlab.tasks.bridging.experiments.humanoid.selector.table import (
  ROLLOUTS_PATH,
  TABLE_PATH,
  Entry,
  EntryTable,
)
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)

CHANNELS = ("root_z", "tilt", "lin_vel", "ang_vel", "joint_pos", "joint_vel")

SCALE = (0.05, 0.05, 0.15, 0.30, 0.10, 1.50)
"""Metres, radians, m/s, rad/s, radians, rad/s.

The bridge's arrival tolerances, so a spread of 1.0 means the cluster is about as wide
as a hand-over is allowed to be off. Copied rather than imported: the bridge splits arms
from legs and needs joint names to do it, and a dataset does not record joint names.
"""


##
# Turning states into something distances can be measured in.
##


def canonical(states: torch.Tensor) -> torch.Tensor:
  """Drop where on the floor and which way round. Same layout in and out.

  Ground position goes to zero and yaw is removed, so two walk states a metre apart
  facing different ways come out identical. Both velocities are rotated with the pose,
  or they would still point the way the robot happened to be facing when recorded.

  This is also the layout the bridge is aimed in: whoever places the target picks the
  ground position and the heading.
  """
  quat = states[:, 3:7]
  heading = yaw_quat(quat)
  out = states.clone()
  out[:, 0:2] = 0.0
  out[:, 3:7] = quat_mul(quat_conjugate(heading), quat)
  out[:, 7:10] = quat_apply_inverse(heading, states[:, 7:10])
  out[:, 10:ROOT_STATE_DIM] = quat_apply_inverse(heading, states[:, 10:ROOT_STATE_DIM])
  return out


def features(states: torch.Tensor) -> torch.Tensor:
  """Canonical states as vectors whose euclidean distance is the clustering metric.

  Each channel is divided by its scale and by the square root of its width, so a
  29-number joint block does not outweigh a 3-number velocity just by being wider.

  Tilt contributes the xyz of its quaternion, doubled, with w forced positive. For the
  tilts a standing robot has, that is the tilt angle in radians, and forcing the sign
  stops one rotation being written two ways.
  """
  num_joints = (states.shape[1] - ROOT_STATE_DIM) // 2
  tilt = states[:, 3:7]
  tilt = tilt * torch.where(tilt[:, :1] < 0.0, -1.0, 1.0)
  blocks = [
    states[:, 2:3],
    2.0 * tilt[:, 1:4],
    states[:, 7:10],
    states[:, 10:ROOT_STATE_DIM],
    states[:, ROOT_STATE_DIM : ROOT_STATE_DIM + num_joints],
    states[:, ROOT_STATE_DIM + num_joints :],
  ]
  scaled = [
    block / (scale * float(np.sqrt(block.shape[1])))
    for block, scale in zip(blocks, SCALE, strict=True)
  ]
  return torch.cat(scaled, dim=-1)


##
# Clustering.
##


def kmedoids(
  feat: torch.Tensor, clusters: int, rounds: int = 5, sample: int = 512, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
  """Cluster the rows. Returns the row index of each center, and every row's label.

  Centers are rows, not averages. The bridge has to be able to aim at one, and an
  averaged quaternion is not a pose.

  Seeded by farthest point sampling: start from the row nearest the middle of the
  cloud, then repeatedly take the row furthest from everything picked so far. It is
  deterministic, and it does not spend every seed in the dense part of the cloud the
  way random seeding does. For a jump that dense part is standing still, and a seed
  spent there is an entry point not found anywhere else.

  The update step picks, out of a random sample of each cluster, the row with the
  smallest total distance to that sample. Sampling makes the step linear in cluster
  size instead of quadratic, and the exact medoid is not worth a 10000 x 10000 matrix.

  An empty cluster keeps its previous center rather than being reseeded. It means the
  cloud holds fewer distinct spots than asked for, which is an answer.
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
# Scoring a cluster.
##


def progress_of(trajectory: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
  """How far into its own rollout each row is. (N,) in [0, 1].

  Normalized per rollout, because rollouts have different lengths and because the
  recording drops the first steps after every reset, so frames do not start at zero.

  Only means something for a skill with episodes. Walk and run do not terminate inside
  a recording, so each environment is one rollout the length of the whole run, and this
  reads as time since recording started rather than phase within a skill. Coverage still
  says something there, since it counts environments that visit a spot; progress does
  not, and `max_progress` filters on noise. Segmenting locomotion by gait cycle rather
  than by episode is what would fix it.
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
# Building the table.
##


@dataclass
class BuildCfg:
  """How the entry table is built."""

  path: Path = ROLLOUTS_PATH
  """Rollouts to read. One source per skill. Written by record.py."""
  out: Path = TABLE_PATH

  skills: tuple[str, ...] = ()
  """Which sources to build entries for. Empty means all of them."""

  clusters: int = 32
  """Candidate spots per skill, before filtering. The filters drop whatever does not
  survive, so this is an upper bound on the table, not its size.

  Measured on walk, run and jump. At 16 the jump's opening stand was not a cluster of
  its own and the entries that survived had a spread of 5 to 6, which is a region rather
  than a place. At 32 the stand comes out at 0.96 coverage and the spreads halve. Read
  `spread` after changing it: a wide cluster holding a large `share` means raise this."""

  min_coverage: float = 0.2
  """Drop a spot fewer than this fraction of rollouts pass through.

  Low on purpose. A skill that branches, kicking with either leg, sends about half its
  rollouts down each branch, and both branches are real entry points."""

  max_progress: float = 0.85
  """Drop spots this far into the skill. Too little left to be worth entering at."""

  clearance_range: tuple[float, float] = (-0.03, 0.05)
  """Metres the lowest part of the robot may sit above the floor.

  The upper bound is what rejects mid-flight. A bridge steers a body it is standing on;
  it cannot put one on a chosen ballistic arc, so an airborne target is unreachable
  however cleanly the skill passes through it.

  The lower bound catches states written into the floor. A solver lets a foot sink a
  little and no further, so a reading well below this is a defect worth seeing rather
  than a state worth aiming at.

  Both bounds measured, not chosen. Over the jump rollouts:

      p0     -0.020
      p50    -0.001     standing, sub-millimetre
      p75    -0.001
      p95    +0.130     airborne
      p100   +0.253

  Grounded and airborne are two separated groups with nothing between them, so anything
  in the gap works as the upper bound. The lower one sits just under the deepest sink a
  contact solver produced."""

  min_dwell_s: float = 0.06
  """Drop spots a rollout crosses in less time than this. Nothing can aim at a target
  that is open for one tick."""

  split: str = "train"
  """Which side of the dataset holdout to measure. The holdout exists for the bridge;
  this is a measurement and 7/8 of the rollouts is plenty. Pass 'eval' to cross-check."""

  device: str = "cuda:0"
  seed: int = 0


def build(cfg: BuildCfg) -> EntryTable:
  """One entry table over every requested skill."""
  data = load_dataset(cfg.path, cfg.device, cfg.split)
  ground = Ground()
  if ground.num_joints != data.num_joints:
    raise SystemExit(
      f"The rollouts hold {data.num_joints}-joint states and this G1 has "
      f"{ground.num_joints}, so they were recorded against a different robot."
    )
  wanted = cfg.skills or data.names
  entries: list[Entry] = []
  for skill in wanted:
    entries += for_skill(data, skill, ground, cfg)
  if not entries:
    raise SystemExit(
      "No entry points survived the filters. Lower --min-coverage or --min-dwell-s, "
      "or check that the dataset holds more than one rollout per skill."
    )
  return EntryTable(entries=tuple(entries), fps=data.fps)


def for_skill(data: Dataset, skill: str, ground: Ground, cfg: BuildCfg) -> list[Entry]:
  """One skill's entry points, best first."""
  rows = data.of((skill,))
  states = canonical(data.states[rows])
  trajectory, frame = data.trajectory[rows], data.frame[rows]

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
    coverage = int(torch.unique(trajectory[members]).numel()) / rollouts
    # The center's own progress, not the median over its members. A skill that comes
    # back to a pose puts both visits in one cluster, and the median of those is a
    # moment nothing was ever at. `frame` and `progress` have to name the same tick,
    # or max_progress filters on a number no state has
    here = float(progress[center])
    seconds = float(dwell[index])
    if coverage < cfg.min_coverage or here > cfg.max_progress:
      continue
    if seconds < cfg.min_dwell_s:
      continue
    pose = states[center].cpu().numpy().astype(np.float32)
    floor = ground.clearance(pose.astype(np.float64))
    low, high = cfg.clearance_range
    if not low <= floor <= high:
      continue
    found.append(
      Entry(
        skill=skill,
        name=f"p{round(here * 100):02d}",
        state=pose,
        frame=int(frame[center]),
        progress=here,
        coverage=coverage,
        spread=float(torch.cdist(feat[members], feat[center : center + 1]).median()),
        clearance=floor,
        dwell_s=seconds,
        hold_s=coverage * seconds,
        share=int(members.numel()) / int(rows.numel()),
      )
    )

  found.sort(key=lambda entry: entry.hold_s, reverse=True)
  print(
    f"[selector] {skill}: {count} clusters over {rollouts} rollouts, {len(found)} kept"
  )
  return rename(found)


def rename(entries: list[Entry]) -> list[Entry]:
  """Make the progress names unique, keeping the order they came in.

  Two clusters can land on the same percentage. Suffixing beats renumbering: the name
  still says where in the skill the entry is, which is the only thing it is for.
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
  print("\n".join(table.lines()))
  table.save(cfg.out)


if __name__ == "__main__":
  main(tyro.cli(BuildCfg, config=mjlab.TYRO_FLAGS))
