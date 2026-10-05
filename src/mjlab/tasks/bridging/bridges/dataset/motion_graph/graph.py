"""Find frames of two different clips that look alike, and pick pairs to stitch.

Only frames the kinematic filter kept are used, so a stitched clip is made of filtered
motion on both sides of its seam.

A frame is described by a heading invariant feature vector: pelvis height, tilt, joint
angles, root and feet velocity, feet position around the pelvis, and foot contacts.
Two frames are a transition when their features are close. Contacts get a large weight, so
a transition never swaps the support foot.
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mjlab.tasks.bridging.bridges.dataset.motion_graph.clips import Corpus
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

CONTACT_HEIGHT = 0.03
"""A sole lower than this is planted, in m."""

# Feature weights. Distances are in metres, so a weight turns its unit into metres
FEET = 3.0
HEIGHT = 3.0
TILT = 1.0
JOINTS = 0.3
ROOT_VEL = 0.3
YAW_RATE = 0.1
FEET_VEL = 0.3
JOINT_VEL = 0.03
CONTACT = 10.0


@dataclass
class Frames:
  """Every valid frame of a corpus, in clip and time order."""

  states: torch.Tensor
  """(N, 13 + 2J)."""
  features: torch.Tensor
  """(N, F)."""
  clip: np.ndarray
  """(N,) index into corpus.clips."""
  category: np.ndarray
  """(N,) index into categories."""
  categories: tuple[str, ...]
  before: np.ndarray
  """(N,) rows of the same valid run before this one."""
  after: np.ndarray
  """(N,) rows of the same valid run after this one."""


def frames(corpus: Corpus, device: str) -> Frames:
  categories = tuple(sorted({c.category for c in corpus.clips}))
  states, feet, feet_vel, clip, frame, category = [], [], [], [], [], []
  for index, c in enumerate(corpus.clips):
    keep = np.flatnonzero(c.valid)
    states.append(c.states[keep])
    feet.append(c.feet[keep])
    feet_vel.append(np.gradient(c.feet, axis=0)[keep] * corpus.fps)
    clip.append(np.full(len(keep), index))
    frame.append(keep)
    category.append(np.full(len(keep), categories.index(c.category)))
  clip_ids, frame_ids = np.concatenate(clip), np.concatenate(frame)

  # A run breaks where the clip changes or a rejected frame was cut out
  rows = np.arange(len(clip_ids))
  breaks = np.flatnonzero((np.diff(clip_ids) != 0) | (np.diff(frame_ids) != 1)) + 1
  firsts = np.concatenate([[0], breaks])
  ends = np.append(breaks, len(clip_ids))
  run = np.searchsorted(ends, rows, side="right")

  def tensor(parts: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.concatenate(parts)).float().to(device)

  s = tensor(states)
  return Frames(
    states=s,
    features=features(s, tensor(feet), tensor(feet_vel)),
    clip=clip_ids,
    category=np.concatenate(category),
    categories=categories,
    before=rows - firsts[run],
    after=ends[run] - 1 - rows,
  )


def features(
  states: torch.Tensor, feet: torch.Tensor, feet_vel: torch.Tensor
) -> torch.Tensor:
  joints = (states.shape[1] - 13) // 2
  pos, quat = states[:, 0:3], states[:, 3:7]
  heading = yaw_quat(quat)
  heading2 = heading[:, None].expand(-1, 2, -1)
  down = torch.tensor([0.0, 0.0, -1.0], device=states.device).expand(len(states), 3)
  planted = (feet[..., 2] < CONTACT_HEIGHT).float()
  return torch.cat(
    [
      HEIGHT * pos[:, 2:3],
      TILT * quat_apply_inverse(quat, down),
      JOINTS * states[:, 13 : 13 + joints],
      JOINT_VEL * states[:, 13 + joints :],
      ROOT_VEL * quat_apply_inverse(heading, states[:, 7:10]),
      YAW_RATE * states[:, 12:13],
      FEET * quat_apply_inverse(heading2, feet - pos[:, None]).flatten(1),
      FEET_VEL * quat_apply_inverse(heading2, feet_vel).flatten(1),
      CONTACT * planted,
    ],
    dim=-1,
  )


@dataclass
class Transitions:
  """Jumps from row source of one clip to row target of another."""

  source: np.ndarray
  target: np.ndarray
  distance: np.ndarray


def transitions(
  f: Frames,
  max_distance: float,
  neighbors: int,
  stride: int,
  before: int,
  after: int,
  chunk: int = 512,
) -> Transitions:
  """The closest neighbors in other clips of every stride-th frame, under max_distance.

  A source needs before valid frames up to it, a target after valid frames from it.
  """
  sources = np.flatnonzero(f.before >= before - 1)[::stride]
  targets = np.flatnonzero(f.after >= after - 1)
  device = f.features.device
  target_features = f.features[targets]
  target_clip = torch.from_numpy(f.clip[targets]).to(device)

  found_source, found_target, found_distance = [], [], []
  for begin in range(0, len(sources), chunk):
    rows = sources[begin : begin + chunk]
    distance = torch.cdist(f.features[rows], target_features)
    clip = torch.from_numpy(f.clip[rows]).to(device)
    distance.masked_fill_(clip[:, None] == target_clip, torch.inf)
    value, index = distance.topk(neighbors, dim=1, largest=False)
    keep = (value < max_distance).cpu().numpy()
    found_source.append(np.repeat(rows, neighbors).reshape(-1, neighbors)[keep])
    found_target.append(targets[index.cpu().numpy()][keep])
    found_distance.append(value.cpu().numpy()[keep])

  return Transitions(
    source=np.concatenate(found_source),
    target=np.concatenate(found_target),
    distance=np.concatenate(found_distance),
  )


def pair_weights(f: Frames, t: Transitions) -> np.ndarray:
  """Probability of picking each transition, every category pair equally likely.

  Without this nearly every stitch is walk to walk, the bulk of the corpus.
  """
  pair = f.category[t.source] * len(f.categories) + f.category[t.target]
  share = 1 / np.bincount(pair)[pair]
  return share / share.sum()
