"""One skill's reference clips, loaded once and read in batch.

`dataset.py` writes a folder of normalized clips per skill; this is what the env reads
them through. Every clip of a skill is concatenated into one flat set of tensors with a
per-clip offset, so looking up "clip c, frame f" for a thousand envs at once is a single
gather rather than a Python loop over envs.

Two things this deliberately does *not* carry, because `dataset.py` already removed them:
absolute position and heading. What comes back is joint state, root height and
orientation, root velocity in the root's own yaw frame, end-effector offsets in that same
frame, foot contacts, and the goal. A policy reading only these cannot tell where in the
world it is or which way it is facing, which is the point: "walk at this speed toward
that goal" should mean the same thing in every corner of the plane.

Frame lookups clamp to the clip's last frame rather than wrapping. A clip that runs out
is an episode that is over (the env terminates on it), and clamping means the reference
holds its final pose for the step or two between the two events instead of teleporting
into a neighbouring clip's opening frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ReferenceFrames:
  """What the reference says the robot should be doing, for a batch of (clip, frame)."""

  joint_pos: torch.Tensor
  """(N, J)"""
  joint_vel: torch.Tensor
  """(N, J)"""
  root_height: torch.Tensor
  """(N,)"""
  root_quat: torch.Tensor
  """(N, 4) wxyz, canonically placed: roll and pitch are absolute, yaw is relative to the
  clip's own first frame."""
  root_lin_vel_b: torch.Tensor
  """(N, 3) in the root's own yaw frame."""
  root_ang_vel_b: torch.Tensor
  """(N, 3) in the root's own yaw frame."""
  ee_offsets: torch.Tensor
  """(N, E, 3) end-effector positions relative to the root, in the root's yaw frame."""
  contacts: torch.Tensor
  """(N, 2) per-foot ground contact, as 0/1."""


class ClipLibrary:
  """Every reference clip of one skill, flattened for batched lookup."""

  def __init__(self, clip_dir: str | Path, device: str) -> None:
    paths = sorted(Path(clip_dir).glob("*.npz"))
    if not paths:
      raise FileNotFoundError(
        f"No reference clips in {clip_dir}. Build them first with "
        "`uv run python -m mjlab.tasks.skills.experiments.parkour.dataset`."
      )

    self.device = device
    self.paths = paths
    self.num_clips = len(paths)

    clips = [np.load(p, allow_pickle=False) for p in paths]
    lengths = [int(c["joint_pos"].shape[0]) for c in clips]

    fps = {float(c["fps"][0]) for c in clips}
    if len(fps) != 1:
      raise ValueError(f"Clips in {clip_dir} disagree on fps: {sorted(fps)}.")
    self.fps = fps.pop()

    channels = {tuple(c["goal_channels"].tolist()) for c in clips}
    if len(channels) != 1:
      raise ValueError(f"Clips in {clip_dir} disagree on goal channels: {channels}.")
    self.goal_channels: tuple[str, ...] = channels.pop()

    self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    self.offsets = torch.tensor(offsets, dtype=torch.long, device=device)
    self.total_frames = int(sum(lengths))

    def stack(field: str) -> torch.Tensor:
      return torch.tensor(
        np.concatenate([c[field] for c in clips], axis=0),
        dtype=torch.float32,
        device=device,
      )

    self.joint_pos = stack("joint_pos")
    self.joint_vel = stack("joint_vel")
    self.root_height = stack("root_height")
    self.root_quat = stack("root_quat_local")
    self.root_lin_vel_b = stack("root_lin_vel_b")
    self.root_ang_vel_b = stack("root_ang_vel_b")
    self.ee_offsets = stack("ee_offsets")
    self.contacts = stack("contacts")
    self.goal = stack("goal")

    self.joint_dim = int(self.joint_pos.shape[1])
    self.goal_dim = int(self.goal.shape[1])
    self.num_ee = int(self.ee_offsets.shape[1])

  def __len__(self) -> int:
    return self.num_clips

  @property
  def duration(self) -> float:
    """Total reference motion, in seconds."""
    return self.total_frames / self.fps

  def describe(self) -> str:
    return (
      f"{self.num_clips} clip(s), {self.duration:.1f} s at {self.fps:g} fps, "
      f"goal {list(self.goal_channels)}"
    )

  def sample_clips(self, num: int) -> torch.Tensor:
    """`num` clip ids, uniform over clips.

    Uniform over clips rather than over frames, so a long cut does not crowd out the
    short ones. The jump set is made of short cuts by construction and would otherwise
    barely be sampled next to a six-second walk.
    """
    return torch.randint(0, self.num_clips, (num,), device=self.device)

  def sample_start_frames(
    self, clip_ids: torch.Tensor, max_start_fraction: float
  ) -> torch.Tensor:
    """A start frame per clip, uniform over the first `max_start_fraction` of it.

    Starting anywhere is the randomization that stops the policy from only ever seeing a
    clip's opening pose. It is capped short of the end because an episode beginning two
    frames from the last one has no motion left to track and no goal left to realize --
    for jump especially, where starting after the landing would ask for a jump that has
    already happened.
    """
    span = (self.lengths[clip_ids].float() * max_start_fraction).long().clamp_min(1)
    return (torch.rand(len(clip_ids), device=self.device) * span).long()

  def _flat_index(self, clip_ids: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """Clip-local frames to indices into the flattened tensors, clamped to each clip."""
    clamped = torch.minimum(frames.clamp_min(0), self.lengths[clip_ids] - 1)
    return self.offsets[clip_ids] + clamped

  def goal_at(self, clip_ids: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """(N, G) the goal each clip states from `frames` onward."""
    return self.goal[self._flat_index(clip_ids, frames)]

  def frames_at(self, clip_ids: torch.Tensor, frames: torch.Tensor) -> ReferenceFrames:
    """The full reference state at (clip, frame), clamped to each clip's last frame."""
    index = self._flat_index(clip_ids, frames)
    return ReferenceFrames(
      joint_pos=self.joint_pos[index],
      joint_vel=self.joint_vel[index],
      root_height=self.root_height[index],
      root_quat=self.root_quat[index],
      root_lin_vel_b=self.root_lin_vel_b[index],
      root_ang_vel_b=self.root_ang_vel_b[index],
      ee_offsets=self.ee_offsets[index],
      contacts=self.contacts[index],
    )

  def joint_pos_at(self, clip_ids: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """(N, J) just the reference joint positions, for the cheap future-frame lookups."""
    return self.joint_pos[self._flat_index(clip_ids, frames)]
