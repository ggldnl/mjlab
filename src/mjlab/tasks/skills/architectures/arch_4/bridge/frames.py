"""How a window is written down so a network can read it and answer in the same terms.

The corpus stores world coordinates: where in the capture volume the subject was and
which way they were facing. A network trained on that would learn the capture volume. So
everything here is expressed in the frame of the **hand-off**, the last context frame
before the hole:

- the root's position becomes an offset from where the body was at the hand-off, rotated
  into the heading it had there;
- the root's orientation becomes a rotation relative to that heading;
- joint angles are already relative to the body and are left alone.

Two things fall out of that choice, and both are the point.

Moving a whole clip elsewhere in the world, or turning it around, changes none of the
numbers. A walk recorded going north through the origin and a walk going east thirty
metres away are the same question, which is what lets a corpus of a few hours stand in
for every place the robot might be.

At the same time, *how far the body has to travel is still in there*. The after-context
does not say "be moving at 1.5 m/s"; it says "be 1.4 m ahead of where you handed off,
turned 20 degrees left, moving at 1.5 m/s". Filling the hole with plausible motion that
ends up somewhere else scores badly, which is exactly the failure this architecture
exists to prevent.

Rotations are carried as the first two columns of the rotation matrix rather than as a
quaternion. A quaternion has two representations for every rotation and a network that
regresses one will average the two; the six-number form is unique and continuous, so the
error surface has no seam in it.

A pose is therefore:

    offset      (3)  root position relative to the hand-off, in its heading frame
    rotation    (6)  root orientation relative to that heading
    joint_pos   (J)

Velocities are absent on purpose. The network reads whole stretches of consecutive
frames, so speed is in what it is given; and anything it produces has velocities that
follow from the positions it produced, which cannot then disagree with them.
"""

from __future__ import annotations

import torch

from mjlab.tasks.skills.architectures.arch_4.bridge.dataset import ROOT_STATE_DIM
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  normalize,
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_from_matrix,
  quat_mul,
  yaw_quat,
)

# Root channels of a pose: the offset and the two rotation columns.
POSE_ROOT_DIM = 9


def pose_dim(num_joints: int) -> int:
  return POSE_ROOT_DIM + num_joints


def rot6d_from_quat(quat: torch.Tensor) -> torch.Tensor:
  """The first two columns of a rotation matrix, flattened. (..., 4) -> (..., 6)."""
  matrix = matrix_from_quat(quat)
  # Transposed before flattening, so the six numbers are the first column followed by the
  # second. Reshaping the (3, 2) block directly interleaves them instead, which still
  # round-trips through a network but means the first three numbers are not a vector and
  # the Gram-Schmidt in `quat_from_rot6d` is orthogonalizing nonsense.
  return matrix[..., :, :2].transpose(-1, -2).reshape(*quat.shape[:-1], 6)


def quat_from_rot6d(rot6d: torch.Tensor) -> torch.Tensor:
  """Back to a wxyz quaternion, re-orthonormalizing on the way. (..., 6) -> (..., 4).

  What a network emits is six free numbers, not two orthonormal columns, so the first is
  normalized, the second has its component along the first removed, and the third is
  their cross product. Any six numbers give a valid rotation, which is the property that
  makes this worth using.
  """
  first = normalize(rot6d[..., 0:3])
  second = rot6d[..., 3:6]
  second = normalize(second - (first * second).sum(-1, keepdim=True) * first)
  third = torch.cross(first, second, dim=-1)
  matrix = torch.stack([first, second, third], dim=-1)

  # Through quat_from_matrix rather than the one-line trace formula. That formula divides
  # by the scalar part, which vanishes at a half turn, and a body that has turned around
  # is an ordinary thing for this corpus to contain.
  return quat_from_matrix(matrix)


def anchor_of(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """The hand-off's position and heading, from a raw state. (..., 13 + 2J)."""
  return state[..., 0:3], yaw_quat(state[..., 3:7])


def encode(
  states: torch.Tensor, anchor_pos: torch.Tensor, anchor_yaw: torch.Tensor
) -> torch.Tensor:
  """Raw states to poses in the hand-off frame. (B, T, 13 + 2J) -> (B, T, 9 + J).

  `anchor_pos` and `anchor_yaw` are (B, 3) and (B, 4), broadcast over the time axis.
  """
  pos = states[..., 0:3]
  quat = states[..., 3:7]
  joints = states[..., ROOT_STATE_DIM : ROOT_STATE_DIM + _num_joints(states)]

  yaw = anchor_yaw.unsqueeze(-2).expand(*quat.shape[:-1], 4)
  offset = quat_apply_inverse(yaw, pos - anchor_pos.unsqueeze(-2))
  relative = quat_mul(quat_conjugate(yaw), quat)
  return torch.cat([offset, rot6d_from_quat(relative), joints], dim=-1)


def decode(
  poses: torch.Tensor, anchor_pos: torch.Tensor, anchor_yaw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Poses back to world root position, world root orientation and joint angles."""
  offset = poses[..., 0:3]
  relative = quat_from_rot6d(poses[..., 3:POSE_ROOT_DIM])
  joints = poses[..., POSE_ROOT_DIM:]

  yaw = anchor_yaw.unsqueeze(-2).expand(*relative.shape[:-1], 4)
  pos = quat_apply(yaw, offset) + anchor_pos.unsqueeze(-2)
  return pos, quat_mul(yaw, relative), joints


def blend(start: torch.Tensor, end: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
  """The naive in-between: straight interpolation from one pose to another.

  The baseline every learned model has to beat, and, because the model predicts a
  correction to this rather than a pose outright, also its starting point. A model that
  learns nothing scores exactly as well as interpolation instead of producing noise, and
  every gradient it does receive is spent on the part interpolation gets wrong: the
  weight shift, the foot that has to be planted, the fact that a body crossing a metre in
  half a second takes a step rather than gliding.

  `alpha` is (B, Q, 1) and the poses are (B, 1, P). Positions and joint angles are
  interpolated straight; the rotation columns are interpolated and left for
  `quat_from_rot6d` to re-orthonormalize, which is a shortest-arc slerp in all but name
  over the angles that occur between two frames of a real motion.
  """
  return start + (end - start) * alpha


def _num_joints(states: torch.Tensor) -> int:
  return (states.shape[-1] - ROOT_STATE_DIM) // 2
