"""How a robot state is written down, compared and scaled.

Shared by build.py, which clusters states, and query.py, which measures how far one is from
another.

A state is a dataset row: root position, orientation, both root velocities, then joint
angles and joint rates, (13 + 2J,).

    canonical     drop where on the floor and which way round
    channel_gap   how far apart two states are, per channel, in natural units
    features      one vector per state whose euclidean distance is the clustering metric
"""

from __future__ import annotations

import numpy as np
import torch

from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.dataset import (
  ROOT_STATE_DIM,
)
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_conjugate,
  quat_error_magnitude,
  quat_mul,
  yaw_quat,
)

CHANNELS = ("root_z", "tilt", "lin_vel", "ang_vel", "joint_pos", "joint_vel")
"""The six ways two states differ. Ground position and heading are not among them: see
canonical for why."""

SCALE = (0.05, 0.05, 0.15, 0.30, 0.10, 1.50)
"""Metres, radians, m/s, rad/s, radians, rad/s. One tolerance's worth of wrong.

The bridge's arrival tolerances, so a distance of 1.0 means about as far off as a
hand-over is allowed to be. Copied rather than imported: the bridge splits arms from
legs and needs joint names to do it, and a dataset does not record joint names.
"""


def canonical(states: torch.Tensor) -> torch.Tensor:
  """Drop where on the floor and which way round. Same layout in and out.

  Ground position goes to zero and yaw is removed, so two walk states a metre apart
  facing different ways come out identical. Both velocities are rotated with the pose,
  or they would still point the way the robot happened to be facing when recorded.

  Two reasons this is the right frame. Clustering: without it, states group by where on
  the floor they happened rather than by what the robot was doing. Aiming: whoever
  places a target picks the ground position and the heading, so those are free
  parameters and never a reason one entry is harder to reach than another.
  """
  quat = states[:, 3:7]
  heading = yaw_quat(quat)
  out = states.clone()
  out[:, 0:2] = 0.0
  out[:, 3:7] = quat_mul(quat_conjugate(heading), quat)
  out[:, 7:10] = quat_apply_inverse(heading, states[:, 7:10])
  out[:, 10:ROOT_STATE_DIM] = quat_apply_inverse(heading, states[:, 10:ROOT_STATE_DIM])
  return out


def channel_gap(here: torch.Tensor, there: torch.Tensor) -> torch.Tensor:
  """How far apart two states are, per channel. (N, 13 + 2J) x2 -> (N, 6).

  Natural units, in CHANNELS order: metres, radians, m/s, rad/s, radians, rad/s. Kept
  apart rather than summed because the units do not mix and because whichever channel
  is worst is the useful thing to report.

  Joints collapse to the worst one, not the mean. A hand-over is wrong if any joint is
  wrong, and averaging over 29 of them hides one that is badly out.
  """
  num_joints = (here.shape[1] - ROOT_STATE_DIM) // 2
  q = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + num_joints)
  qd = slice(ROOT_STATE_DIM + num_joints, ROOT_STATE_DIM + 2 * num_joints)
  return torch.stack(
    [
      (here[:, 2] - there[:, 2]).abs(),
      quat_error_magnitude(here[:, 3:7], there[:, 3:7]),
      (here[:, 7:10] - there[:, 7:10]).norm(dim=-1),
      (here[:, 10:ROOT_STATE_DIM] - there[:, 10:ROOT_STATE_DIM]).norm(dim=-1),
      (here[:, q] - there[:, q]).abs().amax(dim=-1),
      (here[:, qd] - there[:, qd]).abs().amax(dim=-1),
    ],
    dim=-1,
  )


def features(states: torch.Tensor) -> torch.Tensor:
  """Canonical states as vectors whose euclidean distance is the clustering metric.

  Each channel is divided by its scale and by the square root of its width, so a 29-number
  joint block does not outweigh a 3-number velocity just by being wider.

  Tilt contributes the xyz of its quaternion, doubled, with w forced positive. For the
  tilts a standing robot has that is the tilt angle in radians, and forcing the sign stops
  one rotation being written two ways.

  Not the same as channel_gap, on purpose: this one has to be a point in a space a
  clustering algorithm can take thousands of distances in, that one has to be readable per
  channel. They agree on scale and on which channels exist, which is as much as they need.
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
