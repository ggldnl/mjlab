"""One description of "what the robot is doing right now", written two ways.

Everything in arch_4 is expressed in this vector and nothing else: the two context
windows the bridge reads, the reference it is rewarded against, and the profiles it
aims at. That is the whole reason the architecture can be trained on motion capture
and then used on policies -- both sides have to produce the same numbers, and this is
the only place that is arranged.

A frame is:

    height        (1)  root height above the ground
    gravity       (3)  the gravity direction in the root's own frame, i.e. its tilt
    lin_vel       (3)  root linear velocity, in the root frame
    ang_vel       (3)  root angular velocity, in the root frame
    joint_pos     (J)
    joint_vel     (J)

What is deliberately absent is where the robot is and which way it faces. Every
channel is either a height or a body-frame quantity, so the vector is unchanged by
moving the motion somewhere else in the world or turning it around. That is what lets
a clip recorded walking north through the origin be compared against a robot running
east down a corridor thirty metres in: the two are the same motion, and this
representation says so.

It is also the reason arch_4 needs no anchoring machinery of the kind the jump command
carries (`anchor_to_robot` and the rest). A bridge is judged on posture and velocity,
never on absolute position, which is the same choice the parkour experiment already
made for its state view -- see view.py and arena.py.

Two readers, one layout:

- `robot_frame` reads the live simulator through an mjlab `Entity`.
- `clip_frames` reads a converted motion npz (the format `csv_to_npz` and the parkour
  dataset script write): world-frame root pose and velocities, which it rotates into
  the root frame here.

The `Groups` slices at the bottom are what the tracking reward is built out of, so a
change to the layout above moves the reward with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply_inverse

# Channels that come before the joints: height, gravity, linear and angular velocity.
ROOT_CHANNELS = 10


def frame_dim(num_joints: int) -> int:
  """Width of one frame for a robot with `num_joints` actuated joints."""
  return ROOT_CHANNELS + 2 * num_joints


@dataclass(frozen=True)
class Groups:
  """Where each physical quantity sits in a frame, for the reward to read.

  The tracking reward is a set of exponential kernels, one per group, because the
  groups have wildly different units: a tenth of a radian of joint error and a tenth
  of a metre per second of velocity error are not the same mistake and must not share
  a length scale. Splitting them is also what makes the training log readable, since
  each term can be reported on its own.
  """

  num_joints: int

  @property
  def posture(self) -> slice:
    """Height and tilt: the two things that say the robot is still standing."""
    return slice(0, 4)

  @property
  def lin_vel(self) -> slice:
    return slice(4, 7)

  @property
  def ang_vel(self) -> slice:
    return slice(7, 10)

  @property
  def joint_pos(self) -> slice:
    return slice(ROOT_CHANNELS, ROOT_CHANNELS + self.num_joints)

  @property
  def joint_vel(self) -> slice:
    return slice(ROOT_CHANNELS + self.num_joints, frame_dim(self.num_joints))

  def named(self) -> tuple[tuple[str, slice], ...]:
    return (
      ("posture", self.posture),
      ("lin_vel", self.lin_vel),
      ("ang_vel", self.ang_vel),
      ("joint_pos", self.joint_pos),
      ("joint_vel", self.joint_vel),
    )


def robot_frame(entity: Entity, ground_height: torch.Tensor | float = 0.0):
  """The frame the robot is in right now, shaped (num_envs, frame_dim).

  `ground_height` is subtracted from the root height so that the number means "how
  high above the floor" rather than "how high in world coordinates". On the flat
  arenas here that is zero, but an experiment whose environments sit at different
  heights has to pass its origins or every frame it produces is offset from every
  frame in the corpus.
  """
  data = entity.data
  height = data.root_link_pos_w[:, 2:3]
  if isinstance(ground_height, torch.Tensor):
    height = height - ground_height.reshape(-1, 1)
  elif ground_height:
    height = height - ground_height
  return torch.cat(
    [
      height,
      data.projected_gravity_b,
      data.root_link_lin_vel_b,
      data.root_link_ang_vel_b,
      data.joint_pos,
      data.joint_vel,
    ],
    dim=-1,
  )


def clip_frames(
  root_pos_w: torch.Tensor,
  root_quat_w: torch.Tensor,
  root_lin_vel_w: torch.Tensor,
  root_ang_vel_w: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
) -> torch.Tensor:
  """The same frame, for a recorded motion. Inputs are (T, ...) world-frame arrays.

  The gravity channel is computed rather than read: a clip stores an orientation, and
  what the robot observes is that orientation applied to gravity. Producing it here
  from the quaternion is what makes the recorded number and the sensor's number the
  same quantity.
  """
  gravity_w = torch.zeros_like(root_lin_vel_w)
  gravity_w[:, 2] = -1.0
  return torch.cat(
    [
      root_pos_w[:, 2:3],
      quat_apply_inverse(root_quat_w, gravity_w),
      quat_apply_inverse(root_quat_w, root_lin_vel_w),
      quat_apply_inverse(root_quat_w, root_ang_vel_w),
      joint_pos,
      joint_vel,
    ],
    dim=-1,
  )
