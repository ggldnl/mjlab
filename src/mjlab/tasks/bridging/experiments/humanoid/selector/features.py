"""What situation the robot is in, as a handful of numbers.

Two frames whose numbers are close mean the robot is in the same situation. That is the
whole job: give the profiler something to compare frames by, so it can notice that a skill
keeps passing through the same few situations and say what they are.

Position and heading are left out on purpose. A crouch is a crouch at either end of the
field, and a left-planted stride is one whether the robot heads north or south. Every
feature is a height, a joint angle, or a vector rotated into the robot's own heading frame,
so none of them change when the robot is picked up and put down facing another way.

Left and right stay apart. A kick loaded on the left foot and the same kick loaded on the
right are two situations, and the bridge wants both offered so it can aim at whichever is
cheaper to reach.

The feet are the ankle roll links, not the sole contacts, so a foot height here is the
height of the ankle and not a clearance. Good enough to tell stance from swing, which is
all it is asked to do, and it avoids depending on a contact sensor that not every skill's
environment carries.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
KNEES = ("left_knee_joint", "right_knee_joint")

FEATURES = (
  "root_height",
  "root_pitch",
  "root_roll",
  "vel_forward",
  "vel_lateral",
  "yaw_rate",
  "left_foot_height",
  "right_foot_height",
  "left_foot_rise",
  "right_foot_rise",
  "left_foot_forward",
  "right_foot_forward",
  "left_foot_lateral",
  "right_foot_lateral",
  "left_knee",
  "right_knee",
)


class Describer:
  """Turns a live robot into one row of `FEATURES` per environment.

  Body and joint lookups happen once, in the constructor. They are name matches against the
  spec and doing them every control step would cost more than the features themselves.
  """

  def __init__(self, robot: Entity) -> None:
    feet, _ = robot.find_bodies(list(FEET), preserve_order=True)
    knees, _ = robot.find_joints(list(KNEES), preserve_order=True)
    self._feet = feet
    self._knees = knees

  def __call__(self, robot: Entity) -> torch.Tensor:
    """(N, len(FEATURES))."""
    data = robot.data
    root_pos = data.root_link_pos_w
    quat = data.root_link_quat_w
    heading = yaw_quat(quat)

    w, x, y, z = quat.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))

    vel = quat_apply_inverse(heading, data.root_link_lin_vel_w)

    # One heading quaternion per foot, so the offsets come back in the robot's own frame
    feet_quat = heading.unsqueeze(1).expand(-1, len(self._feet), -1)
    feet_pos = data.body_link_pos_w[:, self._feet]
    offset = quat_apply_inverse(feet_quat, feet_pos - root_pos.unsqueeze(1))
    rise = data.body_link_lin_vel_w[:, self._feet, 2]
    knees = data.joint_pos[:, self._knees]

    return torch.stack(
      [
        root_pos[:, 2],
        pitch,
        roll,
        vel[:, 0],
        vel[:, 1],
        data.root_link_ang_vel_w[:, 2],
        feet_pos[:, 0, 2],
        feet_pos[:, 1, 2],
        rise[:, 0],
        rise[:, 1],
        offset[:, 0, 0],
        offset[:, 1, 0],
        offset[:, 0, 1],
        offset[:, 1, 1],
        knees[:, 0],
        knees[:, 1],
      ],
      dim=-1,
    )
