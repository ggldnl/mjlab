"""What the climb adds to the tracking MDP: the obstacle, as the policy sees it.

Everything else comes from the goal conditioned jump's terms, re-exported below, so a climb
is that MDP with one observation bolted on.

The obstacle is read out of the scene rather than out of the config, so the same term reads
a box that a reset event moved. With one fixed box that costs nothing and it is the seam a
variable height task needs: the term already reports the size, so raising the box is a
change to the scene rather than a change to the observation, and a policy trained here has
the input width its successor needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.mdp import *  # noqa: F401, F403
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

ROBOT = "robot"
BOX = "box"


def _heading_quat(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The robot's yaw only orientation, shaped (num_envs, 4).

  Yaw only, never the full root orientation. A robot pitched forward onto a box top swings
  a full base frame vector with it, which would report the box as moving when only the
  robot leaned.
  """
  robot: Entity = env.scene[ROBOT]
  return yaw_quat(robot.data.root_link_quat_w)


def box_pose_b(env: ManagerBasedRlEnv, half_size: tuple[float, float, float]):
  """The obstacle from the robot: where it is, which way it faces, how big it is.

  Eight numbers, shaped (num_envs, 8): the centre in the robot's heading frame, the box's
  yaw relative to that heading as a cosine and sine pair, and the three half extents.

  The angle is a pair rather than a scalar because a box and the same box turned by pi are
  the same obstacle, and a scalar yaw makes that discontinuity land somewhere in the middle
  of the approach.

  The half extents are constant while there is one box, so with a single obstacle this term
  is three live numbers and five fixed ones. They are given anyway: the point of the term is
  that a policy reads the obstacle instead of assuming it, and a fixed number the policy
  ignores costs one input, while a missing one costs a retrain when the box changes.

  Args:
    half_size: The box's half extents, in metres. Taken from the config rather than the
      model, because a compiled geom size is not something a term can ask for cheaply and
      the environment already knows it.
  """
  box: Entity = env.scene[BOX]
  heading = _heading_quat(env)

  robot: Entity = env.scene[ROBOT]
  offset = quat_apply_inverse(
    heading, box.data.root_link_pos_w - robot.data.root_link_pos_w
  )

  # Relative yaw straight off the two quaternions, both yaw only, so their product is a
  # rotation about z and its angle comes out of the w and z components alone
  box_yaw = yaw_quat(box.data.root_link_quat_w)
  relative = 2.0 * torch.atan2(
    heading[:, 0] * box_yaw[:, 3] - heading[:, 3] * box_yaw[:, 0],
    heading[:, 0] * box_yaw[:, 0] + heading[:, 3] * box_yaw[:, 3],
  )

  size = torch.tensor(half_size, device=offset.device, dtype=offset.dtype)
  return torch.cat(
    [
      offset,
      torch.cos(relative).unsqueeze(-1),
      torch.sin(relative).unsqueeze(-1),
      size.expand(env.num_envs, 3),
    ],
    dim=-1,
  )
