"""Where the robot has to stand before a traversal skill will start cleanly.

One question per obstacle, solved once, before the run starts:

    solve(env, skill, obstacle, pos, yaw, frame, cfg) -> Approach

Two kinds of obstacle and they are not the same problem.

A hurdle is a distance. Stand hurdle_takeoff back from the near face along the hurdle's own
normal, facing it.

A box is a solve. The climb was retargeted from a human motion together with its obstacle,
so the clip is only physical against that box at that pose. The pose to arrive in is
whatever puts the reference's own box exactly on the real one, and place finds it: anchor,
measure where the clip's box landed, correct, anchor again.

Three angles come out of a solve and conflating any two of them is the bug this module
exists to prevent:

    yaw         the heading the robot holds on arrival. What the bridge target faces
    anchor_yaw  the clip's direction of travel, which is what anchor_to_robot reads. Off
                yaw by the pelvis twist a run-up carries, twelve to twenty degrees
    hold_xy     where the walk stops, one hold_back back along the direction of travel

The walk stops short on purpose. It arrives roughly, and stopped, and the bridge covers the
last hold_back metres and lands the robot in the full entry state. Driving the walk at the
arrival pose instead puts a still moving robot on top of it, which is not a state any entry
sits near.

Run

Nothing here runs on its own. See controller.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.arena import command_name
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.bridge import quat_from_yaw
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.course import (
  BOX,
  ApproachCfg,
  Obstacle,
  climb_box,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.mdp.commands import (
  JumpCommand,
)

MOTION = "motion"
"""What a clip tracker calls its reference command, before the arena namespaces it."""


##
# Angles.
##


def wrap(angle: torch.Tensor) -> torch.Tensor:
  """An angle folded into (-pi, pi]."""
  return torch.atan2(torch.sin(angle), torch.cos(angle))


def rotate(vec: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  """(N, 2) turned by (N,) radians about z."""
  cos, sin = torch.cos(yaw), torch.sin(yaw)
  return torch.stack(
    [vec[:, 0] * cos - vec[:, 1] * sin, vec[:, 0] * sin + vec[:, 1] * cos], dim=-1
  )


def yaw_of(quat: torch.Tensor) -> torch.Tensor:
  """The yaw of a quaternion, (N, 4) wxyz -> (N,)."""
  return 2.0 * torch.atan2(quat[:, 3], quat[:, 0])


def lean_of(quat: torch.Tensor) -> torch.Tensor:
  """How far a body's own z axis is off the world's, (N, 4) wxyz -> (N,) radians.

  A quaternion's rotation of (0, 0, 1) has z component 1 - 2(x^2 + y^2), which is the cosine
  of the angle wanted.
  """
  cosine = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
  return torch.acos(torch.clamp(cosine, -1.0, 1.0))


##
# The answer.
##


@dataclass(frozen=True)
class Approach:
  """Everything one obstacle asks of the robot before its skill takes over."""

  xy: torch.Tensor
  """(N, 2). Where the robot ends up, which is what the bridge is aimed at."""
  yaw: torch.Tensor
  """(N,). The heading it holds there."""
  anchor_yaw: torch.Tensor
  """(N,). The clip's direction of travel, which is what pins the reference."""
  frame: int
  """The clip frame the hand-over resumes at. Everything above is solved at this frame."""
  hold_xy: torch.Tensor
  """(N, 2). Where the walk stops, hold_back short of xy."""

  @property
  def hold(self) -> tuple[float, float]:
    """The hold point of the first environment, for steering and printing."""
    return float(self.hold_xy[0, 0]), float(self.hold_xy[0, 1])

  @property
  def face(self) -> float:
    """The arrival heading of the first environment."""
    return float(self.yaw[0])

  def row(self) -> str:
    """The hold point, the heading to hold there, and the pose the bridge then lands on."""
    x, y = self.hold
    return (
      f"{x:.2f}, {y:+.2f} at {math.degrees(self.face):+.0f} deg "
      f"-> {float(self.xy[0, 0]):.2f}, {float(self.xy[0, 1]):+.2f}"
    )


##
# Solving.
##


def place(
  env: ManagerBasedRlEnv,
  skill: str,
  frame: int,
  want_xy: torch.Tensor,
  want_yaw: torch.Tensor,
  obstacle: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Pin a clip tracker's reference, then report the pose the robot has to arrive in.

  anchor_to_robot reads its quaternion as the clip's direction of travel, not as the
  robot's heading, and a clip is canonicalized to travel along its own +x while the pelvis
  spends the run-up twelve to twenty degrees off that. So what goes in is not what comes
  back: the heading to steer at is what the reference itself holds at the entry frame,
  which is read off the placement rather than predicted.

  obstacle turns this from a placement into a solve, and the climb needs it. Anchoring maps
  the pose given here to the clip's placement affinely, so correcting the angle and then the
  position lands it exactly. The loop runs more than once because the angle correction moves
  the position too, and it always ends on an anchor: correcting without re-anchoring leaves
  the placement one correction stale.
  """
  command = env.command_manager.get_term(command_name(skill, MOTION))
  assert isinstance(command, JumpCommand)
  env_ids = torch.arange(env.num_envs, device=env.device)
  origin = env.scene.env_origins
  at_xy, at_yaw = want_xy.clone(), want_yaw.clone()
  box = climb_box()

  for _ in range(3 if obstacle is not None else 1):
    at_pos = torch.cat([at_xy, origin[:, 2:3]], dim=-1)
    command.anchor_to_robot(
      env_ids, start_frame=frame, at_pos=at_pos, at_quat=quat_from_yaw(at_yaw)
    )
    if obstacle is None:
      break
    # Where the clip's own box ended up, from the anchor the call just wrote
    pos, yaw = obstacle
    clip_xy = torch.tensor(
      [box.pos[0], box.pos[1]], device=at_xy.device, dtype=at_xy.dtype
    ).expand(at_xy.shape[0], 2)
    here_xy = rotate(clip_xy, command.anchor_yaw) + command.anchor_pos + origin[:, 0:2]
    off_yaw = wrap(yaw - (box.yaw + command.anchor_yaw))
    off_xy = pos[:, 0:2] - here_xy
    if float(off_xy.norm(dim=-1).max()) < 1e-4 and float(off_yaw.abs().max()) < 1e-4:
      break
    at_yaw = at_yaw + off_yaw
    at_xy = at_xy + off_xy

  return command.body_pos_w[:, 0, 0:2], yaw_of(command.body_quat_w[:, 0]), at_yaw


def guess(
  obstacle: Obstacle, pos: torch.Tensor, yaw: torch.Tensor, cfg: ApproachCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  """A first pose to start the solve from, by kind.

  For a hurdle this is the answer: square on to the near face, one take-off back. For a box
  it is only somewhere sensible to start, because the clip carries its own obstacle and
  place is what lines the two up.
  """
  if obstacle.kind == BOX:
    box = climb_box()
    offset = torch.tensor(
      [box.pos[0], box.pos[1]], device=pos.device, dtype=pos.dtype
    ).expand(pos.shape[0], 2)
    robot_yaw = wrap(yaw - box.yaw)
    return pos[:, 0:2] - rotate(offset, robot_yaw), robot_yaw

  back = obstacle.length / 2.0 + cfg.hurdle_takeoff
  offset = torch.tensor([back, 0.0], device=pos.device, dtype=pos.dtype).expand(
    pos.shape[0], 2
  )
  return pos[:, 0:2] - rotate(offset, yaw), yaw


def solve(
  env: ManagerBasedRlEnv,
  skill: str,
  obstacle: Obstacle,
  pos: torch.Tensor,
  yaw: torch.Tensor,
  frame: int,
  cfg: ApproachCfg,
) -> Approach:
  """The pose this obstacle demands, and where the walk stops short of it.

  Solved once per obstacle rather than every step. The entry index is fixed by the demo, so
  the entry frame is fixed, so the pose is a constant of the obstacle: nothing about the
  robot enters this, which is what lets the plan carry plain coordinates.
  """
  want_xy, want_yaw = guess(obstacle, pos, yaw, cfg)
  at_xy, at_yaw, anchor = place(
    env,
    skill,
    frame,
    want_xy,
    want_yaw,
    obstacle=(pos, yaw) if obstacle.kind == BOX else None,
  )
  # Back along the direction of travel, not along the heading. The bridge crosses on the
  # line the entry is moving down, so a hold point off that line is distance it has to
  # cover sideways inside a window that was sized for the straight version
  travel = torch.stack([torch.cos(anchor), torch.sin(anchor)], dim=-1)
  return Approach(
    xy=at_xy,
    yaw=at_yaw,
    anchor_yaw=anchor,
    frame=frame,
    hold_xy=at_xy - cfg.hold_back * travel,
  )
