"""The bridge, as the demo uses it: aim it at a pose, let it cross, hand over.

    bridge.aim(skill, reach, at_xy, at_yaw, duration_s)
    while not bridge.done: action = bridge(obs)

One object owns the bridge policy and the command term that carries its target, so nothing
outside has to know that a target is a state vector, that a window is opened in seconds, or
that a clip has to be pinned before the crossing rather than after it.

Why this is not `tests.stage.aim`
--------------------------------

That function turns the entry state to face the way the robot is facing now and lands the
crossing wherever a decelerating body would end up. Both are right for a couple of skills on
an open plane, where a strike can happen anywhere and any heading is as good as another.

Neither is right in front of an obstacle. The climb's reference and its box are one rigid
thing, so the pose the robot arrives in decides where that box ends up: arrive turned five
degrees and the reference climbs a box five degrees off the real one. So here both are
commanded. The controller solves the pose it wants and the bridge is aimed at exactly that,
heading included.

`ARRIVE_SLACK` is what stops that being a licence. A crossing travels along one line, so a
duration decides how far the robot goes and never which way, and a demanded pose off that
line stays off it. `solve` returns the residual so the controller can wait rather than fire
into a miss.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp import ROOT_STATE_DIM
from mjlab.tasks.bridging.experiments.humanoid.demos.parkour.pool import (
  Skill,
  SkillPool,
)
from mjlab.tasks.bridging.experiments.humanoid.selector import Reach
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import (
  ARRIVE_SLACK,
  BRIDGE_GROUP,
  DEFAULT_DURATION_S,
  PROBE_S,
  Aimed,
  crossing_time,
  turn_state,
)
from mjlab.utils.lab_api.math import quat_conjugate, quat_mul, yaw_quat


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
  """A rotation about z, as a quaternion. `(N,)` -> `(N, 4)`, wxyz."""
  half = 0.5 * yaw
  out = torch.zeros(yaw.shape[0], 4, device=yaw.device, dtype=yaw.dtype)
  out[:, 0] = torch.cos(half)
  out[:, 3] = torch.sin(half)
  return out


def facing_yaw(entry: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  """One recorded state, turned to face a commanded heading.

  `tests.stage.facing` turns it to face the way the robot happens to be facing. This turns
  it to face where the controller says, which is the difference between a hand-over that
  ends up square to an obstacle and one that ends up square to whatever the walk drifted to.

  The turn has to reach the orientation and both velocity vectors, or the pose and the
  momentum disagree about which way the body is going. `turn_state` does that.
  """
  turn = quat_mul(quat_from_yaw(yaw), quat_conjugate(yaw_quat(entry[:, 3:7])))
  return turn_state(entry, turn)


class Bridge:
  """The bridge policy and its target. Aim it, step it, ask whether it has landed."""

  def __init__(self, env: ManagerBasedRlEnv, pool: SkillPool) -> None:
    self.env = env
    self.skill = pool[BRIDGE_GROUP]
    command = env.command_manager.get_term("bridge")
    assert isinstance(command, Aimed)
    self.command = command
    self.target: torch.Tensor | None = None
    """Where the last crossing was aimed. None before the first one."""

  ##
  # What a window may be.
  ##

  @property
  def duration_range(self) -> tuple[float, float]:
    """What the bridge was trained on, in seconds. Outside it, it is being asked a question
    it was never shown."""
    return self.command.cfg.duration_s_range

  def window(self, reach: Reach, solved: float | None = None) -> float:
    """How long the bridge gets, in seconds.

    A window solved from the geometry wins, because it is the one that lands the crossing
    where the entering skill needs the robot. Otherwise the entry's own floor: effort scales
    with 1/seconds, so PROBE_S * effort is the window at which this entry costs exactly one,
    a change as fast as any recorded skill performed. A floor and not a set point, hence the
    max against the default. Clamped to the trained range either way.
    """
    low, high = self.duration_range
    if solved is not None:
      return float(min(max(solved, low), high))
    return float(min(max(PROBE_S * reach.effort, DEFAULT_DURATION_S, low), high))

  ##
  # Aiming.
  ##

  def target_state(
    self, reach: Reach, at_xy: torch.Tensor, at_yaw: torch.Tensor
  ) -> torch.Tensor:
    """The entry state, turned to the commanded heading and moved to the commanded spot."""
    entry = torch.as_tensor(
      reach.entry.state[None], dtype=torch.float32, device=self.env.device
    ).expand(at_xy.shape[0], -1)
    target = facing_yaw(entry.clone(), at_yaw)
    target[:, 0:2] = at_xy
    return target

  def solve(
    self, here: torch.Tensor, target: torch.Tensor, want: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """How long a crossing needs to land on `want`, and how far off the line it would be.

    Both matter. A duration inside the trained range with half a metre of residual is a
    crossing that arrives on time somewhere else, and in front of an obstacle that is a
    robot walking into a corner.
    """
    return crossing_time(here, target, want)

  def aim(
    self,
    skill: Skill,
    reach: Reach,
    here: torch.Tensor,
    at_xy: torch.Tensor,
    at_yaw: torch.Tensor,
    duration_s: float,
    anchor_yaw: torch.Tensor | None = None,
    frame: int | None = None,
  ) -> torch.Tensor:
    """Point the bridge at a pose and start the clock.

    Order matters and is the one thing this method exists to get right. The target is built
    first, then the entering skill is told where it will be, then the window opens. The skill
    is placed against the *target*, not against the robot: pinning a clip to where the robot
    stands now slides the reference onto the arrival and erases the error the hand-over is
    being measured on, and for the climb it drags the reference's obstacle off the real one.

    `at_yaw` is the heading the robot arrives holding, which is what the target faces.
    `anchor_yaw` is what the clip is pinned with, which `anchor_to_robot` reads as a
    direction of travel and which differs from the first by the pelvis twist the clip carries.
    None means they are the same, which is true of a skill with no clip. The controller
    solves both together: see `Controller.place`.
    """
    del here  # the target is commanded, so where the robot is now does not place it
    target = self.target_state(reach, at_xy, at_yaw)
    pinned = at_yaw if anchor_yaw is None else anchor_yaw
    skill.enter(
      self.env,
      target[:, 0:3],
      quat_from_yaw(pinned),
      reach.entry.frame if frame is None else frame,
    )

    self.target = target
    self.command.target[:] = target
    self.command.aimed = True
    env_ids = torch.arange(self.command.num_envs, device=self.command.device)
    # Seconds, not ticks. The command converts, and it is the only thing that should
    self.command.open_window(
      env_ids,
      torch.full((self.command.num_envs,), duration_s, device=self.command.device),
    )
    return target

  ##
  # Crossing.
  ##

  @property
  def done(self) -> bool:
    """Whether the open window has closed."""
    return bool((self.command.step >= self.command.deadline).all())

  def arrival(self, here: torch.Tensor) -> tuple[float, float, float]:
    """How far off the target the robot ended up: metres, radians of heading, radians of joint.

    Printed at every hand-over, and the three say different things. Position is what an
    arrival score already covers. Heading is the one this demo added, because it decides
    whether a clip's own obstacle lines up with the real one. The worst joint is the one that
    says whether the robot is actually in the entry *pose* rather than merely standing in the
    right spot facing the right way: a tracker handed a good root and the wrong limbs is
    being asked to continue a motion from a body that is not in it.
    """
    if self.target is None:
      return 0.0, 0.0, 0.0
    gap = float((here[0, 0:3] - self.target[0, 0:3]).norm())
    turn = quat_mul(
      yaw_quat(here[:, 3:7]), quat_conjugate(yaw_quat(self.target[:, 3:7]))
    )
    angle = float(2.0 * torch.atan2(turn[0, 3].abs(), turn[0, 0].abs()))
    joints = slice(
      ROOT_STATE_DIM, ROOT_STATE_DIM + (here.shape[1] - ROOT_STATE_DIM) // 2
    )
    worst = float((here[0, joints] - self.target[0, joints]).abs().max())
    return gap, angle, worst

  @torch.no_grad()
  def __call__(self, obs) -> torch.Tensor:
    return self.skill(obs)


SLACK = ARRIVE_SLACK
"""How far off the demanded line a crossing may be when the switch fires, in metres.

Re-exported rather than redefined. `tests.stage` measured it: over a 0.6 s window the
ballistic scatter is 0.72 m, and firing anywhere inside that scored 0.146 against 0.554 at
0.10 m. Ten centimetres leaves the model deciding when, within a stride, and the obstacle
deciding where."""
