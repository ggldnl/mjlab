"""The corridor controller: which skill the situation calls for.

Assumed given, like every controller in this study. The obstacle layout is known, so
the decision is a rule read off the robot's position along the corridor rather than
anything learned:

- an obstacle close ahead        -> jump
- an obstacle just behind, or    -> walk
  another one soon after
- nothing ahead for a while      -> run

It reads position and nothing else. It does not know a bridge exists, it does not ask
a skill how it is doing, and it never looks at whether the robot is in a fit state for
what it just asked for. That last part is deliberate and is the whole point of the
experiment: told to jump while running, the controller says jump, and something else
has to make that survivable.

Naming the skill is only half of it. All three skills are goal conditioned, so a
skill that is running is still waiting to be told what to do, and the controller is
the only thing here that knows: the corridor's direction, the speed each skill is for,
and how far the next jump has to carry. So it writes both commands as well.

The velocity command is the same for walking and running apart from its magnitude,
and is written every step: forward down the corridor, at this skill's speed. The jump
command is written once, at the moment the jump is called, because a jump distance
chosen mid-flight is not a goal but a disturbance -- and because the reference is
anchored to the robot on the same step (see skills.py), which has to happen after the
clip it anchors is chosen.

Note what is deliberately not done: nothing here is conditioned on how fast the robot
is currently going. The commanded jump distance is read off the corridor, and the
speed is read off the skill, so a robot that arrives at the box carrying a run's
momentum is still handed a clip that opens from a stand. That is the failure the
bridge exists to remove, and constraining the commands must not quietly remove it
first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.controller import Controller
from mjlab.tasks.skills.experiments.parkour.arena import (
  CORRIDOR,
  JUMP_COMMAND,
  RUN_SPEED,
  TWIST_COMMAND,
  WALK_SPEED,
  Obstacle,
)
from mjlab.tasks.skills.skill import SkillPool

if TYPE_CHECKING:
  from mjlab.tasks.skills.experiments.parkour.jump.mdp.commands import JumpCommand
  from mjlab.tasks.velocity.mdp import UniformVelocityCommand

# Skill ids, fixed by the order `build_pool` puts them in.
WALK = 0
RUN = 1
JUMP = 2


class ParkourController(Controller):
  """Picks walk, run or jump from where the robot is along the corridor."""

  def __init__(
    self,
    pool: SkillPool,
    obstacles: tuple[Obstacle, ...] = CORRIDOR,
    jump_distance: float = 0.8,
    clear_distance: float = 0.6,
    run_distance: float = 6.0,
    entity_name: str = "robot",
    walk_speed: float = WALK_SPEED,
    run_speed: float = RUN_SPEED,
    landing_margin: float = 0.4,
  ) -> None:
    """
    Args:
      pool: The skills, in walk/run/jump order.
      obstacles: The corridor, as the controller believes it to be.
      jump_distance: How far in front of an obstacle the jump is called [m]. Set
        from the skill, not from taste, and from its landing rather than its length:
        the clips touch down between 0.19 and 1.47 m from where they start, and the
        one that reaches furthest is only reliable near its own scale. Calling the
        jump from further back than this asks for a landing no clip can stretch to,
        which is not a harder jump but a missed one. There is room to call it later:
        every clip is airborne within 0.4 m of its first frame.
      clear_distance: How far past an obstacle counts as over it [m].
      run_distance: A gap at least this long is worth running [m].
      entity_name: Whose position to read.
      walk_speed: Forward speed commanded while the walk is running [m/s].
      run_speed: Forward speed commanded while the run is running [m/s].
      landing_margin: How far past an obstacle's far edge the jump is asked to land
        [m]. Not slack: the goal is where the reference root ends up, and a jump
        commanded to land exactly on the far edge is one whose trailing foot is
        still over the box.
    """
    super().__init__(pool)
    self.entity_name = entity_name
    self.jump_distance = jump_distance
    self.clear_distance = clear_distance
    self.run_distance = run_distance
    self.walk_speed = walk_speed
    self.run_speed = run_speed
    self.landing_margin = landing_margin

    self._obstacle_x = torch.tensor([o.x for o in obstacles], dtype=torch.float32)
    self._obstacle_half = torch.tensor(
      [o.depth / 2.0 for o in obstacles], dtype=torch.float32
    )
    self._committed: torch.Tensor | None = None

  def _distances(
    self, env: ManagerBasedRlEnv
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Distance to the next obstacle, the stretch leading to it, and how deep it is.

    Both are measured to the obstacle's near edge. The gap is the second one and it
    is what decides running: how much clear ground this stretch had in total, not how
    much of it is left. Deciding on what is left would make the controller slow the
    robot down as it approaches an obstacle, which is a hand-over problem being
    quietly solved by the thing that is supposed to be causing it.
    """
    entity = env.scene[self.entity_name]
    x = entity.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]

    near = (self._obstacle_x - self._obstacle_half).to(x.device)
    far = (self._obstacle_x + self._obstacle_half).to(x.device)

    # Distance to each obstacle's near edge; negative once the robot is past its far
    # edge. Obstacles already cleared are pushed out of the way with a large value so
    # sorting always names the next one ahead first.
    ahead = near.unsqueeze(0) - x.unsqueeze(1)
    cleared = (x.unsqueeze(1) - far.unsqueeze(0)) > self.clear_distance
    ahead = torch.where(cleared, torch.full_like(ahead, float("inf")), ahead)
    ordered, order = torch.sort(ahead, dim=1)
    distance = ordered[:, 0]

    # The start of this stretch: the far edge of the last obstacle behind the robot,
    # or the corridor's beginning if there is none.
    behind = torch.where(
      cleared, far.unsqueeze(0).expand_as(ahead), torch.full_like(ahead, -float("inf"))
    )
    start = behind.max(dim=1).values
    start = torch.where(torch.isinf(start), torch.zeros_like(start), start)

    next_near = near.to(x.device)[order[:, 0]]
    gap = torch.where(torch.isinf(distance), distance, next_near - start)

    # The next obstacle's depth, which is what a jump over it has to span.
    depth = (2.0 * self._obstacle_half).to(x.device)[order[:, 0]]
    return distance, gap, depth

  def decide(self, env: ManagerBasedRlEnv, target: torch.Tensor) -> torch.Tensor:
    if self._committed is None:
      self._committed = torch.full_like(target, WALK)

    distance, gap, depth = self._distances(env)

    # Running is for the long stretches, and it goes on right up to the jump. That is
    # the point of the experiment: the jump is called on a robot at speed, and nothing
    # in the controller slows it down first.
    wants_run = gap > self.run_distance
    # Close enough to jump, and not already over it.
    wants_jump = (distance <= self.jump_distance) & (distance > -self.clear_distance)

    decision = torch.where(
      wants_jump,
      torch.full_like(target, JUMP),
      torch.where(
        wants_run, torch.full_like(target, RUN), torch.full_like(target, WALK)
      ),
    )

    # A jump, once called, is held until the robot is over the obstacle. Without this
    # the rule would flicker: the moment the robot leaves the ground its distance to
    # the box changes, and the same rule that asked for a jump would ask for something
    # else mid-flight.
    was_jumping = (target == JUMP) | (self._committed == JUMP)
    still_short = distance > -self.clear_distance
    decision = torch.where(
      was_jumping & still_short, torch.full_like(decision, JUMP), decision
    )

    # A jump is commanded before it is started: the goal has to be in place before
    # the skill anchors its reference to the robot, which happens later this step.
    starting_jump = (decision == JUMP) & (self._committed != JUMP)
    if starting_jump.any():
      self._command_jump(env, starting_jump, distance, depth)
    self._command_twist(env, decision)

    self._committed = decision
    return decision

  def _twist(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    from mjlab.tasks.velocity.mdp import UniformVelocityCommand

    term = env.command_manager.get_term(TWIST_COMMAND)
    assert isinstance(term, UniformVelocityCommand)
    return term

  def _jump(self, env: ManagerBasedRlEnv) -> JumpCommand:
    from mjlab.tasks.skills.experiments.parkour.jump.mdp.commands import JumpCommand

    term = env.command_manager.get_term(JUMP_COMMAND)
    assert isinstance(term, JumpCommand)
    return term

  def _command_twist(self, env: ManagerBasedRlEnv, decision: torch.Tensor) -> None:
    """Forward, at the speed the chosen skill is for.

    Written every step rather than on a switch, because the command term's own reset
    puts back whatever its configured range says, and an episode that begins with a
    reset would otherwise run at that instead. The yaw rate is left alone: the arena
    configures this term to hold a heading of +x, and that overwrites the channel on
    every update anyway (see arena.py).
    """
    twist = self._twist(env)
    speed = torch.where(
      decision == RUN,
      torch.full_like(twist.vel_command_b[:, 0], self.run_speed),
      torch.full_like(twist.vel_command_b[:, 0], self.walk_speed),
    )
    twist.vel_command_b[:, 0] = speed
    twist.vel_command_b[:, 1] = 0.0

  def _command_jump(
    self,
    env: ManagerBasedRlEnv,
    starting: torch.Tensor,
    distance: torch.Tensor,
    depth: torch.Tensor,
  ) -> None:
    """Tell the jump what this one has to clear, in the envs just starting one.

    Stated as a landing and a thing to be airborne over rather than as a jump length,
    because those are what the corridor knows and a clip's length is not either of
    them (see `JumpCommand.solve_landing`). Both are measured from where the robot is
    now, which is where the reference is about to be pinned.

    Which clip and how much stretch serve that is the skill's own business. The
    controller names what has to happen and lets the skill answer, and if nothing in
    its library reaches, it jumps as far as it can and the composition fails on the
    obstacle. That is a real result about the pool, not something to paper over here.
    """
    jump = self._jump(env)
    landing = distance + depth + self.landing_margin
    env_ids = starting.nonzero(as_tuple=False).squeeze(-1)
    for env_id in env_ids.tolist():
      motion_id, scale = jump.solve_landing(
        float(landing[env_id]), takeoff_before=float(distance[env_id])
      )
      jump.set_goal(
        torch.tensor([env_id], dtype=torch.long, device=jump.device), motion_id, scale
      )

  def reset(self, mask: torch.Tensor) -> None:
    if self._committed is None:
      self._committed = torch.full_like(mask, WALK, dtype=torch.long)
    self._committed = torch.where(
      mask, torch.full_like(self._committed, WALK), self._committed
    )
