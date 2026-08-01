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
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.controller import Controller
from mjlab.tasks.skills.experiments.parkour.arena import CORRIDOR, Obstacle
from mjlab.tasks.skills.skill import SkillPool

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
    jump_distance: float = 1.6,
    clear_distance: float = 0.6,
    run_distance: float = 6.0,
    entity_name: str = "robot",
  ) -> None:
    """
    Args:
      pool: The skills, in walk/run/jump order.
      obstacles: The corridor, as the controller believes it to be.
      jump_distance: How far in front of an obstacle the jump is called [m]. Set
        from the skill, not from taste: the converted clips cover 0.5 to 2.0 m and
        spend the first part of every one of them crouching, so calling the jump
        much later than this leaves no room to load the legs, and much earlier
        lands the robot short of the box.
      clear_distance: How far past an obstacle counts as over it [m].
      run_distance: A gap at least this long is worth running [m].
      entity_name: Whose position to read.
    """
    super().__init__(pool)
    self.entity_name = entity_name
    self.jump_distance = jump_distance
    self.clear_distance = clear_distance
    self.run_distance = run_distance

    self._obstacle_x = torch.tensor([o.x for o in obstacles], dtype=torch.float32)
    self._obstacle_half = torch.tensor(
      [o.depth / 2.0 for o in obstacles], dtype=torch.float32
    )
    self._committed: torch.Tensor | None = None

  def _distances(self, env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Distance to the next obstacle, and the length of the stretch leading to it.

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
    return distance, gap

  def decide(self, env: ManagerBasedRlEnv, target: torch.Tensor) -> torch.Tensor:
    if self._committed is None:
      self._committed = torch.full_like(target, WALK)

    distance, gap = self._distances(env)

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

    self._committed = decision
    return decision

  def reset(self, mask: torch.Tensor) -> None:
    if self._committed is None:
      self._committed = torch.full_like(mask, WALK, dtype=torch.long)
    self._committed = torch.where(
      mask, torch.full_like(self._committed, WALK), self._committed
    )
