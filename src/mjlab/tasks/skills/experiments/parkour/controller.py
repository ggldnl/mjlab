"""The scripted controller driving the parkour demonstration.

It is assumed given: the decision of which skill should run is read straight off the
robot's position along the corridor, whose obstacles have known positions and shapes
(described in __init__.py). This controller is deliberately memoryless -- it looks
only at where the robot is right now relative to the obstacles:

- within `jump_lead` metres in front of an obstacle (or on top of one, until it has
  cleared the far edge) -> jump;
- nothing within `run_clear` metres ahead -> sprint (run);
- otherwise (approaching, or just past an obstacle) -> walk.

The corridor runs along +x, so "along the corridor" is the robot's world x. Each
obstacle is a span [x0, x1] on that axis (its front and back edges); anything else
about its shape does not affect the decision, only these two edges do.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.controller import Controller
from mjlab.tasks.skills.skill import SkillPool

# Skill ids: the pool order build_pool() registers (see __init__.py). Sprint exists
# in the pool but this controller does not emit it yet -- a clear corridor gets RUN.
WALK = 0
RUN = 1
JUMP = 2
SPRINT = 3

# Scene entity whose position along the corridor drives the decision.
ENTITY_NAME = "robot"


class ParkourController(Controller):
  """Picks walk / run / jump from the robot's position along the corridor.

  `obstacles` is the known layout: each entry is the (front, back) x of one obstacle
  span along the +x corridor. `jump_lead` is how far ahead of an obstacle the jump
  begins; `run_clear` is how much empty corridor ahead triggers a sprint.
  """

  def __init__(
    self,
    pool: SkillPool,
    obstacles: Sequence[tuple[float, float]],
    *,
    jump_lead: float = 1.0,
    run_clear: float = 4.0,
  ) -> None:
    super().__init__(pool)
    self.obstacles = tuple(obstacles)
    self.jump_lead = jump_lead
    self.run_clear = run_clear
    # Built lazily on the first decide, once the env's device is known.
    self._spans: torch.Tensor | None = None

  def _obstacle_spans(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """The obstacle edges as a (num_obstacles, 2) tensor [x0, x1] on the env device."""
    if self._spans is None:
      self._spans = torch.tensor(
        self.obstacles if self.obstacles else [], device=env.device
      ).reshape(-1, 2)
    return self._spans

  def decide(self, env: ManagerBasedRlEnv, target: torch.Tensor) -> torch.Tensor:
    del target  # Memoryless: the position alone names the skill.
    robot_x = env.scene[ENTITY_NAME].data.root_link_pos_w[:, 0]  # (num_envs,)
    spans = self._obstacle_spans(env)

    if spans.numel() == 0:
      # An empty corridor: nothing to jump, just sprint.
      return torch.full_like(robot_x, RUN, dtype=torch.long)

    x0 = spans[:, 0].unsqueeze(0)  # (1, K), obstacle front edges
    x1 = spans[:, 1].unsqueeze(0)  # (1, K), obstacle back edges
    rx = robot_x.unsqueeze(1)  # (N, 1)

    # Jump when within jump_lead in front of an obstacle, and keep jumping until the
    # robot has cleared its back edge.
    over_or_near = ((rx >= x0 - self.jump_lead) & (rx <= x1)).any(dim=1)  # (N,)

    # Distance to the next obstacle's front edge still ahead of the robot (inf if the
    # corridor is clear ahead). A large gap means sprint.
    ahead = x0 > rx  # (N, K)
    gap = torch.where(ahead, x0 - rx, torch.full_like(rx, float("inf")))  # (N, K)
    min_gap = gap.min(dim=1).values  # (N,)
    clear_ahead = min_gap > self.run_clear

    walk = torch.full_like(robot_x, WALK, dtype=torch.long)
    skill = torch.where(clear_ahead, torch.full_like(walk, RUN), walk)
    skill = torch.where(over_or_near, torch.full_like(walk, JUMP), skill)
    return skill
