"""Architecture 4: one bridge for the whole pool, aimed at a chosen moment.

Two components, of which one exists.

The **bridge** crosses a gap between two motions and knows nothing about skills. It is
trained once against a corpus of recorded human motion, in physics, and reused unchanged
by every experiment: it never sees the pool, so a change to the pool cannot invalidate it.
That is in `bridge/`, it trains as its own mjlab task, and it works.

The **chooser** decides *where* to hand over: which moment of the next skill to aim at,
and when to let go of the current one. It does not exist yet. Until it does, this
architecture cannot run a composition, because there is nothing to tell the bridge what to
aim at, and the class below says so rather than pretending otherwise.

What is deliberately not here is a bridge per skill. The claim this architecture is making
is that crossing a gap is a general problem about bodies rather than a specific problem
about a pair of skills, and the corpus is what makes that claim testable: the bridge is
taught by thousands of real transitions between things a person actually did, none of
which are the skills it will be asked to join.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv, VecEnvObs
from mjlab.tasks.skills.architectures.arch_4.bridge import (
  BRIDGE_TASK_ID as BRIDGE_TASK_ID,
)
from mjlab.tasks.skills.meta import MetaPolicy
from mjlab.tasks.skills.skill import SkillPool
from mjlab.tasks.skills.view import StateView


class Arch4(MetaPolicy):
  """The composed architecture. Incomplete: the chooser is missing.

  Constructing this is allowed, so the registry can list arch_4 alongside the others and
  an experiment can name it without the import failing. Running a composition through it
  is not, because the answer would be made up.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    pool: SkillPool,
    view: StateView | None = None,
    entity_name: str = "robot",
  ) -> None:
    """`entity_name` is the one thing arch_4 needs beyond what every architecture needs.

    The bridge works on the robot's root motion and joint state directly rather than on
    the experiment's state view, so it has to be told which scene entity that is, and it
    has to be told at construction because the widths it derives from that entity's joint
    count are fixed there.
    """
    super().__init__(env, pool, view)
    self.entity_name = entity_name

  def begin_switch(
    self, switching: torch.Tensor, source: torch.Tensor, target: torch.Tensor
  ) -> None:
    del switching, source, target
    raise NotImplementedError(_MISSING)

  def bridge_step(
    self,
    obs: VecEnvObs,
    skill_actions: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del obs, skill_actions, source, target, active
    raise NotImplementedError(_MISSING)


_MISSING = (
  "arch_4 has a trained bridge but no chooser, so there is nothing to tell the bridge "
  "which moment of the next skill to aim at. Train and inspect the bridge on its own "
  "with\n"
  f"  uv run train {BRIDGE_TASK_ID}\n"
  "  uv run python -m mjlab.tasks.skills.architectures.arch_4.bridge.evaluate\n"
  "and run arch_0 to arch_3 for a composition that works today."
)
