"""What an experiment hands to an architecture.

An architecture cannot know any of this by itself: which skills it is composing, which
channels of the observation it may work on, how much of each skill is worth recording
around a hand-over, or which scene entity is the robot.

Nothing architecture-specific belongs here. An architecture that needs something else
takes it as its own parameter, so its signature says plainly who needs what: arch_1 is
the only one that needs a per-skill success test, so it is the only one that asks for
one.
"""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.tasks.skills.skill import SkillPool
from mjlab.tasks.skills.view import StateView
from mjlab.tasks.skills.windows import WindowPlan


@dataclass(frozen=True)
class Experiment:
  """One experiment, as an architecture sees it."""

  name: str
  """Folder trained architectures are saved under (see utils.py)."""

  entity_name: str
  """The scene entity that is the robot: what interrupt states are harvested from."""

  pool: SkillPool
  """The skills being composed. Position in the pool is a skill's id (see skill.py)."""

  view: StateView
  """The slice of the observation the architecture works on (see view.py)."""

  windows: WindowPlan
  """How much of each skill to record around a hand-over (see windows.py)."""

  def __post_init__(self) -> None:
    # Fail here rather than partway through a long run.
    self.windows.check(self.pool)
