"""Booster T1 skills available to bridge tools."""

from mjlab.tasks.bridging.config.t1.skills.push import (  # noqa: F401
  PUSH_TASK_ID,
)
from mjlab.tasks.bridging.config.t1.skills.walk import (
  WALK_TASK_ID,
)

SKILLS: dict[str, str] = {
  "walk": WALK_TASK_ID,
  "push": PUSH_TASK_ID,
}
