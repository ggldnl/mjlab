"""Load the trained universal tracker for mixed-planner paths."""

from mjlab.tasks.bridging.bridges.diffusion.execution.learned_tracker import (
  LearnedTrackerExecutor as _LearnedTrackerExecutor,
)


class LearnedTrackerExecutor(_LearnedTrackerExecutor):
  """The existing tracker with a mixed-package import path.

  Its inherited classmethod constructs this subclass, so checkpoints load through
  mixed.tracker without changing their keys or observation normalizer.
  """
