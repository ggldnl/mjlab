"""Universal tracker used to execute mixed-planner trajectories.

The checkpoint and observation format are intentionally identical to the diffusion
tracker, so an already trained checkpoint loads without conversion.
"""

from mjlab.tasks.bridging.bridges.diffusion.tracker import (
  TRACKER_EXPERIMENT as TRACKER_EXPERIMENT,
)
from mjlab.tasks.bridging.bridges.diffusion.tracker import (
  TRACKER_TASK_ID as TRACKER_TASK_ID,
)
from mjlab.tasks.bridging.bridges.diffusion.tracker import (
  tracker_env_cfg as tracker_env_cfg,
)

from .learned_tracker import LearnedTrackerExecutor as LearnedTrackerExecutor
