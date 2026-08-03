"""Watch a diffdrive hand-over: the skill being left, then the skill being entered.

The pair to look at is `drive` into `turn`. The first half is the robot in the last
steps before the bridge would take over, carrying drive's cruise; the second is `turn`
opening from its own reset, which is from rest. The viewer's reset button draws a new
hand-over point, so it can be walked across the range training samples from.

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.inspect
    uv run python -m mjlab.tasks.skills.experiments.diffdrive.inspect --source 1 --target 0
"""

from __future__ import annotations

import tyro

from mjlab.tasks.skills.experiments.diffdrive import (
  DRIVE_TASK_ID,
  WINDOWS,
  build_pool,
)
from mjlab.tasks.skills.inspect import InspectConfig, run_inspect

if __name__ == "__main__":
  run_inspect(tyro.cli(InspectConfig), DRIVE_TASK_ID, build_pool, WINDOWS)
