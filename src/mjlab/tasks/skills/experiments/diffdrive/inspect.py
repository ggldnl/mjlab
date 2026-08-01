"""Watch the diffdrive windows a bridge is trained on.

Left half of a couple: `drive`, in the last steps before the bridge would take over.
Right half: `turn`, opening from its own reset. Previous and Next sweep the hand-over
point across the range training samples from, so you can see the robot go from
carrying most of drive's momentum to carrying all of it.

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.inspect
    uv run python -m mjlab.tasks.skills.experiments.diffdrive.inspect --target 0
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
