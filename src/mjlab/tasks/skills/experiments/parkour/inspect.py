"""Watch the parkour windows a bridge is trained on.

Left half of a couple: the skill being handed away from, in its last steps before the
bridge takes over. Right half: the skill being handed to, opening from its own reset.
Previous and Next sweep the hand-over point across the range training samples from.

The pair worth looking at is run into jump. The left half shows a robot at speed; the
right half shows the jump's clip opening from a stand. The gap between them is the
problem, and how large it looks here is how hard the bridge's job is.

    uv run python -m mjlab.tasks.skills.experiments.parkour.inspect
    uv run python -m mjlab.tasks.skills.experiments.parkour.inspect --target 2
"""

from __future__ import annotations

import tyro

import mjlab
from mjlab.tasks.skills.experiments.parkour import JUMP_TASK_ID, WINDOWS, build_pool
from mjlab.tasks.skills.inspect import InspectConfig, run_inspect

if __name__ == "__main__":
  run_inspect(
    tyro.cli(InspectConfig, config=mjlab.TYRO_FLAGS), JUMP_TASK_ID, build_pool, WINDOWS
  )
