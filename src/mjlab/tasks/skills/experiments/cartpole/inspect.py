"""Watch the cart-pole windows a bridge is trained on.

Left half of a couple: spin_up, in the last steps before the bridge would take over.
Right half: balance, opening from its own reset. Previous and Next sweep the hand-over
point across the range training samples from, so you can see the pole handed over
hanging, mid-swing, and arriving at the top.

    uv run python -m mjlab.tasks.skills.experiments.cartpole.inspect
    uv run python -m mjlab.tasks.skills.experiments.cartpole.inspect --target 0
"""

from __future__ import annotations

import tyro

from mjlab.tasks.skills.experiments.cartpole import (
  SPINUP_TASK_ID,
  WINDOWS,
  build_pool,
)
from mjlab.tasks.skills.inspect import InspectConfig, run_inspect

if __name__ == "__main__":
  run_inspect(tyro.cli(InspectConfig), SPINUP_TASK_ID, build_pool, WINDOWS)
