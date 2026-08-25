"""Watch a cart-pole hand-over: the skill being left, then the skill being entered.

The pair to look at is `spin_up` into `balance`. The first half is the pole in the last
steps before the bridge would take over; the second is `balance` opening from its own
start, which is upright. The viewer's reset button draws a new hand-over point, so the
pole can be seen handed over hanging, mid-swing, and arriving at the top.

    uv run python -m mjlab.tasks.bridging.experiments.cartpole.inspect
    uv run python -m mjlab.tasks.bridging.experiments.cartpole.inspect --source 1 --target 0
"""

from __future__ import annotations

import tyro

from mjlab.tasks.bridging.experiments.cartpole import (
  SPINUP_TASK_ID,
  WINDOWS,
  build_pool,
)
from mjlab.tasks.bridging.inspect import InspectConfig, run_inspect

if __name__ == "__main__":
  run_inspect(tyro.cli(InspectConfig), SPINUP_TASK_ID, build_pool, WINDOWS)
