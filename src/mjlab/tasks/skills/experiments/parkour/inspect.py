"""Watch a parkour hand-over: the skill being left, then the skill being entered.

The pair the experiment is built around is `run` into `jump`. The first half is a robot
at speed in its last steps before the bridge would take over; the second is the jump's
clip opening from a stand. The gap between them is the problem, and how large it looks
here is how hard the bridge's job is. The viewer's reset button draws a new hand-over
point, so it can be walked across the range training samples from.

    uv run python -m mjlab.tasks.skills.experiments.parkour.inspect
    uv run python -m mjlab.tasks.skills.experiments.parkour.inspect --source 1 --target 2
"""

from __future__ import annotations

import tyro

import mjlab
from mjlab.tasks.skills.experiments.parkour import JUMP_TASK_ID, WINDOWS, build_pool
from mjlab.tasks.skills.experiments.parkour.arena import parkour_arena_env_cfg
from mjlab.tasks.skills.inspect import InspectConfig, run_inspect

if __name__ == "__main__":
  # The arena rather than any one skill's task, and not because of the corridor: the
  # three skills read three different observation groups (walk and run a commanded
  # twist, jump a reference clip) and the arena is the only env carrying all of them, so
  # building from the jump task alone gives a pool whose walk and run cannot find their
  # own observations. See arena.py.
  #
  # Obstacles off, though. A hand-over is a thing that happens to the robot, and a
  # couple is rolled from a reset rather than from partway down the corridor, so boxes
  # would be scenery the robot happens to be standing in.
  run_inspect(
    tyro.cli(InspectConfig, config=mjlab.TYRO_FLAGS),
    JUMP_TASK_ID,
    build_pool,
    WINDOWS,
    env_cfg=parkour_arena_env_cfg(obstacles=None),
  )
