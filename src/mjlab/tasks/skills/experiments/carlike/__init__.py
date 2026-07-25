"""
A car-like robot having the following skills:
- drive: commanded to go forward at a given speed; it never sees a turning
    command during its own training, so it doesn't know how to turn safely;
    it starts from zero velocity and ramps up toward the target speed.
- turn: commanded to execute a turn (a nonzero angular-velocity command) at
    a small linear speed; it never sees a fast approach during its own
    training, so it never has to cope with momentum it didn't build up
    itself.

A simple, scripted controller drives the demonstration: run drive for a fixed
number of steps, signal a switch to turn, hold turn for a fixed number of steps,
signal back to drive, and repeat.

To train the skills:

    uv run train Mjlab-Carlike-Drive
    uv run train Mjlab-Carlike-Turn

The same two skills are also available as analytical functions in dynamics.py

The car-like testbed should fail in a more visible way with the car tipping
over.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.skills.experiments.carlike.carlike_env_cfg import (
  carlike_ppo_runner_cfg,
  drive_env_cfg,
  turn_env_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Carlike-Drive",
  env_cfg=drive_env_cfg(),
  play_env_cfg=drive_env_cfg(play=True),
  rl_cfg=carlike_ppo_runner_cfg("carlike_drive"),
)

register_mjlab_task(
  task_id="Mjlab-Carlike-Turn",
  env_cfg=turn_env_cfg(),
  play_env_cfg=turn_env_cfg(play=True),
  rl_cfg=carlike_ppo_runner_cfg("carlike_turn"),
)
