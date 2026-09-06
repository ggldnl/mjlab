"""Kick: a G1 walks into a football and strikes it, following a human kick.

The task is motion tracking with a ball in the way. A published human kick, retargeted to
this exact robot, is tracked frame by frame; the ball is placed where that reference puts the
striking foot at the moment it is moving fastest; and two latched terms pay for connecting
with it and for how hard it leaves. Nothing anneals the reference away, so what the policy
learns is a human kick that also happens to hit a ball, rather than whatever swing maximises
ball speed.

The clips come from PAiD (arXiv 2602.05310), whose thirteen kick motions are published under
CC BY-NC 4.0 and are already in the 29 joint G1's coordinates. dataset.py permutes them into
mjlab's joint order and replays them, and its module docstring explains why that permutation
is the only conversion needed.

The default clip runs in rather than stepping in: 1.05 m of approach at 1.98 m/s, two strides
before the plant. That is not decoration. Measured across all thirteen, the clips that walk
in slowly are also the ones that plant their feet close together, and a 0.22 m ball then has
nowhere to sit that the support foot does not tread on first. The clips with a run-up have
the wide stance that leaves room for it.

What replaced the previous version of this package:

    trigger latch      the old task ran its own locomotion and snapped the clip in when a
                       moving ball was predicted to arrive. Nothing here predicts anything:
                       the ball is steady and the reference covers the walk in
    annealed prior     the old task faded the imitation reward out and paid for where the
                       ball ended up. That is reward shaped kicking and it looked like it
    reward shaped ball the old task carried an approach term, a stance gate and a launch
                       command. The reference says all three, frame by frame

Run

1. Convert the clip. Downloads it on first use.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.kick.dataset

2. Train.

    uv run train Mjlab-G1-Kick --env.scene.num-envs 4096

3. Watch.

    uv run play Mjlab-G1-Kick

A motion's task id is its name in title case, so a second entry named front_kick in
dataset.py's MOTIONS would be Mjlab-G1-Front-Kick.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous import (
  jump_ppo_runner_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.kick.dataset import (
  MOTIONS,
  motion_dir,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.kick.kick_env_cfg import (
  g1_kick_env_cfg,
)
from mjlab.tasks.registry import register_mjlab_task


def task_id(motion: str) -> str:
  """Task id of one motion: kick is Mjlab-G1-Kick."""
  return "Mjlab-G1-" + "-".join(part.capitalize() for part in motion.split("_"))


KICK_TASK_IDS: dict[str, str] = {name: task_id(name) for name in MOTIONS}
"""Motion name to task id. Every one logs to g1_<name>."""

KICK_TASK_ID = KICK_TASK_IDS["kick"]
"""The default kick, which is what the rest of the experiment reaches for by name."""


def kick_ppo_runner_cfg(motion: str) -> RslRlOnPolicyRunnerCfg:
  """The jump's PPO config, under one motion's own experiment name.

  Nothing about the algorithm changes, because nothing about the problem does: the same
  tracking objective on the same robot. It runs longer than the martial motions because the
  ball terms only start paying once the reference is followed well enough to reach the ball,
  so the second half of the curriculum is where this task is actually learned.
  """
  return replace(jump_ppo_runner_cfg(f"g1_{motion}"), max_iterations=12_000)


for _motion, _task_id in KICK_TASK_IDS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=g1_kick_env_cfg(motion_dir(_motion)),
    play_env_cfg=g1_kick_env_cfg(motion_dir(_motion), play=True),
    rl_cfg=kick_ppo_runner_cfg(_motion),
  )
