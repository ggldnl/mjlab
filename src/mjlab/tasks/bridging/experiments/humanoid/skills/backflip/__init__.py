"""A backflip for the Unitree G1, tracked from a standstill.

Run:

    1. Build the reference. Writes to data/backflip.

       uv run python -m \
         mjlab.tasks.bridging.experiments.humanoid.skills.backflip.dataset

    2. Train.

       uv run train Mjlab-G1-Backflip --env.scene.num-envs 4096

    3. Watch.

       uv run play Mjlab-G1-Backflip

The jump's recipe with a harder clip: reference-state initialization, a dense per-frame
tracking reward, and termination the moment tracking is lost. The clip stands still, crouches,
extends, turns once backwards in the air and lands about a third of a metre behind where it
started.

The reference is written rather than recorded. Nothing in the motion sets these skills draw
on contains a backflip, so dataset.py builds one out of eight keyframed poses and a ballistic
arc, and measures the pelvis heights it cannot guess by replaying the poses through the
model. See its module docstring for what that does and does not buy.

Two numbers to watch, both from the reference rather than the policy. dataset.py prints the
takeoff speed it is asking for and the turn rate that goes with it. Longer flight means more
of both, and there is a flight time past which no policy will be able to answer, whatever the
reward says.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.backflip.backflip_env_cfg import (
  g1_backflip_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.jump import jump_ppo_runner_cfg
from mjlab.tasks.registry import register_mjlab_task

BACKFLIP_TASK_ID = "Mjlab-G1-Backflip"


def backflip_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The jump's PPO config, under this experiment's own name.

  The same tracking objective on the same robot, so the same settings, including the low
  initial standard deviation the jump uses. Per-joint exploration noise is expensive in a
  motion that is airborne half the time, because there is no step in which to correct it.
  """
  return replace(jump_ppo_runner_cfg("g1_backflip"), max_iterations=15_000)


register_mjlab_task(
  task_id=BACKFLIP_TASK_ID,
  env_cfg=g1_backflip_env_cfg(),
  play_env_cfg=g1_backflip_env_cfg(play=True),
  rl_cfg=backflip_ppo_runner_cfg(),
)
