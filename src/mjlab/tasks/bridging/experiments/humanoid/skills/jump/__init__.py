"""A single clip jump for the Unitree G1, tracked end to end.

ASAP's stage one: reference state initialization over the whole clip, a dense
per frame tracking reward, and termination the moment tracking is lost.
The clip is jump_forward_level3, which travels 1.54 m.

The difference from skills/jump_continuous is what the policy is asked for. That one covers
a range of distances, this one jumps the one distance the clip jumps: the policy reads the
clip at inference, the way the front kick and the punch combo do.

Run

1. Convert the clips. Same dataset as the continuous jump, same output directory, nothing
   extra to fetch. Writes to data/asap/motions.

    uv run --with joblib python -m \
      mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.dataset

2. Train.

    uv run train Mjlab-G1-Jump --env.scene.num-envs 4096

3. Watch.

    uv run play Mjlab-G1-Jump
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.jump.jump_env_cfg import (
  g1_jump_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous import (
  jump_ppo_runner_cfg as tracking_ppo_runner_cfg,
)
from mjlab.tasks.registry import register_mjlab_task

JUMP_TASK_ID = "Mjlab-G1-Jump"


def jump_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The continuous jump's PPO config, phase one of it, under this experiment's own name.

  Nothing about the algorithm changes, because nothing about the problem does: the same
  tracking objective on the same robot. It runs shorter for the same reason the front kick
  does, which is that there is no goal range to cover, only one clip to follow.
  """
  return replace(tracking_ppo_runner_cfg("g1_jump"), max_iterations=10_000)


register_mjlab_task(
  task_id=JUMP_TASK_ID,
  env_cfg=g1_jump_env_cfg(),
  play_env_cfg=g1_jump_env_cfg(play=True),
  rl_cfg=jump_ppo_runner_cfg(),
)
