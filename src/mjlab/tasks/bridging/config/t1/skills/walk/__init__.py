"""Natural walking for the T1: the stock velocity task plus a LAFAN style reward.

The stock velocity task walks, but mechanically: its reward only asks for the commanded
velocity and some regularity, so PPO settles on the cheapest gait that satisfies it. Here a
style term rewards being close to a retargeted LAFAN walking frame that moves at the
commanded velocity, see style.py. The policy stays a plain velocity tracker with the same
observations and actions, and reads no reference at inference.

    dataset.py        retargets the LAFAN walks and builds the frame library
    style.py          the nearest frame style reward
    walk_env_cfg.py   the stock flat task with the style term on top

Run

1. Install GMR into the venv, once.

    uv pip install --no-deps -e data/GMR

2. Retarget the walks and build the frame library.

    uv run python -m mjlab.tasks.bridging.config.t1.skills.walk.dataset

3. Train.

    uv run train Mjlab-T1-Walk-Natural --env.scene.num-envs 4096

4. Play the latest checkpoint.

    uv run play Mjlab-T1-Walk-Natural
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.bridging.config.t1.skills.walk.walk_env_cfg import t1_walk_env_cfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.t1.rl_cfg import booster_t1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

WALK_TASK_ID = "Mjlab-T1-Walk-Natural"


def walk_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The stock T1 velocity PPO config, under this experiment's own name."""
  return replace(booster_t1_ppo_runner_cfg(), experiment_name="t1_walk_natural")


register_mjlab_task(
  task_id=WALK_TASK_ID,
  env_cfg=t1_walk_env_cfg(),
  play_env_cfg=t1_walk_env_cfg(play=True),
  rl_cfg=walk_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
