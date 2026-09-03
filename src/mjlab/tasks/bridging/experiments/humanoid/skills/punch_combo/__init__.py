"""A punch combination for the Unitree G1, tracked from a standstill.

The front kick's skill against a different window of the same LAFAN1 performance: a stance,
then a run of punches, then a recovery. The converter and the environment both come from
the front_kick package, because a strike is a strike and only the reference motion differs.
What is here is the frame window, in dataset.py, and the registration below.

Harder than the front kick: a kick is one strike, so the tracking reward has one moment
to get right. A combination is several in a row, and missing the timing on the
first puts every one after it out of phase.

Run

1. Cut the clip out of LAFAN1 and convert it. Writes to data/lafan1_g1/punch_combo.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.punch_combo.dataset

2. Train.

    uv run train Mjlab-G1-Punch-Combo --env.scene.num-envs 4096

3. Watch.

    uv run play Mjlab-G1-Punch-Combo
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.front_kick.front_kick_env_cfg import (
  g1_strike_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.jump import jump_ppo_runner_cfg
from mjlab.tasks.bridging.experiments.humanoid.skills.punch_combo.dataset import (
  MOTION_DIR,
)
from mjlab.tasks.registry import register_mjlab_task

PUNCH_COMBO_TASK_ID = "Mjlab-G1-Punch-Combo"


def punch_combo_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The jump's PPO config, under this experiment's own name.

  Nothing about the algorithm changes, because nothing about the problem does: the same
  tracking objective on the same robot with an observation of the same shape. It runs
  shorter than the jump because there is no goal to cover, only one clip to follow.
  """
  return replace(jump_ppo_runner_cfg("g1_punch_combo"), max_iterations=10_000)


register_mjlab_task(
  task_id=PUNCH_COMBO_TASK_ID,
  env_cfg=g1_strike_env_cfg(MOTION_DIR),
  play_env_cfg=g1_strike_env_cfg(MOTION_DIR, play=True),
  rl_cfg=punch_combo_ppo_runner_cfg(),
)
