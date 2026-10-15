"""Single trajectory and goal conditioned jumps for the Booster T1.

Mjlab-T1-Jump tracks retargeted jump_forward_level3 with RSI, starting two frames
before the lowest crouch and keeping the original ending. Phase spans zero to one
over the cropped reference. It uses ordinary PPO and keeps the reference at inference.

    uv run train Mjlab-T1-Jump --env.scene.num-envs 4096
    uv run play Mjlab-T1-Jump

The continuous task below is unchanged.

The T1 mirror of the G1 jump_continuous skill. Same task, same runner, same MDP terms:

    phase one   track every clip with reference state initialization, the goal in the
                observation and the reward, each clip stretched horizontally per episode
    phase two   distil the tracker, in the same run, into a student that reads the
                distance left to jump and its own body only. Deployed, the distance is
                the only input and no clip is loaded

The clips are the G1's too: ASAP's five standing forward jumps, published retargeted to
the G1 only. dataset.py retargets them onto the T1 with inverse kinematics, scaled by the
ratio of the two leg lengths, so the T1 jumps 0.81 times as far.

Run

1. Retarget the G1 clips onto the T1, converting them first if needed. Writes to
   data/asap_t1/motions.

    uv run --with joblib python -m mjlab.tasks.bridging.config.t1.skills.jump.dataset

2. Train, both phases.

    uv run train Mjlab-T1-Jump-Continuous --env.scene.num-envs 4096

   The split is agent.tracking_iterations of agent.max_iterations. Move it with
   --agent.tracking-iterations N.

3. Watch what came out, which is the student, on a distance alone. Pick it with the
   viewer's "Jump distance" slider and press "Jump".

    uv run play Mjlab-T1-Jump-Continuous
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.rl import MjlabTeacherStudentRunner
from mjlab.tasks.bridging.config.g1.skills.jump import jump_ppo_runner_cfg
from mjlab.tasks.bridging.config.g1.skills.jump_continuous import jump_runner_cfg
from mjlab.tasks.bridging.config.t1.skills.jump.jump_env_cfg import (
  t1_jump_continuous_env_cfg,
  t1_jump_env_cfg,
)
from mjlab.tasks.registry import register_mjlab_task

JUMP_TASK_ID = "Mjlab-T1-Jump"
JUMP_CONTINUOUS_TASK_ID = "Mjlab-T1-Jump-Continuous"

register_mjlab_task(
  task_id=JUMP_TASK_ID,
  env_cfg=t1_jump_env_cfg(),
  play_env_cfg=t1_jump_env_cfg(play=True),
  rl_cfg=replace(jump_ppo_runner_cfg(), experiment_name="t1_jump"),
)

register_mjlab_task(
  task_id=JUMP_CONTINUOUS_TASK_ID,
  env_cfg=t1_jump_continuous_env_cfg(),
  play_env_cfg=t1_jump_continuous_env_cfg(deploy=True),
  rl_cfg=jump_runner_cfg("t1_jump_continuous"),
  runner_cls=MjlabTeacherStudentRunner,
)
