"""Martial arts motions for the Unitree G1, each tracked from a standstill.

The jump's recipe against crops of LAFAN1's fight performances: reference state
initialization, a dense per frame tracking reward, and termination the moment tracking is
lost. One motion is one clip and one policy. They all share dataset.py's converter and
martial_env_cfg.py's environment, because a strike is a strike and only the reference
motion differs.

Every clip is given half a second of held stance at the front. A crop out of a fight starts
mid-bounce, and without the hold a policy would only ever be asked to continue a motion
that was already under way. With it, every episode that starts at frame zero starts from a
robot standing still.

    front_kick   a kick out of a standstill: hold a stance, bring one knee up, snap the
                 foot out and recover
    punch_combo  a run of punches out of the same stance. Harder than the kick: a kick is
                 one strike, so the tracking reward has one moment to get right, while a
                 combination is several in a row and missing the timing on the first puts
                 every one after it out of phase

Add a motion by adding a line to MOTIONS in dataset.py. It gets a task named after it and
nothing here has to change.

Run

1. Cut the clips out of LAFAN1 and convert them. Writes to data/lafan1_g1/clips/<name>.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.martial.dataset

2. Train.

    uv run train Mjlab-G1-Front-Kick --env.scene.num-envs 4096

3. Watch.

    uv run play Mjlab-G1-Front-Kick

A motion's task id is its name in title case, so punch_combo is Mjlab-G1-Punch-Combo.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous import (
  jump_ppo_runner_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.martial.dataset import (
  MOTIONS,
  motion_dir,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.martial.martial_env_cfg import (
  g1_martial_env_cfg,
)
from mjlab.tasks.registry import register_mjlab_task


def task_id(motion: str) -> str:
  """Task id of one motion: front_kick is Mjlab-G1-Front-Kick."""
  return "Mjlab-G1-" + "-".join(part.capitalize() for part in motion.split("_"))


MARTIAL_TASK_IDS: dict[str, str] = {name: task_id(name) for name in MOTIONS}
"""Motion name to task id, which is the whole of what this package exports.

Every motion logs to g1_<name>, so a checkpoint search needs nothing more than the name.
"""


def martial_ppo_runner_cfg(motion: str) -> RslRlOnPolicyRunnerCfg:
  """The jump's PPO config, under one motion's own experiment name.

  Nothing about the algorithm changes, because nothing about the problem does: the same
  tracking objective on the same robot with an observation of the same shape. It runs
  shorter than the jump because there is no goal to cover, only one clip to follow.
  """
  return replace(jump_ppo_runner_cfg(f"g1_{motion}"), max_iterations=10_000)


for _motion, _task_id in MARTIAL_TASK_IDS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=g1_martial_env_cfg(motion_dir(_motion)),
    play_env_cfg=g1_martial_env_cfg(motion_dir(_motion), play=True),
    rl_cfg=martial_ppo_runner_cfg(_motion),
  )
