"""Climbing an obstacle: onto a box, across it and down the far side, as one skill.

The jump's recipe against an OmniRetarget climb: reference state initialization, a dense per
frame tracking reward, and termination the moment tracking is lost. One clip is one policy,
and the clip carries its own obstacle.

The obstacle is what makes this different from the other tracking skills here. OmniRetarget
retargets the human motion and the box together and preserves the contacts between them, so
the clip is only physical against that box at that pose. Everything in dataset.py moves the
two as one thing, and the environment reads the box out of the manifest the converter wrote
rather than out of a constant somebody typed.

Getting on and getting off are one skill because the source motion is one motion. The
subject walks in, climbs, crosses the top and steps down the far side, and a policy that
stopped at the top would end its episode standing on an obstacle, in a state no other skill
in this package has an entry point for.

    climb   0.63 m box, 1.48 m along the approach and 0.70 m across, at 8 degrees of yaw.
            Cut from climb_24 at scale 1.0, the middle of the height band the parkour
            demo asks a climb to cover

The approach walk is not in the clip. The controller drives the robot up to the box with
the walk and hands over facing it, so the reference opens standing a quarter of a metre from
the near face. What that hand-over cannot promise exactly is the angle, and
``climb_env_cfg.APPROACH_YAW_RANGE`` is where that tolerance is set.

Add a motion by adding a line to MOTIONS in dataset.py. It gets a task named after it and
nothing here has to change.

Run

1. Fetch the clips and the obstacle, and convert. Writes to data/omniretarget/clips/<name>.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.skills.climb.dataset

2. Train.

    uv run train Mjlab-G1-Climb --env.scene.num-envs 4096

3. Watch.

    uv run play Mjlab-G1-Climb

A motion's task id is its name in title case, so a motion called climb_low would be
Mjlab-G1-Climb-Low.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.climb.climb_env_cfg import (
  g1_climb_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.climb.dataset import (
  MOTIONS,
  motion_dir,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous import (
  jump_ppo_runner_cfg,
)
from mjlab.tasks.registry import register_mjlab_task


def task_id(motion: str) -> str:
  """Task id of one motion: climb is Mjlab-G1-Climb."""
  return "Mjlab-G1-" + "-".join(part.capitalize() for part in motion.split("_"))


CLIMB_TASK_IDS: dict[str, str] = {name: task_id(name) for name in MOTIONS}
"""Motion name to task id. Every one logs to g1_<name>."""

CLIMB_TASK_ID = CLIMB_TASK_IDS["climb"]
"""The one the parkour demo names. See demos/parkour/controller.CLIMB."""


def climb_ppo_runner_cfg(motion: str) -> RslRlOnPolicyRunnerCfg:
  """The jump's PPO config, under one motion's own experiment name.

  Nothing about the algorithm changes, because nothing about the problem does: the same
  tracking objective on the same robot. It runs longer than a martial arts motion because
  the clip is twice as long and half of it is contact with something that is not the floor.
  """
  return replace(jump_ppo_runner_cfg(f"g1_{motion}"), max_iterations=15_000)


for _motion, _task_id in CLIMB_TASK_IDS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=g1_climb_env_cfg(motion_dir(_motion)),
    play_env_cfg=g1_climb_env_cfg(motion_dir(_motion), play=True),
    rl_cfg=climb_ppo_runner_cfg(_motion),
  )
