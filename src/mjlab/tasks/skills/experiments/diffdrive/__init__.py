"""A differential drive robot with two skills whose momentum genuinely couples.

- drive: commanded to go forward at a given (high) speed; it never sees a turning
    command during its own training, so it only knows how to hold a straight line at
    speed. It starts from rest and ramps up toward the target speed.
- turn: commanded to arc to a target heading at a low speed; it never sees a fast
    approach during its own training, so it never has to cope with momentum it did
    not build up itself.

The failure this experiment is built around is a tip-over. turn drives a tight,
fixed-radius arc, whose lateral (centripetal) acceleration is v**2 / R: harmless at
the low speed turn was trained at, but violent if turn is handed the robot while it
still carries drive's cruise speed, enough to roll the (deliberately tall, narrow)
chassis onto its side. A naive hand-off from drive to turn tips; the bridge's job is
to brake the robot into turn's speed regime before handing over.

A simple, scripted controller drives the demonstration: run drive for a fixed
number of steps, signal a switch to turn, hold turn for a fixed number of steps,
signal back to drive, and repeat (see controller.py).

The two skills are available as analytical experts in dynamics.py (recommended) and,
if really wanted, as RL policies trained from the tasks below. RL is discouraged
here: the analytical experts are trivial and reliable, and buy the experiment
nothing over a checkpoint.

    uv run train Mjlab-Diffdrive-Drive
    uv run train Mjlab-Diffdrive-Turn

Watch an analytical expert on its own:

    uv run python -m mjlab.tasks.skills.skill \\
        --task-id Mjlab-Diffdrive-Drive \\
        --factory mjlab.tasks.skills.experiments.diffdrive.dynamics:analytical_drive
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.skills.skill import SkillPool


# Names shared by this experiment's train and demo entry points. EXPERIMENT_NAME
# is the folder architecture checkpoints are saved under; ENTITY_NAME is the scene
# entity the bridges harvest interrupt states from and read state off.
EXPERIMENT_NAME = "diffdrive"
ENTITY_NAME = "robot"

# The two skill tasks. They share one observation/action space; either one's env
# serves as the arena, and the demo/train build it from the drive task.
DRIVE_TASK_ID = "Mjlab-Diffdrive-Drive"
TURN_TASK_ID = "Mjlab-Diffdrive-Turn"

# Analytical skill defaults, shared by train and demo.
DRIVE_SPEED = 3.5  # [m/s], high enough that a naive hand-off to turn tips
TURN_ANGLE = math.pi / 2  # [rad], the 90 deg arc turn
TURN_SPEED = 0.3  # [m/s], the arc's low creep speed


from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.skills.experiments.diffdrive.diffdrive_env_cfg import (
  diffdrive_ppo_runner_cfg,
  drive_env_cfg,
  turn_env_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Diffdrive-Drive",
  env_cfg=drive_env_cfg(),
  play_env_cfg=drive_env_cfg(play=True),
  rl_cfg=diffdrive_ppo_runner_cfg("diffdrive_drive"),
)

register_mjlab_task(
  task_id="Mjlab-Diffdrive-Turn",
  env_cfg=turn_env_cfg(),
  play_env_cfg=turn_env_cfg(play=True),
  rl_cfg=diffdrive_ppo_runner_cfg("diffdrive_turn"),
)


def build_pool(
  env: ManagerBasedRlEnv,
  device: str,
  *,
  analytical: bool = True,
  drive_task_id: str = DRIVE_TASK_ID,
  turn_task_id: str = TURN_TASK_ID,
  drive_checkpoint: str | None = None,
  turn_checkpoint: str | None = None,
  drive_speed: float = DRIVE_SPEED,
  turn_angle: float = TURN_ANGLE,
  turn_speed: float = TURN_SPEED,
) -> SkillPool:
  """The two-skill pool for this experiment: drive (id 0), then turn (id 1).

  Analytical experts by default (recommended). With analytical=False, the frozen
  RL policies are loaded instead; a None checkpoint falls back to the latest
  trained one for that task.
  """

  # Imported lazily so merely importing this package stays cheap (the demo imports
  # it just for the constants above).
  from mjlab.tasks.skills.experiments.diffdrive.dynamics import (
    analytical_drive,
    analytical_turn,
  )
  from mjlab.tasks.skills.skill import PolicySkill, SkillPool
  from mjlab.tasks.skills.utils import retrieve_latest_checkpoint

  if analytical:
    return SkillPool(
      [
        analytical_drive(speed=drive_speed),
        analytical_turn(target_angle=turn_angle, forward_speed=turn_speed),
      ]
    )

  drive_ckpt = drive_checkpoint or retrieve_latest_checkpoint(drive_task_id)
  turn_ckpt = turn_checkpoint or retrieve_latest_checkpoint(turn_task_id)
  return SkillPool(
    [
      PolicySkill("drive", drive_task_id, drive_ckpt, env, device),
      PolicySkill("turn", turn_task_id, turn_ckpt, env, device),
    ]
  )
