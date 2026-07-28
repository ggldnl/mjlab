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

from mjlab.tasks.skills.architectures.arch_1.config import (
  BridgePhase,
  BridgeTraining,
  SwitchPhase,
)
from mjlab.tasks.skills.view import StateViewCfg
from mjlab.tasks.skills.windows import SkillWindowSpec, WindowPlan

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
DRIVE_SPEED = 2.5  # [m/s], high enough that a naive hand-off to turn tips
TURN_ANGLE = math.pi / 2  # [rad], the 90 deg arc turn
TURN_SPEED = 0.3  # [m/s], the arc's low creep speed


# What the bridge is allowed to see, and therefore what it is compared against.
#
# The observation is `base_lin_vel`, `base_ang_vel`, `wheel_vel`, `actions`, `command`.
# The first four describe the robot; `command` describes the task, and it is the one the
# bridge must not be judged on. Its second channel is the live error to a heading target
# fixed when the episode began, so it reads ~0 in every frame of drive's own recordings
# (they all start from a reset) and ~pi/2 wherever turn hands the robot over. A
# discriminator given that channel separates the two halves on it alone, never looks at
# the motion, and the only way for the bridge to close the gap is to steer the robot back
# to the heading the episode started at. That is not bridging, it is navigating, and it
# is what a bridge trained on the full observation actually learns to do -- the hand-over
# it produces ends up worse than the naive one it was meant to fix. Its first channel is
# no better: a per-episode speed target the analytical experts ignore entirely, so it is
# pure noise the bridge cannot act on.
#
# What is left is exactly the pair of statements the bridge needs: "the robot is moving
# this fast, turning this hard, with the wheels here" for the state, and drive's or
# turn's own opening as the thing to look like. Which reduces the problem to "slow down",
# which is what it should have been all along.
BRIDGE_VIEW = StateViewCfg(drop=("command",))

# How much of each skill is recorded around a hand-over. Shared by every architecture's
# training and by inspect.py.
#
# Control is 50 Hz, so 32 steps is 0.64 s: long enough to see the robot doing something
# rather than sitting in one pose. The two skills get different cut ranges because they
# reach their interesting behavior at different times.
#
# drive ramps at 2.5 m/s^2, so it passes 1.6 m/s at step 32 and saturates at its 2.5 m/s
# cruise around step 50. Cutting between 32 and 96 hands the bridge a robot carrying
# between two thirds and all of drive's momentum, which is the whole range it has to
# cope with. Cutting later would only repeat the cruise, and cutting earlier would hand
# it a robot barely moving, which turn can already take on its own.
#
# turn finishes its 90 degree arc in about 40 steps and then just creeps, so its range
# stops there: past that, every cut is the same slow forward crawl. Its closing window
# is shorter so the range can start early enough to catch the arc while it is still
# happening.
#
# Neither range matches when the demo controller actually switches (300 steps of drive,
# 150 of turn). It does not have to, because both skills reach a steady state well
# before then and every later cut is a repeat of one already covered: drive is at cruise
# by step 50, turn is creeping by step 45. A skill that never settles would need a range
# that reaches as far as the controller really goes.
#
# `overrun` is zero for both: no architecture here asks to see what a skill would have
# gone on to do had control not been taken away. Raise it for one that does.
WINDOWS = WindowPlan(
  {
    "drive": SkillWindowSpec(
      opening=64, closing=64, overrun=0, interrupt_range=(64, 128)
    ),
    "turn": SkillWindowSpec(
      opening=64, closing=64, overrun=0, interrupt_range=(64, 128)
    ),
  }
)

# How long to train. Small, because this experiment is small: both skills are analytical
# and the bridge has one thing to learn. A humanoid would want an order of magnitude
# more of everything here.
#
# eval_steps is set by the slower of the two hand-off directions. Handing into turn at
# speed tips within 24 to 32 steps. Handing into drive is slower to go wrong and less
# obvious: the wheel command is acceleration-limited per wheel, so drive ramping both
# wheels up from turn's asymmetric speeds preserves the difference between them for the
# ~47 steps it takes to reach cruise, and the robot arcs harder and harder until it
# rolls.
TRAINING = BridgeTraining(
  bridge=BridgePhase(num_iterations=300, num_windows=512, num_interrupts=4096),
  switch=SwitchPhase(
    num_iterations=600,
    num_interrupts=4096,
    max_transition_steps=64,
    eval_steps=96,
    epsilon_decay_iterations=150,
  ),
)


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
