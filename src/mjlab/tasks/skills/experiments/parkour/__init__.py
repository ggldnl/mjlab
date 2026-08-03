"""Humanoid running a corridor with obstacles.

Three goal-conditioned primitive motion skills (walk, run and jump) are trained and
then frozen. Those three checkpoints are what the bridge composes.

Build the reference clips once:

    uv run --with joblib python -m mjlab.tasks.skills.experiments.parkour.jump.dataset

Train the skills:

    uv run train Mjlab-Parkour-Walk
    uv run train Mjlab-Parkour-Run
    uv run train Mjlab-Parkour-Jump

Watch a trained skill:

    uv run play Mjlab-Parkour-Jump

The controller decides the skill purely from where the robot is along the corridor,
whose obstacle positions and shapes are known (see controller.py):
- the robot is within a given distance in front of an obstacle
    -> start the jump skill;
- the robot has surpassed the obstacle
    -> start the walk skill again;
- the robot has nothing in front of it for a while
    -> start running.

Then compose them:

    uv run python -m mjlab.tasks.skills.experiments.parkour.demo
    uv run python -m mjlab.tasks.skills.experiments.parkour.train --architecture 1
    uv run python -m mjlab.tasks.skills.experiments.parkour.demo --architecture 1

The failure the bridge exists to remove is the one from the problem statement, and it
is here rather than by construction: the controller calls the jump on distance alone,
so the jump is handed a robot that has been running. The jump is a motion-tracking
policy whose clip begins from a stand, and its reference is pinned to wherever the
robot is when control arrives (see skills.py). A robot arriving at 4 m/s is being told
to reproduce a crouch it has no way to reproduce.

The pieces, in the order they matter:

    arena.py       the shared environment: corridor, and every skill's observation
    skills.py      the three frozen policies, wired to their own observation groups
    controller.py  the corridor rule above
    demo.py        run the composition, watch it or count the failures
    train.py       train a bridging architecture on it
    inspect.py     look at the windows a bridge is trained on

    walk/  run/  jump/   the three skills, each a registered task of its own
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.tasks.skills.architectures.arch_1.config import (
  BridgePhase,
  BridgeTraining,
  SwitchPhase,
)
from mjlab.tasks.skills.experiments.parkour.jump import JUMP_TASK_ID
from mjlab.tasks.skills.experiments.parkour.run import RUN_TASK_ID
from mjlab.tasks.skills.experiments.parkour.walk import WALK_TASK_ID
from mjlab.tasks.skills.view import StateViewCfg
from mjlab.tasks.skills.windows import SkillWindowSpec, WindowPlan

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.skills.skill import SkillPool

# EXPERIMENT_NAME is the folder architecture checkpoints are saved under; ENTITY_NAME
# is the scene entity the bridges harvest interrupt states from and read state off.
EXPERIMENT_NAME = "parkour"
ENTITY_NAME = "robot"

__all__ = [
  "BRIDGE_VIEW",
  "ENTITY_NAME",
  "EXPERIMENT_NAME",
  "JUMP_TASK_ID",
  "RUN_TASK_ID",
  "TRAINING",
  "WALK_TASK_ID",
  "WINDOWS",
  "build_pool",
]

# What the bridge is allowed to see.
#
# Unlike the diffdrive, this needs no `drop` list, because the arena already provides a
# group that is nothing but proprioception. That is not a convenience: the three skills
# do not share an observation at all, so there is no single task vector to carve a view
# out of. `jump_actor` is two thirds reference clip and `velocity_actor` carries a
# commanded twist, and a discriminator handed either would separate a bridge from its
# target on the task signal alone without ever looking at the robot. The arena's `actor`
# group is base velocity, gravity, joint state and last action -- the robot, and nothing
# about what anyone was asked to do. See arena.py and view.py.
BRIDGE_VIEW = StateViewCfg()

# How much of each skill is recorded around a hand-over.
#
# Control is 50 Hz, so 64 steps is 1.28 s. The velocity skills are periodic: a G1 stride
# is roughly 0.6 s at a walk and shorter at a run, so a window of this length holds two
# or more full strides, and a bridge matching it has to match a gait rather than a pose.
#
# The interrupt ranges are where the two kinds of skill differ. Walk and run reach a
# steady gait within a second or so of their reset and then repeat it, so anywhere past
# that is representative and the range is wide. Jump does not: its clip is a single
# 3.5 to 4.5 s event whose parts are all different, and a hand-over sampled uniformly
# across it would mostly land in the stand at either end. Its range covers the run-up
# and the crouch, which is where a hand-over into it actually has to work.
#
# `overrun` is zero throughout: no architecture here asks what a skill would have gone
# on to do had control not been taken away.
WINDOWS = WindowPlan(
  {
    "walk": SkillWindowSpec(
      opening=64, closing=64, overrun=0, interrupt_range=(64, 256)
    ),
    "run": SkillWindowSpec(
      opening=64, closing=64, overrun=0, interrupt_range=(64, 256)
    ),
    "jump": SkillWindowSpec(
      opening=128, closing=64, overrun=0, interrupt_range=(64, 160)
    ),
  }
)

# How long to train. An order of magnitude above the diffdrive's budget, because a
# humanoid is not a two-wheeled cart: the state is 29 joints rather than two wheels, the
# skills are learned rather than analytic, and a bad bridge here means falling over
# rather than a tipped chassis.
#
# `max_transition_steps` is 96 (about 2 s), long enough for the bridge to actually shed
# a run's momentum. `eval_steps` is 128, long enough to see a botched hand-off into the
# jump land: the clip takes roughly a second to reach its takeoff, and a robot that
# mistimes it falls during the second after that.
#
# `entropy_coef` is a tenth of the family default. Measured, not guessed: at 0.01 the
# actor's std climbs about a percent per iteration and never stops, because a bridge
# that cannot yet move the discriminator has a surrogate term far weaker than the
# entropy bonus. Over 3000 iterations that is `exp` of a large number, and the run dies
# in PPO with a NaN std. `action_max_std` now bounds it either way, but a policy parked
# at the bound is exploring at random rather than learning, so the coefficient is the
# thing to set. Watch `std` in the phase 1 log: if it still climbs to the bound and
# stays, lower this again.
TRAINING = BridgeTraining(
  bridge=BridgePhase(
    num_iterations=3000,
    num_windows=2048,
    num_interrupts=8192,
    entropy_coef=0.001,
  ),
  switch=SwitchPhase(
    num_iterations=4000,
    num_interrupts=8192,
    max_transition_steps=96,
    eval_steps=128,
    epsilon_decay_iterations=1000,
  ),
)


def build_pool(
  env: ManagerBasedRlEnv,
  device: str,
  *,
  walk_task_id: str = WALK_TASK_ID,
  run_task_id: str = RUN_TASK_ID,
  jump_task_id: str = JUMP_TASK_ID,
  walk_checkpoint: str | None = None,
  run_checkpoint: str | None = None,
  jump_checkpoint: str | None = None,
) -> SkillPool:
  """The three-skill pool: walk (id 0), run (id 1), jump (id 2).

  The order fixes the ids the controller emits and the bridge is conditioned on, so
  it is part of the experiment; controller.py names them.

  There are no analytical stand-ins here, unlike the diffdrive and the cart-pole.
  Nothing about a humanoid gait or a humanoid jump is worth hand-writing, so every
  skill is a frozen checkpoint and all three have to be trained before the
  composition can run at all.
  """
  # Imported lazily so that merely importing this package (which the demo does just
  # for the constants above) does not drag in torch and the whole RL stack.
  from mjlab.tasks.skills.experiments.parkour.arena import (
    JUMP_CRITIC_GROUP,
    JUMP_OBS_GROUP,
    VELOCITY_CRITIC_GROUP,
    VELOCITY_OBS_GROUP,
  )
  from mjlab.tasks.skills.experiments.parkour.skills import ArenaSkill, JumpSkill
  from mjlab.tasks.skills.skill import SkillPool
  from mjlab.tasks.skills.utils import retrieve_latest_checkpoint

  walk_ckpt = walk_checkpoint or retrieve_latest_checkpoint(walk_task_id)
  run_ckpt = run_checkpoint or retrieve_latest_checkpoint(run_task_id)
  jump_ckpt = jump_checkpoint or retrieve_latest_checkpoint(jump_task_id)

  return SkillPool(
    [
      ArenaSkill(
        "walk",
        walk_task_id,
        walk_ckpt,
        env,
        device,
        obs_group=VELOCITY_OBS_GROUP,
        critic_group=VELOCITY_CRITIC_GROUP,
      ),
      ArenaSkill(
        "run",
        run_task_id,
        run_ckpt,
        env,
        device,
        obs_group=VELOCITY_OBS_GROUP,
        critic_group=VELOCITY_CRITIC_GROUP,
      ),
      JumpSkill(
        "jump",
        jump_task_id,
        jump_ckpt,
        env,
        device,
        obs_group=JUMP_OBS_GROUP,
        critic_group=JUMP_CRITIC_GROUP,
      ),
    ]
  )
