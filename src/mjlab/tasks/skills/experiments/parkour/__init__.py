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

Score a bridge on the hand-over itself, walk into jump, rather than on a corpus window:

    uv run python -m mjlab.tasks.skills.experiments.parkour.tests.walk2jump

Problem: the controller calls the jump on distance alone, so the jump is handed
a robot that has been running. The jump is a motion-tracking policy whose clip
begins from a stand, and its reference is pinned to wherever the robot is when
control arrives (see skills.py). A robot arriving at 4 m/s is being told
to reproduce a crouch it has no way to reproduce.

Training as we do now won't work: on a cartpole random actions are survivable,
same on a two-wheeled robot; on a 29-joint humanoid, a random policy falls over ù
immediately. It is standard practice with adversarial imitation on high-joint-count
robots to warm-start the robot, precisely because random exploration can't discover
anything by flailing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.tasks.skills.architectures import Budgets
from mjlab.tasks.skills.architectures.arch_1.config import (
  BridgePhase,
  BridgeTraining,
  SwitchPhase,
)
from mjlab.tasks.skills.architectures.arch_3.config import ResidualTraining
from mjlab.tasks.skills.experiments.parkour.jump import JUMP_TASK_ID
from mjlab.tasks.skills.experiments.parkour.run import RUN_TASK_ID
from mjlab.tasks.skills.experiments.parkour.walk import WALK_TASK_ID
from mjlab.tasks.skills.view import StateViewCfg
from mjlab.tasks.skills.windows import SkillWindowSpec, WindowPlan

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.skills.experiment import Experiment
  from mjlab.tasks.skills.skill import SkillPool

# EXPERIMENT_NAME is the folder architecture checkpoints are saved under; ENTITY_NAME
# is the scene entity the bridges harvest interrupt states from and read state off.
EXPERIMENT_NAME = "parkour"
ENTITY_NAME = "robot"

__all__ = [
  "BRIDGE_VIEW",
  "BUDGETS",
  "ENTITY_NAME",
  "EXPERIMENT_NAME",
  "JUMP_TASK_ID",
  "RUN_TASK_ID",
  "WALK_TASK_ID",
  "WINDOWS",
  "build_experiment",
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

# How long to train. The state is 29 joints, the skills are learned rather than analytic,
# and a bad bridge here means falling over. We should train for longer.
#
# `max_transition_steps` is 96 (about 2 s), long enough for the bridge to actually shed
# a run's momentum.
#
# `eval_steps` is 160, and it is the jump's clip that sets it rather than any judgment
# about how long a failure takes to develop. A hand-over into the jump is judged on
# whether the clip got where it was going (see `_make_jump_success` in train.py), so the
# measurement has to fall after the clip lands and before it runs out. Measured on the
# current checkpoints: the clips are 177 to 227 steps long and touch down between steps
# 90 and 141. At 128 steps, which is what this used to be, only just over half of them
# had landed and the goal error was still mid-flight at half a metre; at 160 all of them
# have landed and the error has settled to under 0.2 m at the ninetieth percentile. Past
# about 200 the reference ends, the policy is left tracking nothing, and episodes start
# being lost for reasons that have nothing to do with the hand-over.
#
# `entropy_coef` is a tenth of the family default. Measured, not guessed: at 0.01 the
# actor's std climbs about a percent per iteration and never stops, because a bridge
# that cannot yet move the discriminator has a surrogate term far weaker than the
# entropy bonus. Over 3000 iterations that is `exp` of a large number, and the run dies
# in PPO with a NaN std. `action_max_std` now bounds it either way, but a policy parked
# at the bound is exploring at random rather than learning, so the coefficient is the
# thing to set. Watch `std` in the phase 1 log: if it still climbs to the bound and
# stays, lower this again.
_BRIDGE = BridgeTraining(
  bridge=BridgePhase(
    num_iterations=100,
    num_windows=2048,
    num_interrupts=8192,
    entropy_coef=0.001,
  ),
  switch=SwitchPhase(
    num_iterations=200,
    num_interrupts=8192,
    max_transition_steps=96,
    eval_steps=160,
    epsilon_decay_iterations=1000,
  ),
)

# arch_3's fade range is arch_1's `max_transition_steps` by another name, and its tail is
# `eval_steps`: both are set by the jump's clip, as explained above. `entropy_coef` is
# lowered for the same reason it is in phase 1 here.
BUDGETS = Budgets(
  arch_1=_BRIDGE,
  arch_2=_BRIDGE,
  arch_3=ResidualTraining(
    num_iterations=200,
    num_windows=2048,
    steps=(32, 96),
    tail_steps=160,
    inference_steps=64,
    entropy_coef=0.001,
  ),
  # arch_4_backup has never been run on this experiment; its defaults stand until it is.
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


def build_experiment(env: ManagerBasedRlEnv, device: str, **pool_kwargs) -> Experiment:
  """Everything an architecture needs from this experiment, in one object."""
  from mjlab.tasks.skills.experiment import Experiment

  return Experiment(
    name=EXPERIMENT_NAME,
    entity_name=ENTITY_NAME,
    pool=build_pool(env, device, **pool_kwargs),
    view=BRIDGE_VIEW.resolve(env),
    windows=WINDOWS,
  )
