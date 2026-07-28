"""Humanoid running a corridor of obstacles, composed of LAFAN1 locomotion skills.

Three goal-conditioned primitive motion skills -- walk, run and jump -- are trained from
LAFAN1 and then frozen. Those three checkpoints are what the bridge composes; the
corridor demo is what they are composed for. (Sprint used to be a fourth. It had one
source clip, a speed band overlapping run's, and nothing the corridor asks of it that run
does not already cover.)

Each skill is a DeepMimic-style tracker with a goal:

- the tracking half pays for reproducing a reference clip frame by frame -- joint pose
  and velocity, end-effector placement, root height and orientation, root velocity, foot
  contact pattern. Nobody writes down what a walk looks like; the clips say;
- the goal half pays for realizing what that clip realizes -- the displacement it
  covers, the heading it turns through, the apex it reaches. That is what makes a trained
  skill *addressable*, so the controller and the bridge can steer it.

Both halves reach the reward, which is the part that is easy to get wrong: a goal wired
into the observation but not into the reward is one the policy can ignore while still
scoring full marks, and the skill then looks conditioned right up until something tries
to steer it.

The pieces:

    dataset.py           cuts LAFAN1 into per-skill clips: normalized, labelled with goals
    motions.py           one skill's clips, loaded flat for batched lookup
    mdp.py               the command term, the reference observations, the rewards
    parkour_env_cfg.py   the three skill envs (one factory, three clip folders)
    controller.py        which skill should run, from where the robot is in the corridor
    train.py / demo.py   the bridging architecture on top of the frozen skills
    style.py             an AMP discriminator, kept for comparison; the env does not use
                         it (the skills track their references directly instead)

Build the reference clips once (needs HF_TOKEN, see dataset.py):

    uv run python -m mjlab.tasks.skills.experiments.parkour.dataset

Check the segmentation without writing anything:

    uv run python -m mjlab.tasks.skills.experiments.parkour.dataset --dry-run True

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

The corridor-with-obstacles environment itself is not built yet; the controller is
written against that described interface (a known obstacle layout plus the robot's
position along the corridor).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.skills.experiments.parkour.parkour_env_cfg import (
  SKILL_NAMES,
  parkour_skill_env_cfg,
)
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.skills.skill import SkillPool

# Names shared by this experiment's train and demo entry points. `EXPERIMENT_NAME` is
# the folder architecture checkpoints are saved under; `ENTITY_NAME` is the scene
# entity the bridges harvest interrupt states from and read state off.
EXPERIMENT_NAME = "parkour"
ENTITY_NAME = "robot"

WALK_TASK_ID = "Mjlab-Parkour-Walk"
RUN_TASK_ID = "Mjlab-Parkour-Run"
JUMP_TASK_ID = "Mjlab-Parkour-Jump"

# Skill name -> its task id. The order is the pool order, so it is also the skill ids
# the controller emits and the bridges are conditioned on (see `build_pool`).
SKILL_TASK_IDS: dict[str, str] = {
  "walk": WALK_TASK_ID,
  "run": RUN_TASK_ID,
  "jump": JUMP_TASK_ID,
}

assert set(SKILL_TASK_IDS) == set(SKILL_NAMES), (
  f"Every skill in parkour_env_cfg needs a task id: {sorted(SKILL_NAMES)}."
)


def _skill_rl_cfg(skill: str) -> RslRlOnPolicyRunnerCfg:
  """The G1 *tracking* PPO config, under this skill's own experiment name so its
  checkpoints land in their own `logs/rsl_rl/<experiment_name>/` directory.

  The tracking config rather than the velocity one, because this is a tracking task: the
  observation is a few hundred dimensions of proprioception, reference and goal in
  thoroughly mixed units, which wants both the wider network and the input normalization
  the tracking config turns on and the velocity one does not.

  The tracking task's own runner is *not* used. It bakes one `MotionCommand`'s single
  motion file into the ONNX export, and a skill here is conditioned on a library of
  clips plus a goal, so there is no one motion to bundle. The default runner exports the
  policy alone, which is all the composition ever loads.
  """
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = f"parkour_{skill}"
  return cfg


for _skill, _task_id in SKILL_TASK_IDS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=parkour_skill_env_cfg(_skill),
    play_env_cfg=parkour_skill_env_cfg(_skill, play=True),
    rl_cfg=_skill_rl_cfg(_skill),
  )


def build_pool(
  env: ManagerBasedRlEnv,
  device: str,
  *,
  checkpoints: dict[str, str] | None = None,
) -> SkillPool:
  """The three-skill pool: walk (0), run (1), jump (2).

  Each skill is a frozen checkpoint of the matching task. `checkpoints` overrides
  where one is loaded from, keyed by skill name; any skill left out falls back to the
  latest trained checkpoint for its task. (There is no analytical variant here --
  these skills are learned.)

  Every skill is evaluated against the one `env` the caller built, which is sound
  because all three tasks share an observation and action space by construction (see
  parkour_env_cfg).
  """
  from mjlab.tasks.skills.skill import PolicySkill, SkillPool
  from mjlab.tasks.skills.utils import retrieve_latest_checkpoint

  checkpoints = checkpoints or {}
  skills = []
  for skill, task_id in SKILL_TASK_IDS.items():
    checkpoint = checkpoints.get(skill) or retrieve_latest_checkpoint(task_id)
    skills.append(PolicySkill(skill, task_id, checkpoint, env, device))
  return SkillPool(skills)
