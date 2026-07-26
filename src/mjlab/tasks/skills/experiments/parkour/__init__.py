"""Humanoid running a corridor of obstacles, composed of LAFAN1 locomotion skills.

Four goal-conditioned skills (walk, run, jump, sprint) are trained from LAFAN1
and then frozen. Those four checkpoints are what the bridge composes; the corridor
demo is what they are composed for.

Each skill is trained with a goal reward plus a style reward, decoupled as in "AMP:
Adversarial Motion Priors for Stylized Physics-Based Character Control":

- the goal is a velocity the policy observes and is paid for realizing, so a trained
  skill can be *driven* (the controller and the bridge steer it by writing that
  command);
- the style is a discriminator trained against that skill's own reference clips, so
  "run" means the motion is indistinguishable from the LAFAN1 run cuts rather than
  matching one clip frame by frame.

The pieces:

    dataset.py           cuts LAFAN1 into per-skill reference clips with velocity labels
    style.py             the AMP features and the discriminator
    mdp.py               the command term and the goal/style reward terms
    parkour_env_cfg.py   the four skill envs (one factory, four command boxes)
    controller.py        which skill should run, from where the robot is in the corridor
    train.py / demo.py   the bridging architecture on top of the frozen skills

Build the reference clips once (needs HF_TOKEN, see dataset.py):

    uv run python -m mjlab.tasks.skills.experiments.parkour.dataset

Train the skills:

    uv run train Mjlab-Parkour-Walk
    uv run train Mjlab-Parkour-Run
    uv run train Mjlab-Parkour-Jump
    uv run train Mjlab-Parkour-Sprint

Drive a trained skill by hand. The Twist folder in the viewer has a slider per
command channel; tick Enable and move v_fwd, or v_up for jump:

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
position along the corridor), and it does not yet emit the sprint skill.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.skills.experiments.parkour.parkour_env_cfg import (
  SKILL_NAMES,
  parkour_skill_env_cfg,
)
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

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
SPRINT_TASK_ID = "Mjlab-Parkour-Sprint"

# Skill name -> its task id. The order is the pool order, so it is also the skill ids
# the controller emits and the bridges are conditioned on (see `build_pool`).
SKILL_TASK_IDS: dict[str, str] = {
  "walk": WALK_TASK_ID,
  "run": RUN_TASK_ID,
  "jump": JUMP_TASK_ID,
  "sprint": SPRINT_TASK_ID,
}

assert set(SKILL_TASK_IDS) == set(SKILL_NAMES), (
  f"Every skill in parkour_env_cfg needs a task id: {sorted(SKILL_NAMES)}."
)


def _skill_rl_cfg(skill: str) -> RslRlOnPolicyRunnerCfg:
  """The G1 velocity PPO config, under this skill's own experiment name so its
  checkpoints land in their own `logs/rsl_rl/<experiment_name>/` directory."""
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = f"parkour_{skill}"
  return cfg


for _skill, _task_id in SKILL_TASK_IDS.items():
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=parkour_skill_env_cfg(_skill),
    play_env_cfg=parkour_skill_env_cfg(_skill, play=True),
    rl_cfg=_skill_rl_cfg(_skill),
    runner_cls=VelocityOnPolicyRunner,
  )


def build_pool(
  env: ManagerBasedRlEnv,
  device: str,
  *,
  checkpoints: dict[str, str] | None = None,
) -> SkillPool:
  """The four-skill pool: walk (0), run (1), jump (2), sprint (3).

  Each skill is a frozen checkpoint of the matching task. `checkpoints` overrides
  where one is loaded from, keyed by skill name; any skill left out falls back to the
  latest trained checkpoint for its task. (There is no analytical variant here --
  these skills are learned.)

  Every skill is evaluated against the one `env` the caller built, which is sound
  because all four tasks share an observation and action space by construction (see
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
