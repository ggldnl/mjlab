"""Humanoid running a corridor of obstacles, composed from LAFAN1 locomotion skills.

Goal-conditioned locomotion skills -- walk, run (sprint) and jump -- are extracted
from the LAFAN1 dataset and trained as motion-tracking policies, reusing mjlab's
tracking task as-is: each skill is the G1 flat tracking env pointed at its own clip.
The three frozen checkpoints are the skills the bridge composes.

Building the skill clips (once):

    # 1. segment LAFAN1 into command plateaus (needs HF_TOKEN; see build_dataset.py)
    uv run python -m mjlab.tasks.skills.experiments.parkour.build_dataset
    # 2. convert the chosen walk / run / jump segments to motion npz with
    #    mjlab.scripts.csv_to_npz, saving them where the tasks below expect them
    #    (data/lafan1_g1/skills/{walk,run,jump}.npz). You may instead point the
    #    tracking training at any clip with --env.commands.motion.motion-file or a
    #    WandB --registry-name, exactly as any mjlab tracking task.

Training the individual skills (standard mjlab tracking training):

    uv run train Mjlab-Parkour-Walk
    uv run train Mjlab-Parkour-Run
    uv run train Mjlab-Parkour-Jump

The controller decides the skill purely from where the robot is along the corridor,
whose obstacle positions and shapes are known (see controller.py):
- the robot is within a given distance in front of an obstacle
    -> start the jump skill;
- the robot has surpassed the obstacle
    -> start the walk skill again;
- the robot has nothing in front of it for a while
    -> start sprinting (run).

The corridor-with-obstacles environment itself is not built yet; the controller is
written against that described interface (a known obstacle layout plus the robot's
position along the corridor).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
  from mjlab.tasks.skills.skill import SkillPool

# Names shared by this experiment's train and demo entry points. `EXPERIMENT_NAME` is
# the folder architecture checkpoints are saved under; `ENTITY_NAME` is the scene
# entity the bridges harvest interrupt states from and read state off.
EXPERIMENT_NAME = "parkour"
ENTITY_NAME = "robot"

WALK_TASK_ID = "Mjlab-Parkour-Walk"
RUN_TASK_ID = "Mjlab-Parkour-Run"
JUMP_TASK_ID = "Mjlab-Parkour-Jump"

# Per-skill motion clips (npz). Built from LAFAN1 as described above; point these at
# your generated files (or override the motion_file at train time). The clips only
# need to exist to *build the env* (train the skill or load it into the pool) -- the
# registrations below are cheap and do not read them.
_MOTION_ROOT = Path("data/lafan1_g1/skills")
WALK_MOTION = str(_MOTION_ROOT / "walk.npz")
RUN_MOTION = str(_MOTION_ROOT / "run.npz")
JUMP_MOTION = str(_MOTION_ROOT / "jump.npz")


def _skill_env_cfg(motion_file: str, play: bool = False) -> ManagerBasedRlEnvCfg:
  """The G1 flat tracking env, pointed at one skill's clip."""
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = motion_file
  return cfg


def _skill_rl_cfg(experiment_name: str):
  """The G1 tracking PPO config, under this skill's own experiment name so its
  checkpoints land in their own `logs/rsl_rl/<experiment_name>/` directory."""
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = experiment_name
  return cfg


for _task_id, _motion, _experiment in (
  (WALK_TASK_ID, WALK_MOTION, "parkour_walk"),
  (RUN_TASK_ID, RUN_MOTION, "parkour_run"),
  (JUMP_TASK_ID, JUMP_MOTION, "parkour_jump"),
):
  register_mjlab_task(
    task_id=_task_id,
    env_cfg=_skill_env_cfg(_motion),
    play_env_cfg=_skill_env_cfg(_motion, play=True),
    rl_cfg=_skill_rl_cfg(_experiment),
    runner_cls=MotionTrackingOnPolicyRunner,
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
  """The three-skill pool for this experiment: walk (id 0), run (id 1), jump (id 2).

  Each skill is a frozen motion-tracking checkpoint; a `None` checkpoint falls back to
  the latest trained one for that task. (There is no analytical variant here -- these
  skills are learned.)
  """
  from mjlab.tasks.skills.skill import PolicySkill, SkillPool
  from mjlab.tasks.skills.utils import retrieve_latest_checkpoint

  walk_ckpt = walk_checkpoint or retrieve_latest_checkpoint(walk_task_id)
  run_ckpt = run_checkpoint or retrieve_latest_checkpoint(run_task_id)
  jump_ckpt = jump_checkpoint or retrieve_latest_checkpoint(jump_task_id)
  return SkillPool(
    [
      PolicySkill("walk", walk_task_id, walk_ckpt, env, device),
      PolicySkill("run", run_task_id, run_ckpt, env, device),
      PolicySkill("jump", jump_task_id, jump_ckpt, env, device),
    ]
  )
