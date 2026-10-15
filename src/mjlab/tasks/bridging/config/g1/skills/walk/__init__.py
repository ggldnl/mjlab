"""G1 velocity walking and forward walking for Sokoban.

Mjlab-G1-Walk keeps the stock velocity task and logs under g1_walk.
Mjlab-G1-Walk-Straight samples forward speed and standing only, and logs under
g1_walk_straight.

Run

    uv run train Mjlab-G1-Walk --env.scene.num-envs 4096
    uv run train Mjlab-G1-Walk-Straight --env.scene.num-envs 4096

"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

WALK_TASK_ID = "Mjlab-G1-Walk"
STRAIGHT_WALK_TASK_ID = "Mjlab-G1-Walk-Straight"


def straight_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Forward walking and standing for Sokoban"""
  cfg = unitree_g1_flat_env_cfg(play=play)
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  cfg.commands["twist"] = replace(
    twist,
    heading_command=False,
    rel_heading_envs=0.0,
    rel_world_envs=0.0,
    rel_forward_envs=0.0,
    ranges=UniformVelocityCommandCfg.Ranges((0.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
  )
  cfg.curriculum.pop("command_vel", None)
  return cfg


def walk_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The stock G1 velocity PPO config, under this experiment's own name."""
  return replace(unitree_g1_ppo_runner_cfg(), experiment_name="g1_walk")


register_mjlab_task(
  task_id=WALK_TASK_ID,
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=walk_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=STRAIGHT_WALK_TASK_ID,
  env_cfg=straight_walk_env_cfg(),
  play_env_cfg=straight_walk_env_cfg(play=True),
  rl_cfg=replace(unitree_g1_ppo_runner_cfg(), experiment_name="g1_walk_straight"),
  runner_cls=VelocityOnPolicyRunner,
)
