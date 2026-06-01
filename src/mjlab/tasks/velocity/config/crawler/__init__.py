from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import crawler_velocity_env_cfg
from .rl_cfg import crawler_ppo_runner_cfg

"""
register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Crawler",
    env_cfg=crawler_velocity_env_cfg(),
    play_env_cfg=crawler_velocity_env_cfg(play=True),
    rl_cfg=crawler_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
"""

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Crawler",
  env_cfg=crawler_velocity_env_cfg(),
  play_env_cfg=crawler_velocity_env_cfg(play=True),
  rl_cfg=crawler_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
