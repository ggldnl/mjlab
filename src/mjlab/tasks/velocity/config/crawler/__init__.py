"""Crawler velocity task registration.

Registers a flat-terrain velocity task for the crawler quadruped, trained by
*gait-guided* RL: standard PPO with a dense foot-trajectory reference reward
derived from the open-loop gait (see ``env_cfgs``). Uses the standard
``VelocityOnPolicyRunner``. The behavioral-cloning ``CrawlerDistillRunner`` in
``distill_runner.py`` is kept but not used.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import crawler_velocity_flat_env_cfg
from .rl_cfg import crawler_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Crawler",
  env_cfg=crawler_velocity_flat_env_cfg(),
  play_env_cfg=crawler_velocity_flat_env_cfg(play=True),
  rl_cfg=crawler_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
