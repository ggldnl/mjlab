"""Crawler velocity task registration.

Registers two flat-terrain velocity-tracking tasks for the crawler quadruped:
a classic reward-shaped task and an abstraction-guided variant.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  crawler_velocity_abstraction_env_cfg,
  crawler_velocity_flat_env_cfg,
)
from .rl_cfg import (
  crawler_abstraction_ppo_runner_cfg,
  crawler_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Crawler",
  env_cfg=crawler_velocity_flat_env_cfg(),
  play_env_cfg=crawler_velocity_flat_env_cfg(play=True),
  rl_cfg=crawler_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Crawler-Abstraction",
  env_cfg=crawler_velocity_abstraction_env_cfg(),
  play_env_cfg=crawler_velocity_abstraction_env_cfg(play=True),
  rl_cfg=crawler_abstraction_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
