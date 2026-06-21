"""
Skill-bridging: compose analytic skills with a transition bridge.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.skills.experiments.diffdrive.bridge.bridge_env_cfg import (
  BridgeOnPolicyRunner,
  bridge_env_cfg,
  bridge_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Bridge-Diffdrive",
  env_cfg=bridge_env_cfg(),
  play_env_cfg=bridge_env_cfg(play=True),
  rl_cfg=bridge_ppo_runner_cfg(),
  runner_cls=BridgeOnPolicyRunner,
)
