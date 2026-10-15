"""G1 turns by a relative angle or to a world heading.

Train: uv run train Mjlab-G1-Turn --agent.run-name sokoban_turn
Play:  uv run play Mjlab-G1-Turn
"""

from dataclasses import replace

from mjlab.tasks.bridging.config.t1.skills.turn import make_turn_env_cfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

TURN_TASK_ID = "Mjlab-G1-Turn"

runner = replace(unitree_g1_ppo_runner_cfg(), experiment_name="g1_turn")
assert runner.actor.distribution_cfg is not None
runner.actor.distribution_cfg["init_std"] = 0.5

register_mjlab_task(
  task_id=TURN_TASK_ID,
  env_cfg=make_turn_env_cfg(unitree_g1_flat_env_cfg()),
  play_env_cfg=make_turn_env_cfg(unitree_g1_flat_env_cfg(play=True), play=True),
  rl_cfg=runner,
)
