"""Push a 1 m box a variable number of whole cells straight ahead, hands only.

The robot starts at the center of a cell facing the rear face of the box in the next cell.
The push goal is a destination one to four cells ahead, latched at reset. The policy reads
what is left of it every step, never a twist, so it cannot be told to sidestep or turn.

    command.py        the destination and the reference speed derived from it
    mdp.py            observations, rewards, terminations and the box reset
    push_env_cfg.py   builder shared with the G1, and the T1 robot numbers

Command, observed

    remaining     distance left along the push axis, m
    lateral       box offset from the push line, m, positive left
    axis          push axis in the robot heading frame, cos and sin
    speed         reference box speed: ramps up, holds push_speed, brakes onto the target

Gait

The walking rewards of the flat velocity task stay on. Their twist is the pace command,
speed forward and a yaw rate steering back onto the axis, so the robot is paid for a
steady walk at the box speed and to stand once the box is in place. A cost on both feet
leaving the ground rules out hopping.

Arms

Straight forward is the default pose, so it is the posture target and the zero action.
The body clearance cost keeps the trunk an arm length off the box, and any contact other
than the hands is penalized.

What to watch

    Metrics/push/speed_error           box speed against the reference
    Metrics/push/position_error        distance to the target at episode end
    Metrics/push/settled_at_goal       box settled on the target
    Episode_Metrics/push_hands_contact fraction of the hands on the box
    Episode_Metrics/push_body_contact  illegal contact, should fall to zero
    Episode_Metrics/push_flight        both feet airborne, should fall to zero

Run

1. Train.

    uv run train Mjlab-T1-Push --env.scene.num-envs 4096

2. Play the latest checkpoint, or a fixed distance.

    uv run play Mjlab-T1-Push
    uv run play Mjlab-T1-Push --env.commands.push.min-cells 3 --env.commands.push.max-cells 3
"""

from dataclasses import replace

from mjlab.tasks.bridging.config.t1.skills.push.push_env_cfg import t1_push_env_cfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.t1.rl_cfg import booster_t1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

PUSH_TASK_ID = "Mjlab-T1-Push"

register_mjlab_task(
  task_id=PUSH_TASK_ID,
  env_cfg=t1_push_env_cfg(),
  play_env_cfg=t1_push_env_cfg(play=True),
  rl_cfg=replace(
    booster_t1_ppo_runner_cfg(), experiment_name="t1_push", max_iterations=15000
  ),
  runner_cls=VelocityOnPolicyRunner,
)
