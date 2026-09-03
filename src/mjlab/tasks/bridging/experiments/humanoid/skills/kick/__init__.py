"""Kicking: a standing G1 strikes a football with its toe, at a commanded launch velocity.

Run:

    uv run train Mjlab-G1-Kick --env.scene.num-envs 4096
    uv run play Mjlab-G1-Kick

The same command as the pass, a launch speed and a heading offset, and the same reward for
answering it. What differs is the motion allowed to satisfy it. The approach is measured
from the toe, meeting the ball with the foot on the floor costs, meeting it with a toe that
is not moving costs, and a rung of reward is paid for the ball reaching the speed it was
asked for. See kick_env_cfg.py for the differences and mdp.py for the two sites and what
they measure.

This exists because the pass did not turn out to be a kick. Trained on a ball inside a
foot's reach and scored on nothing but the ball's velocity, the policy found a low shove
with the sole, which is a legitimate way to move a ball and is not the motion the skill was
meant to demonstrate. Rather than retune the pass into something else, the pass is kept as
what it is and this is the version that asks for a strike.

Two earlier attempts are worth knowing about, because both look reasonable written down.

Asking for the strike geometrically, by putting the ball out near the edge of a standing
robot's reach so nothing but a swing could touch it, trained through every curriculum stage
and learned to fall over. A dense position kernel whose maximum sits at the edge of the
reach envelope points at the balance boundary, and a policy following it goes past.

Asking for it with a height gate on the contact charged the motion it was trying to buy. A
toe kick of a ball resting on the floor meets it below the ball's centre with the foot low,
so the gate fired on real kicks while a heel planted, toes up shovel passed it untouched.

What rules out the shove now is that the foot has to be off the ground. That is the one
test a flat footed push cannot pass, and unlike the other two it does not constrain the
strike itself at all.

The observation is eight numbers wider than the pass's, so the pass's checkpoint will not
load and should not: it holds a policy that solved the easier problem.

What to check first:

    Episode length                should hold up through the curriculum. A monotone fall
                                  from the first iteration is the penalties outweighing the
                                  floor, a collapse timed to a Curriculum/ stage change is
                                  a positive term pulling the robot over
    Metrics/pass/speed_achieved   the number that says whether this is a kick. A shove
                                  plateaus under 1 m/s, a strike reaches the command range
    Episode/rew_sole_strike       should rise from its early trough toward zero. While it
                                  is large and negative the foot is still on the floor when
                                  it meets the ball
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.kick.kick_env_cfg import (
  g1_kick_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.passing import pass_ppo_runner_cfg
from mjlab.tasks.registry import register_mjlab_task

KICK_TASK_ID = "Mjlab-G1-Kick"


def kick_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The pass's PPO config, under its own experiment name.

  Nothing about the algorithm changes, because nothing about the problem's shape does: the
  same robot, one more rung on the same ladder, an observation eight numbers wider. Longer,
  because the standing stage has to be got through before any of the strike terms pay and
  the swing is a harder thing to stumble into than the shove was.
  """
  return replace(
    pass_ppo_runner_cfg(), experiment_name="g1_kick", max_iterations=15_000
  )


register_mjlab_task(
  task_id=KICK_TASK_ID,
  env_cfg=g1_kick_env_cfg(),
  play_env_cfg=g1_kick_env_cfg(play=True),
  rl_cfg=kick_ppo_runner_cfg(),
)
