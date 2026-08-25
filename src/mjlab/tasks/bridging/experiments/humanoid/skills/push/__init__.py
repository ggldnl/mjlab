"""The pushing skill: a G1 drives a one metre crate along at the commanded velocity.

    uv run train Mjlab-Parkour-Push --env.scene.num-envs 4096
    uv run play Mjlab-Parkour-Push

The crate is the asset zoo's parametric box, given a mass so that it has a freejoint and
physics can move it, spawned a stride or two ahead of the robot in the robot's own
heading frame. It is a cube a metre on a side: too tall to step over, too wide to
sidestep without a detour, and light enough that a leaning G1 can shift it.

This starts from locomotion rather than from scratch. The environment is mjlab's flat G1
velocity task with the crate added and a three rung ladder layered on top, so the robot
is still learning to walk and to track a commanded twist while it learns to push, and
every gait term that makes a G1 walk properly is the tuned one rather than a re-guessed
copy. See push_env_cfg.py for the ladder and mdp.py for why each rung is shaped the way
it is.

The goal is the twist itself, not a second command. The robot is told to travel at a
velocity, the crate in front of it has to travel at that velocity too, and the top rung
of the ladder is the fraction of the commanded velocity the crate is actually carrying.
That way the conditioning reaches the observation and the reward through the machinery
that was already there, instead of through a push command whose relationship to the
locomotion one would have to be negotiated every step.

Two things to check before concluding that a run has worked:

  Episode_Metrics/push_contact_rate tells you whether the crate is being touched at all.
  Everything else is meaningless until this is high, and a policy that has learned to walk
  around the crate scores well on velocity tracking with this near zero.

  Episode_Metrics/push_box_displacement tells you whether the crate is going anywhere. It
  is the furthest the crate got in an episode, in metres, and it is the headline number.

This skill is not wired into the parkour arena. That corridor's bridge reads a shared
proprioception group (see ../../../parkour/arena.py), and this observation carries crate
state that group has no channel for. Adding it means giving the corridor a crate, which
is a change to the corridor rather than to this task.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.push.push_env_cfg import (
  g1_push_env_cfg,
)
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

PUSH_TASK_ID = "Mjlab-Parkour-Push"


def push_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The stock G1 velocity PPO config, under its own experiment name.

  Unchanged apart from the name and the length. This is the velocity task with extra
  reward terms and a slightly wider observation, which is exactly what that config is
  tuned for, and there is nothing to be gained from guessing at differences before a run
  has shown one. The length is the run task's rather than the walk task's, because the
  command curriculum has three stages to climb and the crate makes each one harder than
  the bare locomotion version.

  If training plateaus, check Loss/learning_rate before touching the reward. The KL
  adaptive schedule clamps to a floor of 1e-5 and prints as 0.00000 once it gets there,
  which looks exactly like a reward problem and is not one.
  """
  return replace(
    unitree_g1_ppo_runner_cfg(),
    experiment_name="parkour_push",
    save_interval=200,
    max_iterations=15_000,
  )


register_mjlab_task(
  task_id=PUSH_TASK_ID,
  env_cfg=g1_push_env_cfg(),
  play_env_cfg=g1_push_env_cfg(play=True),
  rl_cfg=push_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
