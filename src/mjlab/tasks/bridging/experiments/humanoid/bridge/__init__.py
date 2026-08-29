"""The bridge: one policy that takes the robot from where a skill left it to where the
next one can start.

A skill is interrupted. The robot is left in whatever dynamic state that interruption
produced, carrying whatever momentum it had, and some other skill has to take over. The
bridge is the piece in between: given the state it inherits and a state it has to reach
within a fixed number of steps, it produces a motion that gets there without falling over.
What it does in the middle is its own business. Nothing scores the middle.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.dataset.dataset
    uv run train Mjlab-G1-Bridge --env.scene.num-envs 4096
    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.evaluate
    uv run play Mjlab-G1-Bridge

    dataset/      where a window's endpoints come from; rollouts of the skills, so far
    mdp/          the window, what it pays, and how it ends
    env_cfg.py    the whole thing as an ordinary mjlab task
    evaluate.py   a trained bridge against a statue, and against its own tolerances
    warm_start.py         the runner, and the optional warm start from a locomotion policy

##
# What is different from the attempts before it
##

Everything before this one was built on LAFAN1: take a clip, cut a hole in it, ask a
policy to fill the hole. Supervised, student-teacher, MaskedMimic, PPO -- none of them
worked, and the diagnosis in hindsight is that the objective was wrong rather than the
optimizer. Motion in-betweening asks a model to recover *the* motion an artist recorded
between two frames, scored against that recording. This task does not care what happens
between two states as long as physics allows it, so a squared error against one particular
crossing penalizes correct answers, and the average of the ways a body can cross a hole is
a foot through the floor.

Two things follow, and they are the whole of the change.

**The middle is not scored.** No reference, no in-between, no reconstruction term. The
reward is the arrival, ramped so its mass sits at the deadline, plus a wide distance kernel
so the first half of a window has something to follow.

**The endpoints come from the robot, not from a corpus.** A retargeted human clip is a
description of a motion, not a state a G1 is ever in, and a pair of them is usually an
impossible thing to ask for. Both ends of every window here are frames of a rollout of a
trained skill, and the pair is then constructed to be feasible rather than merely drawn.
`dataset.py` and `mdp/commands.py` argue this at length; it is the part most likely to be
what was actually wrong.

##
# What this deliberately does not do yet
##

There is no kinematic stage. A reference trajectory between the two endpoints -- a
minimum-jerk interpolation to begin with, an in-betweening transformer or a diffusion
model later -- would turn the sparse terminal reward into a dense tracking signal, and it
is the obvious next thing to try. It is left out here so that the first measurement is of
the task alone: if PPO solves this without a reference, the generative stage was never
needed, and if it does not, the reason will be legible before another moving part is added.

There is also no chaining. The bridge is trained and measured on its own, against states
drawn from skills rather than against skills actually running. Handing it a live
hand-over is a different piece of work and it needs this one to exist first.
"""

from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.bridging.experiments.humanoid.bridge.env_cfg import bridge_env_cfg
from mjlab.tasks.bridging.experiments.humanoid.bridge.warm_start import (
  BridgeOnPolicyRunner,
  BridgeRunnerCfg,
)
from mjlab.tasks.registry import register_mjlab_task

BRIDGE_TASK_ID = "Mjlab-G1-Bridge"


def bridge_ppo_runner_cfg() -> BridgeRunnerCfg:
  """PPO for the bridge.

  Follows the jump's config, which is the right starting point because both are goal
  conditioned tasks on the same robot at the same control rate.

  `init_std` is 0.6 rather than 1.0: at this action scale a unit standard deviation is a
  lot of noise on every joint every step, and the shortest window is three tenths of a
  second, which is not enough time to recover from it.

  `warm_start` is off by default. See warm_start.py for what it copies and why an initialization
  is a weaker lever than a reward term.

  `num_learning_epochs` is 4 against `desired_kl` 0.015 because the KL-adaptive schedule
  walks the learning rate down to rsl_rl's 1e-5 floor on tasks with wide mixed-unit
  observations, and a rate pinned at the floor is indistinguishable from a plateau. If a
  run stalls, read the learning rate before reading anything else.
  """
  return BridgeRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.6,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=4,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.015,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_bridge",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=15_000,
  )


register_mjlab_task(
  task_id=BRIDGE_TASK_ID,
  env_cfg=bridge_env_cfg(),
  play_env_cfg=bridge_env_cfg(play=True, split="eval"),
  rl_cfg=bridge_ppo_runner_cfg(),
  runner_cls=BridgeOnPolicyRunner,
)
