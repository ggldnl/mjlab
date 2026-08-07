"""The in-betweener: the piece of arch_4 that crosses a gap between two motions.

A policy that drives a real robot across a hole cut out of recorded motion. It is handed
the state a body was actually in at the hand-off, momentum included, told where it has to
arrive, and paid for reproducing what the body did in between. It is never shown that
in-between, because at inference there is not one.

This is one component of the architecture, not the architecture. arch_4 as a whole is a
single bridge for the whole pool plus a chooser that decides which moment of the next
skill to aim at; only the bridge exists so far, which is why it lives in a folder of its
own rather than spread across arch_4's top level.

    uv run python -m mjlab.tasks.skills.architectures.arch_4.bridge.dataset
    uv run train Mjlab-Bridge-G1 --env.scene.num-envs 4096
    uv run python -m mjlab.tasks.skills.architectures.arch_4.bridge.evaluate

An earlier attempt at this component was supervised: a network that regressed the missing
frames. It beat interpolation on four windows in five and was still unusable, because a
squared error is minimized by the average of the ways a body could cross a hole, and the
average of a planted foot and a lifted one is a foot through the floor. It put one there
in 41% of windows. Physics is not a detail that can be added afterwards to a motion that
never had any; it has to be in the loop that produces the motion.

    dataset.py    LAFAN1 into windows: context, hole, context.
    frames.py     the hand-off frame, what makes a window position-independent.
    mdp/          the window command, its reward, and how an attempt ends.
    env_cfg.py    the whole thing as an ordinary mjlab task.
    evaluate.py   a trained bridge against the clip it was cut from.
    view.py       the coloured ghost both viewers draw.
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.skills.architectures.arch_4.bridge.env_cfg import bridge_env_cfg

BRIDGE_TASK_ID = "Mjlab-Bridge-G1"


def bridge_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO for the bridge.

  Follows the parkour jump's config, which is the right starting point because both are
  tracking tasks on the same robot, and for the same two reasons it gives.

  `init_std` is 0.6 rather than 1.0: at this action scale a unit standard deviation is a
  lot of noise on every joint every step, and a hole is under a second and a half long,
  which is not enough time to recover from it.

  `num_learning_epochs` is 4 with `desired_kl` 0.015, because the KL-adaptive schedule
  walks the learning rate down to rsl_rl's 1e-5 floor on tasks with wide mixed-unit
  observations, and a rate stuck at the floor is indistinguishable from a plateau.
  """
  return RslRlOnPolicyRunnerCfg(
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
    experiment_name="bridge_in_betweener",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=15_000,
  )


register_mjlab_task(
  task_id=BRIDGE_TASK_ID,
  env_cfg=bridge_env_cfg(),
  play_env_cfg=bridge_env_cfg(play=True),
  rl_cfg=bridge_ppo_runner_cfg(),
)
