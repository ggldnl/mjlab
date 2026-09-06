"""A goal conditioned jump for the Unitree G1, learned in two phases.

The clips shape the jump. The goal is all that is left when it runs.

Phase one is ASAP (RSS 2025, https://agile.human2humanoid.com/): a G1 learns to jump by
tracking a retargeted human jump frame by frame, with reference state initialization, a dense
per frame tracking reward, and termination the moment tracking is lost. Five forward jumps of
increasing length share the policy, each episode stretches its clip horizontally, and the
target displacement is in the observation and in the reward, so the reachable distances come
out continuous rather than five discrete points.

Phase two takes the reference back out. A tracker reads mostly the clip it is chasing. It
is distilled, in the same run, into a policy that reads the goal and its own body and nothing
else. The student is not reconstructing hidden state from partial observations; it is
internalizing a deterministic map whose input it already has (teacher: clip -> goal).

The single clip sibling is skills/jump, which tracks one of these clips end to end and keeps
the reference at inference. Reach for that one when the jump has to be the same jump every
time.

`MjlabTeacherStudentRunner` runs the phases back to back and hands phase one's actor to
phase two as a frozen teacher in memory, so nothing is written out and read back in between.
What lands in the checkpoint at the end is a student.

Run

1. Fetch and convert the clips. Writes to data/asap/motions.

    uv run --with joblib python -m \
      mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.dataset

2. Train, both phases.

    uv run train Mjlab-G1-Jump-Continuous --env.scene.num-envs 4096

   The split is agent.tracking_iterations of agent.max_iterations. Move it with
   --agent.tracking-iterations N.

3. Watch what came out, which is the student.

    uv run play Mjlab-G1-Jump-Continuous

4. Record where it can be entered from, for compositions.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.record \
      --skills "('jump_continuous',)"
"""

from __future__ import annotations

from mjlab.rl import (
  MjlabTeacherStudentRunner,
  RslRlDistillationAlgorithmCfg,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
  RslRlTeacherStudentRunnerCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.jump_continuous_env_cfg import (
  g1_jump_continuous_env_cfg,
)
from mjlab.tasks.registry import register_mjlab_task

JUMP_CONTINUOUS_TASK_ID = "Mjlab-G1-Jump-Continuous"

TRUNK = (512, 256, 128)
"""Hidden dims, shared by all three networks.

The student keeps the teacher's width even though it reads a much smaller observation. A
narrower student is the thing to try if this were about deployment cost; it is not, it is
about removing the reference, and changing two things at once makes a regression impossible
to attribute."""


def jump_ppo_runner_cfg(
  experiment_name: str = "g1_jump_continuous",
) -> RslRlOnPolicyRunnerCfg:
  """One phase of PPO against a tracking reward. The jump's first phase, on its own.

  Close to mjlab's tracking config, since that is what this is. Two deliberate differences:

    init_std 0.6            1.0 is a lot of per-joint noise every step at this action scale,
                            and the airborne part of the motion has no time to recover from
                            it
    num_learning_epochs 4   fewer chances per iteration for the KL schedule to ratchet the
    desired_kl 0.015        learning rate down to its 1e-5 floor, where a run looks plateaued
                            but is only crawling

  Registered on its own by the front kick and the punch combo, which are single clip trackers
  that read their reference at inference and have no goal to be distilled onto. They stay one
  phase, and this is the phase they are.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=TRUNK,
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.6,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=TRUNK,
      activation="elu",
      obs_normalization=True,
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
    experiment_name=experiment_name,
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=15_000,
  )


def jump_runner_cfg(
  experiment_name: str = "g1_jump_continuous",
) -> RslRlTeacherStudentRunnerCfg:
  """Both phases of the jump, as one run.

  Phase one is `jump_ppo_runner_cfg` reused rather than restated, so the tracker this trains
  and the tracker the front kick trains cannot drift apart by somebody editing one of them.

  Phase two is regression and is configured as one:

    init_std 0.1      the student explores only enough to visit states around the teacher's.
                      The exploration a policy gradient needs is here just noise on the
                      states being labelled
    epochs 1          the data is on policy and thrown away, so a second pass over it fits
                      the same states twice rather than seeing new ones
    mse               the teacher's action is a mean, not a sample, and squared error is what
                      recovers a conditional mean. huber is the thing to reach for if a few
                      states turn out to dominate the loss

  The 12000 of 15000 split is a starting point, not a measurement. Phase two is supervised
  against a target the teacher already computes, so it converges in far fewer iterations than
  the policy gradient before it; what it cannot do is rescue a teacher that never learned to
  jump. Watch the tracking rewards plateau in phase one and move the boundary to where that
  happened.
  """
  tracking = jump_ppo_runner_cfg(experiment_name)
  return RslRlTeacherStudentRunnerCfg(
    # The deployment mapping. The runner overrides it per phase while it is learning, and
    # this is what is left for anything that loads the checkpoint in order to act
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    teacher_obs_group="teacher",
    tracking_iterations=12_000,
    # The student, and the policy that gets deployed
    actor=RslRlModelCfg(
      hidden_dims=TRUNK,
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.1,
        "std_type": "scalar",
      },
    ),
    teacher=tracking.actor,
    critic=tracking.critic,
    algorithm=tracking.algorithm,
    distillation=RslRlDistillationAlgorithmCfg(
      num_learning_epochs=1,
      gradient_length=15,
      learning_rate=1.0e-3,
      max_grad_norm=1.0,
      loss_type="mse",
    ),
    experiment_name=tracking.experiment_name,
    save_interval=tracking.save_interval,
    num_steps_per_env=tracking.num_steps_per_env,
    max_iterations=tracking.max_iterations,
  )


register_mjlab_task(
  task_id=JUMP_CONTINUOUS_TASK_ID,
  env_cfg=g1_jump_continuous_env_cfg(),
  play_env_cfg=g1_jump_continuous_env_cfg(play=True),
  rl_cfg=jump_runner_cfg(),
  runner_cls=MjlabTeacherStudentRunner,
)
