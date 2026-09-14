"""The distillation bridge. MaskedMimic's second stage, applied to the bridge problem.

Task id Mjlab-G1-Distillation-Bridge, checkpoints under logs/rsl_rl/g1_distillation_bridge.

One run, two phases, and the thing that comes out is a student that reads no reference:

    phase one    PPO trains a teacher that is shown the recorded crossing frame by frame.
                 A tracking problem, which this project trains routinely
    phase two    the teacher is frozen and a student is regressed onto it over the
                 student's own rollouts, reading a randomly masked subset of that crossing

Why two phases rather than one policy

The mapping the bridge wants, (start state, target state, duration) -> motion, is not a
function. Many motions cross between one pair of states, the corpus holds exactly one of
them per window, and a network fitted to that with a squared error is fitted to the mean of
the ones it saw. The mean of two ways to cross is usually not a way to cross.

Neither phase is asked that question. The teacher is given the whole crossing, which leaves
nothing to be multimodal about: there is one motion and the job is to track it. The student
is asked the ambiguous question but is supervised on the states its own policy reaches, and
is told how much of the answer it is allowed to see. It is the mask that makes that a
curriculum rather than a wall.

    in     state (root velocities, gravity, joint angles and rates, last action)
           + per channel gap to the target, clock, reward baseline, tolerance scales
           + 3 interior keyframes, each behind a core bit and an arm bit
    out    29 joint position targets

The mask

The target is never masked. It reaches the policy through the base command, the same way it
does in the imitation bridge, so with every keyframe bit off the student's observation is
imitation's observation followed by a block of zeros. That pattern is drawn outright on
about a third of windows and is the only one play ever shows, because it is the question the
student is deployed on.

The rest of the time a keyframe is readable with probability slot_prob, and its arms on top
of that with arm_prob. So one student covers the whole ladder from "here is where you have
to end up" to "here is most of the motion", and the two ends share a network. That is what
MaskedMimic buys and it is the reason to prefer it here: the bare question on its own gives
a policy nothing to hold on to early in training.

What is not here

No latent. MaskedMimic's student is a conditional VAE, a prior over a latent read from the
masked constraints and a posterior read from the full ones, with a KL term between them, and
that latent is their answer to the residual multimodality. Under the bare mask this student
is a deterministic regression and will pay for it in the usual way: where two crossings are
equally good it will aim between them.

Two reasons it is worth running first anyway. The loss is closed loop, not open loop, so a
student that commits to one crossing sees states consistent with that crossing and is
labelled accordingly, and the averaging is over one step of action rather than over a whole
trajectory. And a latent needs a custom rsl-rl model and a custom loss, which is a large
change to make before there is a measurement saying it is needed. The measurement to make
is Metrics/bridge/score and fixed_arrived under bridge_pattern 1 against the same numbers
under bridge_pattern 0: a student that tracks well with keyframes and arrives badly without
them is the mode averaging, and that is when to write the latent.

Run

1. Build the corpus. Shared, so it lives one level up. See bridges/dataset.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.tracker

2. Inspect it: per source counts, then a window replayed as a ghost.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.view

3. Train. Both phases, one command. Phase one has to plateau before the boundary at
   TRACKING_ITERATIONS or phase two distils a teacher that never learned to track.

    uv run train Mjlab-G1-Distillation-Bridge --env.scene.num-envs 4096

4. Watch what came out, which is the student, on the deployment mask.

    uv run play Mjlab-G1-Distillation-Bridge

Layout

    mdp/commands.py       the window, with its interior exposed and maskable
    mdp/observations.py   the two views of that interior
    env_cfg.py            imitation's environment, read three ways

Reading a run

Phase one is read like a tracker: Episode_Reward/guidance is whether the teacher is on the
crossing at all, and Metrics/bridge/score is whether tracking it lands the arrival. Phase
one ending with a high guidance and a low score means the teacher follows the motion and
misses the state it ends in, which no amount of distillation repairs.

Phase two is read like a regression, plus two metrics this task adds:

    visible_slots    how many interior keyframes this window let through, 0 to keyframes
    bridge_pattern   1 where nothing of the interior is readable. Sits above bridge_prob,
                     since a window that was not drawn bare can still flip every keyframe
                     off: 0.35 + 0.65 * 0.5 ** 3, about 0.43 at the defaults

Everything else is the imitation bridge's and means what it means there. score,
fixed_arrived, reach_* and worst_channel are computed by the same code against the same
baseline, which is what makes a number from this architecture comparable to one from that.

Staged from outside

tests.stage and the parkour demo build their aimed command with tests.stage.aimed_cfg,
which derives the pair from whatever window config the architecture declares. So this one
keeps its mask fields and its own command, and the student's keyframe observation has
something to read. What it reads is the deployment pattern: a window aimed from outside
has no recorded interior, has_reference is zero and every bit is off, which is the only
question the student was trained to answer anyway.

    uv run python -m ...tests.transitions.walk2kick --bridge distillation

A checkpoint saved before TRACKING_ITERATIONS holds a teacher and no student, and loading
one for a transition refuses by name rather than deploying a policy that reads a reference
the arena does not have.
"""

from mjlab.rl import (
  MjlabTeacherStudentRunner,
  RslRlDistillationAlgorithmCfg,
  RslRlModelCfg,
  RslRlTeacherStudentRunnerCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges import BridgeSpec
from mjlab.tasks.bridging.experiments.humanoid.bridges.distillation.env_cfg import (
  distillation_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation import (
  bridge_ppo_runner_cfg,
)
from mjlab.tasks.registry import register_mjlab_task

BRIDGE_TASK_ID = "Mjlab-G1-Distillation-Bridge"
BRIDGE_EXPERIMENT = "g1_distillation_bridge"

MAX_ITERATIONS = 15_000
TRACKING_ITERATIONS = 11_000
"""Where phase one stops and phase two starts.

A starting point, not a measurement. Phase two is supervised against a target the teacher
already computes and converges in a fraction of what a policy gradient needs; what it
cannot do is rescue a teacher that never learned to track. Watch Episode_Reward/guidance
plateau in phase one and move the boundary to where that happened.
"""


def distillation_runner_cfg() -> RslRlTeacherStudentRunnerCfg:
  """Both phases, as one run.

  Phase one is the imitation bridge's PPO config reused rather than restated, so the
  teacher trained here and the bridge trained there cannot drift apart by somebody editing
  one of them. The task they solve is not the same, but the robot, the rate and the action
  scale are, and those are what the config is about.

  Phase two is regression and is configured as one:

      init_std 0.1      the student explores only enough to visit states around the
                        teacher's. The exploration a policy gradient needs is here just
                        noise on the states being labelled
      epochs 1          the data is on policy and thrown away
      mse               the teacher's action is a mean, and squared error is what recovers
                        a conditional mean. huber is what to reach for if a few states
                        turn out to dominate the loss
  """
  tracking = bridge_ppo_runner_cfg()
  return RslRlTeacherStudentRunnerCfg(
    # The deployment mapping. The runner overrides it per phase while it is learning, and
    # this is what is left for anything that loads the checkpoint in order to act
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    teacher_obs_group="teacher",
    tracking_iterations=TRACKING_ITERATIONS,
    # The student. Kept as wide as the teacher even though it reads a narrower observation:
    # it is solving the harder of the two problems, not the cheaper one
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
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
    experiment_name=BRIDGE_EXPERIMENT,
    save_interval=200,
    num_steps_per_env=tracking.num_steps_per_env,
    max_iterations=MAX_ITERATIONS,
  )


register_mjlab_task(
  task_id=BRIDGE_TASK_ID,
  env_cfg=distillation_env_cfg(),
  play_env_cfg=distillation_env_cfg(play=True, split="eval"),
  rl_cfg=distillation_runner_cfg(),
  runner_cls=MjlabTeacherStudentRunner,
)

BRIDGE = BridgeSpec(
  kind="distillation",
  task_id=BRIDGE_TASK_ID,
  experiment=BRIDGE_EXPERIMENT,
  env_cfg=distillation_env_cfg,
)
"""What bridges.resolve hands to a script that asked for this architecture."""
