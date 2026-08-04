"""arch_4's training budget.

Its own rather than arch_1's `BridgeTraining` because arch_4 is trained on a motion
corpus rather than on the pool: there is no discriminator, no expert window collected
from a skill, and no switch-decider, so most of the fields there would be dead. What it
needs instead is where the clips live, how much context to read on either side of a
masked window, how long the mask is, and the weights of the tracking reward.

An experiment sets this as the `arch_4` field of its `Budgets` (see
architectures/__init__.py), which is also what puts every field below on the command
line as `--budgets.arch-4.<field>`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaskedTraining:
  """The whole budget for arch_4: one corpus, one policy, one phase."""

  ##
  # The corpus.
  ##

  motion_dir: str = "data/lafan1/motions"
  """Directory of converted motion npz files to mask windows out of.

  Whatever wrote them is arch_4's business only in that the layout has to be the one
  mjlab's tracking pipeline uses (per-body world poses and velocities plus joint state).
  For the parkour experiment that is LAFAN1, converted by its dataset.py."""

  motion_pattern: str = "*.npz"
  """Which files in that directory to take. Narrow it to select a subset of the corpus
  without moving files around: `walk*.npz` trains a bridge that has only ever seen
  walking."""

  ##
  # The mask.
  ##

  past_steps: int = 16
  """Frames before the gap the bridge reads: the motion it is stitching from.

  At 50 Hz this is about a third of a second, which on a humanoid is half a stride. Long
  enough to say what the robot was doing rather than merely where it was."""

  future_steps: int = 16
  """Frames after the gap the bridge reads: the motion it is stitching to.

  This is the only thing that says where a transition is headed -- there is no skill id
  anywhere in the input. It is also the window the reward is measured on once the gap
  has run out, so it doubles as the arrival test."""

  context_stride: int = 2
  """Take every n-th frame of the two context windows rather than every one.

  A context has to cover enough time to say what the robot is doing, and on a humanoid
  that means a stride or so: sixteen consecutive frames at 50 Hz is a third of a second,
  which is half a step and looks much the same whether the robot is walking or running.
  Striding covers `past_steps * this` steps of motion for the width of `past_steps`
  frames, and motion at 50 Hz is smooth enough that the frames in between carry almost
  nothing the neighbours do not.

  The width is worth caring about for a second reason. Both windows are part of the
  policy's observation, so they are stored for every step of every rollout: at 4096
  environments and a 64-step horizon, each frame of context costs about a hundred
  megabytes of rollout storage. This is the knob that keeps that in hand."""

  gap_range: tuple[int, int] = (32, 64)
  """How long a masked window is, drawn per environment [env steps].

  Sampled rather than fixed because the policy reads how far through the gap it is, so
  one set of weights covers a family of gap lengths and the length becomes something the
  caller picks at inference. Half a second to a second and a quarter at 50 Hz: long
  enough for a humanoid to change what it is doing, short enough that the in-between is
  still determined by its two ends."""

  inference_gap: int = 48
  """How long a transition lasts at inference. Inside `gap_range`, or the bridge is
  asked to fill a gap of a length it never saw."""

  eventfulness: float = 2.0
  """How hard to bias window sampling toward the busy parts of a clip.

  Windows are drawn with probability proportional to (1 + score)^this, where the score
  measures how much changes across the window: acceleration, turning, vertical motion,
  joint activity (see dataset.py). At 0 windows are drawn uniformly, which spends most
  of the budget on the standing and idling that a motion capture corpus is largely made
  of. A stitch between two different behaviors is what this architecture is for, so the
  windows worth masking are the ones where the behavior changes."""

  ##
  # The reward: how closely the robot reproduced the masked motion.
  ##

  weight_posture: float = 1.0
  std_posture: float = 0.3
  """Height and tilt. The term that says the robot is still standing up."""

  weight_lin_vel: float = 1.0
  std_lin_vel: float = 0.5
  """Root linear velocity in its own frame [m/s]. On a transition between a run and a
  jump this is the channel that carries the whole problem."""

  weight_ang_vel: float = 0.5
  std_ang_vel: float = 1.5
  """Root angular velocity [rad/s]."""

  weight_joint_pos: float = 2.0
  std_joint_pos: float = 0.5
  """Joint angles [rad]. The heaviest term, and the same length scale ASAP's tracking
  reward uses; posture is most of what makes a motion recognisable."""

  weight_joint_vel: float = 0.5
  std_joint_vel: float = 5.0
  """Joint velocities [rad/s]."""

  termination_penalty: float = 25.0
  """Subtracted on the step the robot breaks the episode.

  The five terms above are all positive and sum to at most 5 per step, so a robot that
  cannot track has an incentive to end the episode early and stop accruing nothing --
  the suicide trap. This has to outweigh the reward left on the table over a rollout,
  which is why it is several times the per-step maximum rather than comparable to it."""

  ##
  # The rollout.
  ##

  num_iterations: int = 1000 # 3000
  """How many rollouts. A humanoid learning to produce arbitrary in-betweens is a
  tracking problem of the same size as the jump task, which trains for 15k iterations;
  this is smaller because every environment is teleported onto a reference every time."""

  log_every: int = 10

  ##
  # PPO.
  ##

  learning_rate: float = 1.0e-3
  num_learning_epochs: int = 4
  num_mini_batches: int = 4
  clip_param: float = 0.2
  gamma: float = 0.99
  lam: float = 0.95
  value_loss_coef: float = 1.0
  entropy_coef: float = 0.001
  """A tenth of the family default, for the reason the parkour experiment gives for its
  own: on a 29-joint humanoid an entropy bonus that outweighs a weak early surrogate
  drives the standard deviation up until it overflows."""

  max_grad_norm: float = 1.0
  ppo_schedule: str = "adaptive"
  desired_kl: float = 0.015
  actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
  critic_hidden_dims: tuple[int, ...] = (512, 256, 128)

  action_init_std: float = 0.3
  """Exploration at the start, in normalized action units (see spaces.py), where the
  range the pool commands spans roughly [-1, 1]."""

  action_max_std: float = 1.0
  """Ceiling on it. rsl_rl's Gaussian keeps log_std as a free parameter and nothing
  bounds it; left alone it overflows and surfaces as a NaN thousands of iterations in."""

  state_clip: float = 5.0
  """Cap on any one standardized channel reaching the networks (spaces.py)."""

  ##
  # The profiles: what a transition aims at, at inference.
  ##

  profile_rows: int = 64
  """How many recorded windows of each skill to keep as future context.

  At inference the second half of the bridge's question cannot come from a clip, since
  there is none: it comes from here, a bank of frames recorded by letting each skill run.
  Sampled from per environment at the moment of a switch, and kept in the checkpoint."""

  profile_offset: tuple[int, int] = (0, 0)
  """How far into a skill's own rollout the recorded window is taken from [env steps].

  The knob for "where in the next policy's rollout should the bridge be aiming". Zero is
  the literal reading: stitch to how the skill begins. A later offset aims instead at
  the skill in its steady state, which for a periodic gait is a more honest target than
  its first stride and for the jump is the middle of a clip the robot cannot reach."""
