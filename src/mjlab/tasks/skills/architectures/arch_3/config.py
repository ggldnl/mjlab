"""arch_3's training budget.

Its own rather than the one in arch_1/config.py because arch_3 has one training phase
and no switch-decider, so half of `BridgeTraining` would be dead here and the two knobs
that matter most (how long a transition lasts, how far past it to keep watching) have no
counterpart there. An experiment declares one of these beside its `BridgeTraining`, and
its train entry point hands the architecture the one it wants.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResidualTraining:
  """The whole budget for arch_3: one phase, one residual per target skill."""

  num_iterations: int = 300
  """How many rounds of the copying game."""

  num_windows: int = 512
  """How many opening windows of the target skill to record as the "real" half."""

  steps: tuple[int, int] = (16, 48)
  """Range the transition length is drawn from, one draw per iteration [env steps].

  Sampled rather than fixed because the residual reads alpha, so one network can cover
  a whole family of fade rates and the length becomes something the caller picks per
  pair rather than something baked into the weights. It also stops the residual from
  learning the schedule as a step counter: the same alpha arrives at different times.
  """

  tail_steps: int = 48
  """How long to keep rolling after the fade ends, with the target skill alone driving.

  The single most important number here, and the reason arch_3 needs no switch-decider
  and no success function. During the tail alpha is zero, so the residual has no
  authority whatsoever and whatever the reward reports is the target skill's own doing.
  Grading a transition on it is therefore grading it on whether it actually delivered
  the robot somewhere the skill can run from. Long enough that a bad hand-over has
  visibly gone wrong by then, the same consideration as arch_1's `eval_steps`.
  """

  inference_steps: int = 32
  """The transition length the trained architecture uses. Inside `steps`, or the
  residual is being asked about a fade rate it never saw."""

  residual_scale: float = 1.0
  """Bound on the correction, in normalized action units (see spaces.py), where the
  range the pool's skills command spans roughly [-1, 1]. So 1.0 reads as "the correction
  may move the action by the pool's whole commanded range", which is generous; it is
  there to stop an untrained network from driving the robot into itself, not to shape
  the solution."""

  residual_penalty: float = 0.05
  """Cost per step on the size of the correction actually applied.

  The schedule already forces the correction to zero by the end. This applies the same
  pressure everywhere else, so of two transitions that work equally well the one that
  leans less on the correction wins, and the fade is left to do as much of the job as it
  can. Sized well below the imitation reward: it is a tie-breaker, not an objective."""

  termination_penalty: float = 2.0
  """Subtracted on the step the robot breaks the episode. An imitation reward has no way
  to know that falling over is worse than merely looking wrong."""

  reward_clip: float = 10.0
  """Bound on the imitation reward handed to PPO, in either direction. A referee that
  has separated the two halves outright drives it to tens, and the critic is then
  fitting returns in the hundreds with a small MLP."""

  state_clip: float = 5.0
  """Cap on any one standardized observation channel reaching any network (spaces.py)."""

  log_every: int = 10

  # PPO, over the residual.
  learning_rate: float = 1e-3
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  clip_param: float = 0.2
  gamma: float = 0.99
  lam: float = 0.95
  value_loss_coef: float = 1.0
  entropy_coef: float = 0.01
  max_grad_norm: float = 1.0
  critic_hidden_dims: tuple[int, ...] = (64, 64)
  ppo_schedule: str = "fixed"

  action_init_std: float = 0.2
  """Exploration on the correction, in normalized action units.

  Lower than arch_1's, and for a reason that is specific to this architecture: the
  action reaching the env is a competent blend plus this, so exploration here perturbs
  something that already works rather than searching for a behavior from nothing.
  """

  action_max_std: float = 1.0
  """Ceiling on that exploration. rsl_rl's Gaussian keeps log_std as a free parameter
  and PPO's entropy bonus raises it without limit; left alone it eventually overflows
  and surfaces as a NaN thousands of iterations into a run."""

  # The discriminator: the referee of the copying game.
  disc_hidden_dims: tuple[int, ...] = (100, 100)
  disc_learning_rate: float = 3e-4
  disc_epochs: int = 4
  disc_batch_size: int = 512
  disc_gamma: float = 0.99
