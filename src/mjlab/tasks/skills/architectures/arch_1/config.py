"""arch_1's training budget. arch_2 reads the same one; nothing else does.

Split in two because the two phases are trained separately and share nothing beyond the
env. An experiment declares one of these next to its skill pool, so a cart-pole and a
humanoid can want very different numbers without either being hard-coded in a trainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BridgePhase:
  """Phase 1: teaching a bridge to move like its target skill (AIRL + PPO)."""

  num_iterations: int = 500
  """How many rounds of the copying game. The number to raise for a harder robot."""

  num_windows: int = 512
  """How many opening windows of the target skill to record as the "real" half."""

  num_interrupts: int = 4096
  """How many harvested states to start training episodes from."""

  steps_per_env: int = 64
  """How long each training episode runs before the rollout is cut for an update."""

  log_every: int = 10

  termination_penalty: float = 2.0
  """Subtracted on the step the robot breaks the episode.

  The imitation reward carries both signs on its own, so this is not the only thing that
  can say "that was bad". It is kept because falling over is worse than merely looking
  unlike the target skill, and nothing in an imitation reward knows that."""

  reward_clip: float = 5.0
  """Outlier bound on the imitation reward, in standard deviations of that reward.

  Note the units: the reward is centered and divided by its own running spread first
  (see `RewardScaler`), so 5 means five sigma, not five reward units. This knob used to
  be in the referee's raw units at 10, and once the referee drove its output past that,
  every transition scored exactly the bound and phase 1 had nothing left to learn from."""

  # PPO.
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

  action_init_std: float = 0.4
  """Exploration at the start, in normalized action units (see spaces.py), where the
  pool's commanded range spans roughly [-1, 1]. So 0.4 reads as "a fifth of the range in
  either direction"."""

  action_max_std: float = 1.0
  """Ceiling on it. PPO's entropy bonus raises a Gaussian's std without limit unless
  something stops it, and the surrogate that should balance it is weak here by
  construction. Left unbounded it eventually overflows and surfaces as a NaN thousands
  of iterations in. At 1.0 the actor already spans the pool's entire range."""

  ppo_schedule: str = "fixed"
  """rsl_rl adapts its learning rate to hit a target KL unless this is "fixed". Left
  adaptive it halves down to its 1e-5 floor and the bridge stops moving: KL scales with
  the squared mean shift over the variance, so a policy whose std is small next to its
  action range reports a large KL for an ordinary step."""

  # The referee.
  disc_hidden_dims: tuple[int, ...] = (100, 100)
  disc_learning_rate: float = 3e-4
  disc_epochs: int = 4
  disc_batch_size: int = 512
  disc_gamma: float = 0.99

  state_clip: float = 5.0
  """Cap on any one standardized observation channel reaching any network (spaces.py).
  A transition starts in states no skill visits, so without a cap a single unusual
  channel dominates the first layer of everything reading it."""


@dataclass(frozen=True)
class SwitchPhase:
  """Phase 2: teaching a frozen bridge when to let go (Double-DQN)."""

  num_iterations: int = 2000
  """How many hand-over windows to practise on."""

  num_interrupts: int = 4096
  """How many harvested states to start hand-over windows from."""

  max_transition_steps: int = 64
  """How long the bridge gets before it must commit. Running out counts as a failure, so
  this also sets how patient the decider is allowed to be."""

  eval_steps: int = 96
  """How long the target skill drives before the hand-over is judged.

  Long enough that a bad hand-over has visibly gone wrong by then. Judging too early
  reports a failure still in progress as a success; the `late` column in the log is what
  catches that, and it is the number to watch when setting this."""

  epsilon_start: float = 1.0
  epsilon_end: float = 0.05
  epsilon_decay_iterations: int = 300
  """Exploration in iterations (one window each), not env steps: with a thousand parallel
  envs a single window is tens of thousands of steps, and a step-counted schedule hits
  its floor partway through the first iteration."""

  learning_rate: float = 1e-4
  gamma: float = 0.99
  replay_capacity: int = 200_000
  """Capacity of *each* of the two replay buffers, one per decision."""

  batch_size: int = 512
  """Total rows per update, drawn half from each buffer.

  An env produces at most one "switch" row per window but a "stay" row on every step it
  did not, so a single shared buffer runs about fifty to one against the decision that
  matters. Two buffers sampled evenly is the simplest fix."""

  updates_per_iteration: int = 8
  update_target_every: int = 20
  warmup_iterations: int = 10
  log_every: int = 25


@dataclass(frozen=True)
class BridgeTraining:
  """The whole budget for one experiment, both phases."""

  bridge: BridgePhase = field(default_factory=BridgePhase)
  switch: SwitchPhase = field(default_factory=SwitchPhase)
