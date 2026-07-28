"""Everything the bridge family's training budget is made of, in one place.

These are the knobs that want changing per experiment rather than per architecture. A
cart-pole is done in a few hundred iterations; a humanoid is not, and burying that
number in the trainer means an experiment cannot say so. An experiment declares one of
these next to its skill pool and its window plan, and every architecture in the family
(arch_1, arch_2, arch_3) reads the same object, so a comparison between them is a
comparison of the signal and nothing else.

Split in two because the two phases are trained separately and have nothing in common
beyond the env: `BridgePhase` is the copying game, `SwitchPhase` is the hand-over
decision. An architecture with no switch-decider simply never reads the second.
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

  # The discriminator: the referee of the copying game.
  disc_hidden_dims: tuple[int, ...] = (100, 100)
  disc_learning_rate: float = 3e-4
  disc_epochs: int = 4
  disc_batch_size: int = 512
  disc_gamma: float = 0.99

  disc_input_clip: float = 10.0
  """Cap on any one standardized input channel reaching the discriminator. Not a
  formality: a skill holds some channels near-constant (the diffdrive's drive keeps
  lateral velocity at zero to within a thousandth), and dividing by that channel's
  expert standard deviation blows an ordinary deviation up by a thousand, which
  saturates the discriminator before the first update and leaves the bridge with a flat
  zero reward. Raise it only if the reward looks healthy and the bridge is plateauing
  because the referee cannot tell two nearby behaviors apart."""


@dataclass(frozen=True)
class SwitchPhase:
  """Phase 2: teaching a frozen bridge when to let go (Double-DQN)."""

  num_iterations: int = 2000
  """How many hand-over windows to practise on."""

  num_interrupts: int = 4096
  """How many harvested states to start hand-over windows from."""

  max_transition_steps: int = 64
  """How long the bridge gets before it must commit. Running out counts as a failure,
  so this also sets how patient the decider is allowed to be."""

  eval_steps: int = 96
  """How long the target skill drives before the hand-over is judged.

  Long enough that a bad hand-over has visibly gone wrong by then. Judging too early
  reports a failure still in progress as a success; the `late` column in the training
  log is what catches that, and it is the number to watch when setting this."""

  epsilon_start: float = 1.0
  epsilon_end: float = 0.05
  epsilon_decay_iterations: int = 300
  """Exploration, measured in iterations (one hand-over window each), not env steps:
  with a thousand parallel envs a single window is tens of thousands of env steps, and a
  step-counted schedule hits its floor partway through the first iteration."""

  learning_rate: float = 1e-4
  gamma: float = 0.99
  replay_capacity: int = 200_000
  batch_size: int = 512
  updates_per_iteration: int = 8
  update_target_every: int = 20
  warmup_iterations: int = 10
  log_every: int = 25


@dataclass(frozen=True)
class BridgeTraining:
  """The whole budget for one experiment, both phases."""

  bridge: BridgePhase = field(default_factory=BridgePhase)
  switch: SwitchPhase = field(default_factory=SwitchPhase)
