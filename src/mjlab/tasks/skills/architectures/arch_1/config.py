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

  termination_penalty: float = 2.0
  """Subtracted from the reward on the step the robot breaks the episode.

  The imitation reward is `softplus(...)`, which is always positive, so on its own it
  can only ever say "that was more or less like the target skill" and never "that was a
  disaster". Without this term nothing in phase 1 pays anything for tipping the robot
  over. Sized to be worth a few steps of ordinary imitation reward, which is around 0.7
  per step when the referee is undecided."""

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
  """How much of the useful action range the bridge explores per step, at the start.

  In normalized action units (see spaces.py), where the range the pool's skills command
  spans roughly [-1, 1]. So 0.4 reads directly as "a fifth of the range in either
  direction". Lower it if the robot is being thrown around too hard to learn anything;
  raise it if the bridge is clearly not finding a behavior that exists."""

  action_max_std: float = 1.0
  """Ceiling on the bridge's exploration, in the same normalized units.

  PPO's entropy bonus raises the std of a Gaussian policy without limit unless something
  stops it, and the surrogate term that is supposed to balance it is weak here by
  construction: early on, the bridge cannot move the discriminator much. Left unbounded
  the std grows geometrically and eventually overflows, which surfaces as a NaN inside
  `Normal` thousands of iterations into a run. At 1.0 the actor is already exploring the
  pool's entire commanded range, so there is nothing above this worth reaching."""

  ppo_schedule: str = "fixed"
  """rsl_rl's PPO adapts its learning rate to hit a target KL divergence unless this is
  "fixed". Left adaptive it repeatedly halves the rate down to its 1e-5 floor and the
  bridge stops moving: the KL between two Gaussians scales with the squared mean shift
  divided by the variance, so a policy whose std is small next to its own action range
  reports a large KL for an ordinary step. Normalized actions make the schedule sane
  again, but pinning it keeps one less thing changing underneath a diagnosis."""

  # The discriminator: the referee of the copying game.
  disc_hidden_dims: tuple[int, ...] = (100, 100)
  disc_learning_rate: float = 3e-4
  disc_epochs: int = 4
  disc_batch_size: int = 512
  disc_gamma: float = 0.99

  state_clip: float = 5.0
  """Cap on any one standardized observation channel reaching any bridge network.

  Applied by the run's `StateSpace` (see spaces.py), so the actor, the critic, the
  discriminator and the switch-decider all see the same capped vector. A bridge starts
  in states no skill in the pool visits, so without a cap a single unusual channel
  dominates the first layer of every network reading it. Raise it only if the reward
  looks healthy and the bridge is plateauing because the referee cannot tell two nearby
  behaviors apart."""


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
  """Capacity of *each* of the two replay buffers, one per decision (see below)."""

  batch_size: int = 512
  """Total rows per update, drawn half from each of the two replay buffers.

  The two decisions are recorded at wildly different rates: an env produces at most one
  "switch" row per window (it can only hand over once) but a "stay" row on every step it
  did not, so a single shared buffer ends up around fifty to one against the decision
  that actually matters. Two buffers sampled evenly is the simplest way to keep the
  Q-network from learning the value of switching from a handful of rows drowned in the
  alternative."""

  updates_per_iteration: int = 8
  update_target_every: int = 20
  warmup_iterations: int = 10
  log_every: int = 25


@dataclass(frozen=True)
class BridgeTraining:
  """The whole budget for one experiment, both phases."""

  bridge: BridgePhase = field(default_factory=BridgePhase)
  switch: SwitchPhase = field(default_factory=SwitchPhase)
