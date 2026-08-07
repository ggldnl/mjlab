"""The concrete architectures, and the one place an experiment picks between them.

- arch_0 is the no-bridge baseline (a direct hand-off), used to show that naive
  stitching fails. Nothing to train.
- arch_1 has one transition policy and one referee per skill, plus a hand-over decider
  trained against a hand-written success test.
- arch_2 is arch_1 with that test removed: it judges a hand-over by whether the episode
  survived.
- arch_3 replaces the hand-over decision with a schedule: a fixed-length fade from the
  skill being left into the one being entered, plus a learned residual that expires with
  it. No decider, no success test, one phase.
- arch_4 has a single bridge for the whole pool, trained in physics on holes cut out of
  recorded human motion: the policy is handed the state at the hand-off and paid for
  reproducing what the body did across the gap, given only the frames on either side. It
  learns from a motion corpus rather than from the pool, and so is never retrained when
  the pool changes. Its second half, the chooser that decides which moment of the next
  skill to aim at, is not written yet, so arch_4 builds but does not yet compose.

They are built almost the same way and trained differently, and both halves of that are
deliberate.

Building is near-uniform because a meta policy is a meta policy: `(env, pool, view)` is
all most of them need, which is what lets the demo load any architecture by id without
knowing which one it got. arch_4 is the exception -- it reads the robot's joints and root
motion directly, so it also has to be told which scene entity that is.

Training is not uniform, because these do not share a training procedure. arch_0 trains
nothing, arch_1 runs two phases and needs a success test, arch_3 runs one and needs none,
arch_4 trains its bridge in a different environment entirely. Forcing them through one
signature meant seven parameters, three of them deleted on the first line of most
trainers, and two typed `Any`. So each architecture's `train` takes what it actually
uses, and `train` below is the single place that knows who takes what. Adding an
architecture is one `Budgets` field and one branch here, in a file where a missing one is
visible.

One architecture-specific extra arrives through `train` rather than through `Experiment`,
for the same reason: `success_fns` is arch_1's alone, and a signature that names it says
who needs what. If a second appears, they should move onto `Experiment` as optional
fields instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.architectures.arch_0 import Arch0
from mjlab.tasks.skills.architectures.arch_1 import Arch1
from mjlab.tasks.skills.architectures.arch_1.config import BridgeTraining
from mjlab.tasks.skills.architectures.arch_1.switch import SuccessFn
from mjlab.tasks.skills.architectures.arch_1.train import train as train_arch_1
from mjlab.tasks.skills.architectures.arch_2 import Arch2
from mjlab.tasks.skills.architectures.arch_2.train import train as train_arch_2
from mjlab.tasks.skills.architectures.arch_3 import Arch3
from mjlab.tasks.skills.architectures.arch_3.config import ResidualTraining
from mjlab.tasks.skills.architectures.arch_3.train import train as train_arch_3
from mjlab.tasks.skills.architectures.arch_4 import Arch4
from mjlab.tasks.skills.architectures.arch_4.config import InBetweenerTraining
from mjlab.tasks.skills.architectures.arch_4.train import train as train_arch_4
from mjlab.tasks.skills.experiment import Experiment
from mjlab.tasks.skills.meta import MetaPolicy

ARCHITECTURES: dict[int, type[MetaPolicy]] = {
  0: Arch0,
  1: Arch1,
  2: Arch2,
  3: Arch3,
  4: Arch4,
}


@dataclass(frozen=True)
class Budgets:
  """How long to train, one field per architecture that trains anything.

  An experiment declares one of these beside its skill pool and overrides the
  architectures it actually runs; the rest keep sensible defaults. One field rather than
  one shared object because the architectures do not share a training procedure, and one
  dataclass rather than a dict because these are the CLI: every field is reachable as
  `--budgets.arch-3.tail-steps`.
  """

  arch_1: BridgeTraining = field(default_factory=BridgeTraining)
  arch_2: BridgeTraining = field(default_factory=BridgeTraining)
  arch_3: ResidualTraining = field(default_factory=ResidualTraining)
  arch_4: InBetweenerTraining = field(default_factory=InBetweenerTraining)


def build(env: ManagerBasedRlEnv, architecture: int, exp: Experiment) -> MetaPolicy:
  """The meta policy for `architecture`, untrained. What the demo loads a checkpoint into."""
  if architecture not in ARCHITECTURES:
    raise ValueError(
      f"Unknown architecture {architecture}; registered: {sorted(ARCHITECTURES)}."
    )
  if architecture == 4:
    # arch_4 works on the robot's joint state and root motion, not only on the
    # experiment's state view, so it is the one that has to know the entity.
    return Arch4(env, exp.pool, exp.view, entity_name=exp.entity_name)
  return ARCHITECTURES[architecture](env, exp.pool, exp.view)


def train(
  env: ManagerBasedRlEnv,
  architecture: int,
  exp: Experiment,
  budgets: Budgets,
  success_fns: Mapping[int, SuccessFn] | None = None,
) -> MetaPolicy:
  """Build `architecture` and run its own training. The result is ready to `save`.

  `success_fns` is arch_1's alone: one test per skill saying whether the robot ended up
  somewhere that skill can take over from. Every other architecture ignores it, so an
  experiment that never runs arch_1 need not write one.
  """
  if architecture == 0:
    return Arch0(env, exp.pool, exp.view)

  if architecture == 1:
    if success_fns is None:
      raise ValueError(
        f"arch_1 judges a hand-over with a success test per skill, and "
        f"'{exp.name}' supplied none. Pass success_fns, or run arch_2, which is the "
        f"same architecture judged on survival alone."
      )
    return train_arch_1(
      env, exp, Arch1(env, exp.pool, exp.view), budgets.arch_1, success_fns
    )

  if architecture == 2:
    return train_arch_2(env, exp, Arch2(env, exp.pool, exp.view), budgets.arch_2)

  if architecture == 3:
    return train_arch_3(env, exp, Arch3(env, exp.pool, exp.view), budgets.arch_3)

  if architecture == 4:
    meta = Arch4(env, exp.pool, exp.view, entity_name=exp.entity_name)
    return train_arch_4(env, exp, meta, budgets.arch_4)

  raise ValueError(
    f"Unknown architecture {architecture}; registered: {sorted(ARCHITECTURES)}."
  )
