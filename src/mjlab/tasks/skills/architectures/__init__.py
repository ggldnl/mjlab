"""Concrete bridging architectures.

Maps an architecture id to a factory that, given the env and the skill pool, builds the
`MetaPolicy` that runs the composition, and to the training entry point that fills it
in. Each architecture *is* a meta policy: it holds the skill pool and whatever
machinery it needs to switch between skills.

- Architecture 0 is the no-bridge baseline (a direct hand-off), used to show that naive
  stitching fails.
- Architecture 1 has one transition policy and one discriminator for each skill, and a
  switch-decider trained against a hand-written success test.
- Architectures 2 and 3 are architecture 1 with that hand-written test removed: 2 judges
  a hand-over by whether the episode survived, 3 by the reward the task already defines.
  They exist to show how much of the result depends on the test rather than the bridge.
- Architecture 4 has one transition policy and one discriminator for all the skills.
"""

from collections.abc import Callable
from typing import Protocol

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.architectures.arch_0 import Arch0
from mjlab.tasks.skills.architectures.arch_0.train import train as train_arch_0
from mjlab.tasks.skills.architectures.arch_1 import Arch1
from mjlab.tasks.skills.architectures.arch_1.train import train as train_arch_1
from mjlab.tasks.skills.architectures.arch_2 import Arch2
from mjlab.tasks.skills.architectures.arch_2.train import train as train_arch_2
from mjlab.tasks.skills.architectures.arch_3 import Arch3
from mjlab.tasks.skills.architectures.arch_3.train import train as train_arch_3
from mjlab.tasks.skills.architectures.arch_4 import Arch4
from mjlab.tasks.skills.architectures.arch_4.train import train as train_arch_4
from mjlab.tasks.skills.meta import MetaPolicy
from mjlab.tasks.skills.skill import SkillPool
from mjlab.tasks.skills.view import StateView


class MetaPolicyFactory(Protocol):
  """How every architecture is built: the env, the skill pool, and the state view.

  The view (see view.py) is the slice of the observation the bridging machinery works
  on. It belongs to the experiment rather than to the architecture, so all of them take
  it and the ones with no machinery to point it at ignore it. Omitting it means the whole
  observation, which is the right default only for an experiment that has checked.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    pool: SkillPool,
    view: StateView | None = None,
  ) -> MetaPolicy: ...


ARCHITECTURES: dict[int, MetaPolicyFactory] = {
  0: Arch0,
  1: Arch1,
  2: Arch2,
  3: Arch3,
  4: Arch4,
}

# Every architecture exposes the same train(env, pool, entity_name, meta, success_fns)
# entry point; this maps an architecture id to it. arch_0's is a no-op stub, arch_4's
# raises until it is implemented, and arch_2 and arch_3 accept success_fns and ignore
# it, so an experiment can dispatch by id without special-casing any of them. Typed
# loosely (each trainer takes its own concrete MetaPolicy subclass) so the id-keyed
# dispatch type-checks.
Trainer = Callable[..., MetaPolicy]
TRAINERS: dict[int, Trainer] = {
  0: train_arch_0,
  1: train_arch_1,
  2: train_arch_2,
  3: train_arch_3,
  4: train_arch_4,
}
