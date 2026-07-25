"""Concrete bridging architectures.

Maps an architecture id to a factory that, given the env and the skill pool, builds
the `MetaPolicy` that runs the composition. Each architecture *is* a meta policy: it
holds the skill pool and whatever machinery it needs to switch between skills.

- Architecture 0 is the no-bridge baseline (a direct hand-off), used to show naive
stitching drifts.
- Architecture 1 has one transition policy and one discriminator (for the hand-off)
for each skill.
- Architecture 2 has one transition policy and one discriminator for all the skills.
"""

from collections.abc import Callable

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.architectures.arch_0 import Arch0
from mjlab.tasks.skills.architectures.arch_1 import Arch1
from mjlab.tasks.skills.architectures.arch_2 import Arch2
from mjlab.tasks.skills.meta import MetaPolicy
from mjlab.tasks.skills.skill import SkillPool

MetaPolicyFactory = Callable[[ManagerBasedRlEnv, SkillPool], MetaPolicy]

ARCHITECTURES: dict[int, MetaPolicyFactory] = {
  0: Arch0,
  1: Arch1,
  2: Arch2,
}
