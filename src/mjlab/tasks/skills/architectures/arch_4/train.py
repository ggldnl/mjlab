"""Training for arch_4, which is not a thing that happens here yet.

Every other architecture in this package trains its transition against the pool, inside
the experiment's own arena. arch_4 does not, and that is the design rather than an
omission: its bridge is trained once against a motion corpus, in its own environment,
with no skill in sight, and is then reused by every experiment unchanged.

    uv run train Mjlab-Bridge-G1 --env.scene.num-envs 4096

So what belongs in this file is not the bridge's training. It is the chooser's, and the
chooser does not exist. When it does, this is where it gets fitted: the bridge frozen, the
pool's skills frozen, and the only thing learning being which moment of the next skill to
aim at.

The signature matches the other architectures' so that `architectures/__init__.py` can
keep one branch per architecture and nothing else has to change on the day it works.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.architectures.arch_4 import Arch4
from mjlab.tasks.skills.architectures.arch_4.config import InBetweenerTraining
from mjlab.tasks.skills.experiment import Experiment


def train(
  env: ManagerBasedRlEnv,
  exp: Experiment,
  meta: Arch4,
  cfg: InBetweenerTraining,
) -> Arch4:
  """Fit arch_4's chooser against `exp`'s pool. Not implemented."""
  del env, meta, cfg
  raise NotImplementedError(
    f"arch_4 cannot be trained against '{exp.name}' yet: its bridge is finished but the "
    f"chooser that aims it is not, and a composition needs both. The bridge trains on "
    f"its own, against the motion corpus rather than this pool:\n"
    f"  uv run train Mjlab-Bridge-G1 --env.scene.num-envs 4096\n"
    f"Run arch_0 to arch_3 for a composition that trains today."
  )
