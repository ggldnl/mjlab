"""Architecture 2: arch_1's bridge, with nothing hand-written telling it when to let go.

Same machinery as arch_1 (one bridge actor and one switch-decider per target skill,
same networks, same checkpoint layout) and the same phase 1: the bridge learns to move
like the target skill by playing a copying game against a discriminator, which needs no
notion of success at all.

The difference is phase 2. arch_1 judges a hand-over with a test the experiment author
wrote by hand, one per skill, which is both the most controllable and the most fragile
part of it: the test has to know what "safe to take over from" means, and if it is
wrong the decider optimizes the wrong thing without any sign that it has. arch_2 throws
that away and keeps only what the environment already reports: a hand-over was good if
the episode did not end. Nothing to write, nothing to tune.

That works exactly when the failure being bridged around is a termination, which is the
case the experiments are built on (the diffdrive tips, a robot falls). It says nothing
on a task that never terminates, and there the decider will learn no more than "commit
before the window runs out".
"""

from __future__ import annotations

from mjlab.tasks.skills.architectures.arch_1 import Arch1


class Arch2(Arch1):
  """arch_1's meta policy. Only how its switch-decider was trained differs."""

  _CHECKPOINT = "arch_2.pt"
