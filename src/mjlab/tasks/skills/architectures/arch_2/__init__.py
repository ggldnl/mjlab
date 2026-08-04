"""Architecture 2: arch_1, with nothing hand-written telling it when to let go.

Same machinery, same networks, same checkpoint layout, same phase 1 (which never had a
notion of success to begin with). The difference is phase 2's verdict.

arch_1 judges a hand-over with a test the experiment author wrote, one per skill: the
most controllable part of it and the most fragile, since the test has to know what "safe
to take over from" means and there is no sign when it is wrong. arch_2 throws that away
and keeps only what the environment already reports: a hand-over was good if the episode
did not end.

That works exactly when the failure being bridged around is a termination, which is what
the experiments are built on. It says nothing on a task that never terminates, and there
the decider will learn no more than "commit before the window runs out".
"""

from __future__ import annotations

from mjlab.tasks.skills.architectures.arch_1 import Arch1


class Arch2(Arch1):
  """arch_1's meta policy. Only how its decider was trained differs."""

  _CHECKPOINT = "arch_2.pt"
