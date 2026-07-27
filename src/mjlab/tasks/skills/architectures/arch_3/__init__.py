"""Architecture 3: arch_1's bridge, judged by the reward the task already defines.

Same machinery as arch_1 and the same phase 1. The difference, again, is only what
phase 2 learns from.

arch_2 asks whether the episode survived, which is free but blunt: it cannot tell a
clean hand-over from a barely-recovered one, and on a task that never terminates it
says nothing at all. arch_3 asks the environment's own reward instead. Every skill task
already defines what doing well looks like, so the target skill is let drive for a
while after the hand-over and the reward it collects is the verdict.

The bar is measured rather than chosen. Before training, the target skill is rolled from
its own reset and the reward it earns per step is recorded; a hand-over passes if it
earns about that much. So "good" means "the skill did about as well as it does when it
starts normally", with no threshold to invent per experiment, which was the whole
complaint about arch_1's oracle.
"""

from __future__ import annotations

from mjlab.tasks.skills.architectures.arch_1 import Arch1


class Arch3(Arch1):
  """arch_1's meta policy. Only how its switch-decider was trained differs."""

  _CHECKPOINT = "arch_3.pt"
