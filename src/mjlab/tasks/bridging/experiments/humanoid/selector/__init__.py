"""Where each skill can be entered, measured from its own rollouts.

The bridge has a fixed window and then has to hand over, so it needs somewhere to aim.
This package says where, and how good each spot is.

Idea: cluster the states a skill visits and keep the spots most of its rollouts pass
through. Those are the moments the skill has to go through to do its job, so they are
the moments another skill can join it at. No contacts, no gait, no clip landmarks, so
the same code works for a jump, a kick or a backflip.

Layout
------

    record.py    drives each trained skill and writes down what it does. Runnable
    build.py     finds the entry points in those rollouts. Runnable
    table.py     the table they are written to, and what its columns mean
    view.py      draws one skill's entries side by side. Runnable
    selector.py  the old API the transition tests read, backed by the table

Everything lands under data/selector/. The package owns its input as well as its output,
so nothing here depends on the bridge's dataset being built or current.

Run
---

    1. Record the skills, once. Needs a trained checkpoint per skill.

        uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.record

    2. Build the table. Runs on the rollouts alone, no checkpoints, a few seconds.

        uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.build

    3. Look at it.

        uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.view

    4. Aim at it.

        uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2jump

Query it
--------

    table = EntryTable.load()
    for entry in table.of("jump"):
      entry.state     # the pose to aim at
      entry.frame     # the phase to enter the skill at
      entry.coverage  # how many rollouts pass through here

What this does not do
---------------------

It does not say a spot is survivable. A cluster every rollout passes through is a spot
the skill visits, which is not the same as a spot it can be restarted from. Confirming
that needs the skill's own critic or a rollout sweep, and neither is here yet.

It does not say a spot is reachable. Mid-flight states cluster well and no bridge can
deliver one. What the bridge can reach is the bridge's own arrival score, applied to
this list at run time.

It does not rank for a transition. Rows are sorted by how long a rollout spends in each
spot, which is a property of the skill alone. Which entry is best depends on where the
outgoing skill left the robot, and only the bridge knows that.
"""

from mjlab.tasks.bridging.experiments.humanoid.selector.selector import (
  ScoringCfg as ScoringCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.selector import (
  Selector as Selector,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.selector import (
  Shortlist as Shortlist,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.table import (
  Entry as Entry,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.table import (
  EntryTable as EntryTable,
)
