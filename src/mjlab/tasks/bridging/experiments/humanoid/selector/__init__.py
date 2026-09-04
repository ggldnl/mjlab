"""Where each skill can be entered, measured from its own rollouts.

record.py  ->  build.py  ->  filter.py  ->  query.py
 (drive)       (measure)      (decide)      (choose)

      view.py (look the entry points for a given skill)

The bridge has a fixed window and then has to hand over, so it needs somewhere to aim.
This package says where, and how good each spot is.

Idea: cluster the states a skill visits and keep the spots most of its rollouts pass
through. Those are the moments the skill has to go through to do its job, so they are the
moments another skill can join it at.

    record.py    drives each trained skill and writes down what it does
    build.py     clusters those states and produces candidate
    filter.py    accepts or rejects candidates. Every criterion lives here
    query.py     picks which entry to aim at, given where the robot is now
    view.py      draws one skill's entries side by side. Runnable

    state.py     the canonical frame and the channels. Shared by build and query
    ground.py    measures a pose against the floor
    table.py     the table and what its columns mean

Measuring and deciding are kept apart on purpose. build.py writes every cluster it found,
filter.py says which check rejected each one and by how much. Re-filtering is instant
because the clustering does not repeat.

Everything lands under data/selector/.

Two things it does not answer.
1. Whether a spot is survivable: a cluster every rollout passes through is one
  the skill visits, not one it can necessarily be restarted from, and confirming
  that needs the skill's own critic or a rollout sweep.
2. Whether a spot is reachable: filter.py drops the geometrically impossible ones
  and query.py ranks the rest by the rate of change they demand, but a grounded
  state at 3 m/s passes both and is still out of reach from a standstill. Only
  the bridge's own arrival score answers that.

Run

1. Record the skills, once. Needs a trained checkpoint per skill.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.record

2. Find the candidates. Runs on the rollouts alone, no checkpoints, a few seconds.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.build

3. Accept or reject them. Prints every candidate with the check that failed it.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.filter

4. Look at it.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.view

Then read the table, or ask which entry is easiest to reach from where the robot is:

    table = EntryTable.load()
    for entry in table.of("jump"):
      entry.state     # the pose to aim at
      entry.frame     # the phase to enter the skill at
      entry.coverage  # how many rollouts pass through here

    reaches = nearest(table, "jump", state, seconds=0.7)
    reaches[0].entry     # aim here
    reaches[0].effort    # below 1 is reachable
    reaches[0].binding   # which channel makes it hard
"""

from mjlab.tasks.bridging.experiments.humanoid.selector.query import (
  Cost as Cost,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.query import (
  RateCost as RateCost,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.query import (
  Reach as Reach,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.query import (
  best as best,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.query import (
  nearest as nearest,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.table import (
  Entry as Entry,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.table import (
  EntryTable as EntryTable,
)
