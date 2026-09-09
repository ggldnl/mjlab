"""Where each skill can be entered.

    record.py  ->  build.py  ->  query.py
     (drive)        (pick)       (choose)

           view.py (look at what was picked)

The bridge runs for a fixed window and then hands over, so it needs somewhere to aim.
This package says where.

One rule. The states a skill can be entered at are the states it passes through during a
stretch of its own timeline, and that stretch is written down here, per skill, by hand.
WINDOWS below is the whole tuning surface. Nothing is discovered and nothing is rejected.

    WINDOWS[skill].phase   the stretch a bridge may aim into
    WINDOWS[skill].states  how many states to take from it, equally spaced

build.py cuts the window into that many equal slices and keeps one state per slice: the
medoid of every rollout that was inside that slice. The medoid is what makes it robust.
A few rollouts drift or fall and land far from the rest, so the middle of the cloud is a
state the skill really was in, without a threshold having to say which.

query.py then hands back whichever of those states is closest to where the robot is now.
Equally spaced along the window means the states differ mostly in momentum, so a robot
arriving fast enters late and one arriving slow enters early.

Equally spaced in frames is not equally spaced on the ground, and view.py draws the second
one. EntryTable.trail integrates the velocities the states carry to say how far apart they
really are, so a window reads as the stretch of rollout it was cut from: the kick's six are
half a metre of run-up, the jump's six are one tile the robot crouches on.

Why a window and not the whole skill. A tracker passes through its landing and its
recovery too, and no bridge would aim there: entering a jump after the landing buys
nothing, and entering a climb once the hands are on the box needs the box already under
them. Worse, a post-landing stand and a pre-jump stand are the same state, so nothing
reading the state alone can keep one and drop the other. Only someone who has looked at
the skill knows which half of the clip is worth entering, so that is what gets written
down.

A skill absent from WINDOWS gets no entry states. It can still be driven, it just cannot
be handed over to.

Run

1. Record the skills, once. Needs a trained checkpoint per skill.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.record

2. Pick the entry states. Reads the rollouts alone, takes seconds, safe to repeat after
   moving a window.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.build

3. Look at them.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.view

Then read the table, or ask which entry is closest to where the robot is:

    table = EntryTable.load()
    for entry in table.of("jump"):
      entry.state    # pose to aim at
      entry.frame    # frame to resume the skill at
      entry.seconds  # where that frame sits in the skill

    reach = best(table, "jump", state, seconds=0.7)
    reach.entry      # aim here
    reach.effort     # below 1 is reachable
    reach.binding    # which channel makes it hard
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
  """The stretch of one skill's timeline a bridge may aim into."""

  phase: tuple[int, int]
  """Half open [first, last) in the skill's own clock, at 50 Hz.

  For a tracker that clock is the frame of the reference it is reading, which is what
  the skill gets resumed at. For a skill trained by reward there is no reference, so it
  is control steps since the episode started and there is nothing to resume: the window
  only decides which poses get picked.

  Recording drops the first 25 steps after every reset, so nothing below about frame 26
  exists whatever is asked for here. build.py clamps to what was recorded and prints
  what it clamped to.
  """

  states: int
  """How many states to take, equally spaced along the window.

  Two is a floor and a ceiling of the window with nothing between. Six over a stretch
  where the robot is accelerating is enough to tell 0.3 m/s from 1.4 m/s apart, which is
  the resolution the closest-state choice runs on.
  """


WINDOWS: dict[str, Window] = {
  # The clip stands still until frame 90, crouches to 110, is airborne to 140 and lands
  # by 155, so 100 is the last frame worth entering at and everything past it is either
  # unreachable or already over. It opens at 55 rather than at the start because
  # entering earlier only buys dead time: the robot stands and waits for the crouch.
  # The late states have begun lowering, so a bridge arriving sinking is put straight
  # into the crouch instead of standing first
  "jump": Window(phase=(55, 110), states=6),
  # The approach walk toward the ball, frames 95 to 145, ending just before contact.
  # Speed runs 0.1 to 1.5 m/s across it, which is the widest spread of any window here
  # and the one where picking the closest state matters most. After contact the skill is
  # recovering and there is nothing left to enter
  "kick": Window(phase=(95, 145), states=6),
  # The last free strides before the box. The clip is 455 frames and the robot is on the
  # box from about 130 on, so this is the only part of it that is about the robot rather
  # than about the robot and an obstacle together. Entering later needs the box to
  # already be under the hands, which no bridge can arrange.
  # It stops at 55 and not later because frames 58 to 70 are the hop that starts the
  # climb: clearance reads +0.07 there, both feet off the floor, and a bridge cannot put
  # a body on a chosen ballistic arc
  "climb": Window(phase=(30, 55), states=1),
  # Both martial clips open in a fighting stance and start moving inside half a second,
  # so the window is what is left of the opening before the first strike commits
  "front_kick": Window(phase=(27, 60), states=4),
  "punch_combo": Window(phase=(27, 55), states=4),
  # Trained by reward, not tracking anything, so the clock is step count and one frame is
  # as good as another: the robot is standing over a ball the whole episode. A short
  # window near the start, purely so there is something to aim at
  "pass": Window(phase=(27, 50), states=3),
  # Same, and the parkour controller hands back to walk after every obstacle, so it needs
  # entry states even though walk would accept a hand-over at any state at all
  "walk": Window(phase=(27, 60), states=1),
}
"""Skill name to the stretch of it a bridge may aim into. The tuning surface.

Frames were read off the recorded rollouts: root height and forward speed against clip
frame say where a skill stops standing and starts committing. Move one, re-run build.py,
look at view.py. Nothing else has to be redone.

A skill left out of here gets no entry states, and demos.parkour refuses to hand over to
it. Adding one means adding a line, not writing a rule.
"""

from mjlab.tasks.bridging.experiments.humanoid.selector.query import (  # noqa: E402
  Cost as Cost,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.query import (  # noqa: E402
  RateCost as RateCost,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.query import (  # noqa: E402
  Reach as Reach,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.query import (  # noqa: E402
  best as best,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.query import (  # noqa: E402
  nearest as nearest,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.table import (  # noqa: E402
  Entry as Entry,
)
from mjlab.tasks.bridging.experiments.humanoid.selector.table import (  # noqa: E402
  EntryTable as EntryTable,
)
