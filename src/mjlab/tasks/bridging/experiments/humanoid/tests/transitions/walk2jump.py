"""Walk, press the button, bridge, jump.

Run:

    1. Build the entry table. The states the bridge aims at come out of it, so this
       has to exist first.

       uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.view \
         --skill jump

    2. Watch the transition.

       uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2jump

       # headless, firing the switch on step 120
       uv run python -m ...transitions.walk2jump --viewer none --auto 120

       # further down the shortlist, and a longer window
       uv run python -m ...transitions.walk2jump --entry 40 --steps 45

The jump has no object and no precondition, so the switch fires on the panel button, or on
--auto N headless. A robot can jump anywhere.

The first couple wired end to end, and the one worth reading the verdict off. `entry 0` is
the state the jump was measured to open best from, and the slider walks down the shortlist
from there. Every state on it is one the jump really started from, so a FAIL is the bridge
failing to get there, never the target being a state the jump could not have used. That
separation is what the whole selector exists for: before it, a bad hand-over and a badly
chosen target looked the same.

Scored, not labelled, and that is what changed. The old set accepted any state the jump
survived three seconds from, which meant most of it was states the jump stumbles out of and
recovers from. Aiming at the middle of those and then judging the hand-over by the same
forgiving rule passed transitions that visibly were not clean ones. Both ends now use the
jump's own discounted return, so a stumble on entry costs the score whatever happens
afterwards, and the top of the shortlist is a state the jump opens well from rather than one
it merely gets away with.

What to expect. The jump's best states are mostly a settled stand and a crouch, and the walk
arrives carrying about a metre per second, so the window has to shed nearly all of it. The
`cross` line prints the acceleration that implies and says when it is past what a body does.
Expect the verdict to read lower than it used to on the same hand-over: a discounted score is
a harder bar than a mean, and the ones it now fails are the ones it should have been failing.
"""

from mjlab.tasks.bridging.experiments.humanoid.tests.actors import JUMP, WALK
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import Couple, main

if __name__ == "__main__":
  main(Couple(leaving=WALK, entering=JUMP, duration_s=0.7))
