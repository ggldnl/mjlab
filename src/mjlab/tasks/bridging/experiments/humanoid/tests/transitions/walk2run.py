"""Walk, press the button, bridge, run.

Run:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.view --skill run
    uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2run

    # headless, firing on step 120
    uv run python -m ...transitions.walk2run --viewer none --auto 120

The run has no object and no precondition, so the switch fires on the panel button or on
--auto.

The couple where the hand-over is a gain rather than a loss. Walking at 1 m/s into a run
held at 3, the bridge has to add 2 m/s inside its window instead of shedding what the
leaving skill built up. At the 3 m/s^2 a body sustains that is 0.67 s of pure acceleration,
so the window is 45 steps rather than the usual 30.

##
# The shortfall this couple exists to show
##

Measured before the shortlist replaced the hand-picked target frame, and worth keeping
because the finding is about the bridge and not about how the target was chosen: asked for
3.00 m/s the bridge arrived at 1.24, and it undershoots commanded velocity by about half
whatever it is asked for at every target and every window length.

    asked 0.56  arrived 0.24    asked 1.70  arrived 0.67
    asked 1.19  arrived 0.61    asked 3.00  arrived 1.24

So it is not a feasibility problem and not a top speed: the smallest of those asks for
1.0 m/s^2 and is missed by the same fraction as the largest.

Every other couple hides this, because their targets barely move and half of a small number
is inside the arrival tolerance. The pool the bridge trained on says why: of 113k recorded
states the median moves at 0.39 m/s and 0.57% exceed 3 m/s, so a target at running speed was
roughly one window in two hundred. Fixing it is a sampling change in the bridge's dataset,
not anything here.
"""

from mjlab.tasks.bridging.experiments.humanoid.tests.actors import RUN, WALK
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import Couple, main

if __name__ == "__main__":
  # The speed the run takes over with is the run's own control now, defaulted to a run and
  # adjustable on its own panel folder. It used to be a field on the couple, because the two
  # skills share one twist term and whichever wrote it last won; the term is owned by
  # whichever skill is driving instead
  main(Couple(leaving=WALK, entering=RUN, duration_s=0.9))
