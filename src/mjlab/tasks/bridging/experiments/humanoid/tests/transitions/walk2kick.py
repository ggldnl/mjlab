"""Walk up to the ball, and kick it when it comes into reach.

Run:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2kick

    # fire on the button instead of on the ball, to separate the two failures
    uv run python -m ...transitions.walk2kick --auto 130

The kick's ball sits at 0.42 to 0.50 m from the striking foot instead of the pass's 0.24 to
0.32, so the switch has to fire earlier in the walk, and the trigger reads that box out of
the kick's own config rather than being told it here. The panel drives where the ball is
sent: a speed and an aim, over the range the skill was trained on.

##
# Why this couple shows what the duration is for
##

The kick is delivered to a ball it has to swing at, and the box it has to land in is 8 cm
deep. Every centimetre the bridge overshoots comes off a target that has not moved.

With a fixed window there was one distance at which the hand-over worked: the switch could
choose only *when* to fire, so it had to walk the robot until the arithmetic of a single
window happened to land in the box, and if the robot was already inside that distance the
moment never came. The bridge takes a duration now, so the switch chooses both. It asks, of
every window the bridge was trained on, where the robot would end up, and fires as soon as
one of them puts the ball in the box, at the middle of whichever ones do.

So the hand-over happens early with a long window when the ball is far and late with a short
one when it is close, and the range of moments it can fire in is as wide as the range of
windows the bridge knows. Measured here, the switch picks 1.20 s from a metre out, where a
window fixed at the entry's own 0.70 s would have had to wait another third of a metre.

The two failures are worth keeping apart. `--auto N` fires on a step count instead, so a
transition that fails under the ball rule and works under the button is a trigger problem,
and one that fails under both is the bridge or the entry state.
"""

from mjlab.tasks.bridging.experiments.humanoid.tests.actors import KICK, WALK
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import Couple, main

if __name__ == "__main__":
  main(Couple(leaving=WALK, entering=KICK))
