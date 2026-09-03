"""Walk up to the ball, and pass it when it comes into reach.

Run:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2pass

No button needed. The ball is put a few metres out on the striking foot's own line, and the
switch fires when walking has brought it into the box the pass was trained from. Steering
is on the Drive sliders if the walk wanders off the line.

This was walk2kick until the skill was renamed. What the policy learned is a low shove with
the sole rather than a strike, so the task is now called what it does. walk2kick is the
same transition into the version that has to swing.
"""

from mjlab.tasks.bridging.experiments.humanoid.tests.actors import PASS, WALK
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import Couple, main

if __name__ == "__main__":
  main(Couple(leaving=WALK, entering=PASS))
