"""Walk up to the ball, and pass it when it comes into reach.

Run:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2pass

No button needed. The ball is put a few metres out on the striking foot's own line, and the
switch fires when walking has brought it into the box the pass was trained from. Steering
is on the Drive sliders if the walk wanders off the line.

What the policy learned is a low shove with the sole rather than a strike, so the task is
called what it does. The skill that swings is the kick, and it has a couple of its own in
walk2kick.

The pass has three entry states over frames 27 to 50, which is a short window near the
opening and deliberately so: the robot is standing over a ball for the whole episode, and
this skill is trained by reward rather than tracking anything, so one frame is much like
another and the window exists only to give the bridge somewhere to aim.

That also makes its trail almost nothing. The three states are a standing robot a fraction
of a second apart, so `selector.view` draws them on top of each other and the transition
draws them on top of the target. Real, and the reason `--gap` exists:

    uv run python -m ...selector.view --skill pass --gap 0.8
"""

from mjlab.tasks.bridging.experiments.humanoid.tests.actors import PASS, WALK
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import Couple, main

if __name__ == "__main__":
  main(Couple(leaving=WALK, entering=PASS))
