"""Walk up to the crate, and start pushing when it comes into reach.

Run:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2push

No button needed. The crate is put a few metres out in the robot's own heading frame, and
the switch fires when walking has brought it into the box the push was trained from.

The window is worth setting by hand here. The push's profiled rollout is already shoving
the crate at over a metre a second by frame 50, and asking a walking robot to be in that
state in 0.6 s asks for an acceleration no body produces:

    2 m/s in 0.6 s   =   3.3 m/s^2

Frame zero is the stance the push actually opens from, and 35 steps is enough time to
arrive standing in it.
"""

from mjlab.tasks.bridging.experiments.humanoid.tests.actors import PUSH, WALK
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import Couple, main

if __name__ == "__main__":
  main(Couple(leaving=WALK, entering=PUSH, duration_s=0.7))
