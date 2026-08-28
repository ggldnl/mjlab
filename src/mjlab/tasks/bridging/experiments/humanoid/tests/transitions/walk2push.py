"""Walk up to the crate, and start pushing when it comes into reach.

No button: the crate is put a few metres out in the robot's own heading frame, and the
switch fires when walking has brought it into the box the push was trained from.

The window is the one that is worth setting by hand here. The push's profiled rollout is
already shoving the crate at over a metre a second by frame 50, and a walking robot asked
to be in that state in 0.6 s is being asked for 3.3 m/s^2, which no body does. Frame zero
is the stance the push actually opens from, and 35 steps is enough time to arrive standing
in it.
"""

from mjlab.tasks.bridging.experiments.humanoid.tests.actors import PUSH, WALK
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import Couple, main

if __name__ == "__main__":
  main(Couple(leaving=WALK, entering=PUSH, steps=35, frame=0))
