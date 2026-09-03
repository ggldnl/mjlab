"""Walk, press the button, bridge, front kick.

Run:

    uv run python -m \
      mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2front_kick

    # headless, firing on step 120
    uv run python -m ...transitions.walk2front_kick --viewer none --auto 120

Nothing on the floor and no precondition, so the switch fires on the panel button or on
--auto, the way the jump's does.

This couple and walk2punch_combo are the cleanest measurement the bridge gets, and that is
why they exist. Every other transition mixes two questions: did the robot arrive in the
right state, and did it arrive in the right place relative to a ball or a crate. A strike
can happen anywhere, so the second question does not apply and the arrival score is about
the pose and the velocities and nothing else.

The clip opens on half a second of held stance, added by the front kick's own converter so
that frame zero is a robot standing still. That makes frame 0 a real hand-over target
rather than a mid-bounce, and it also makes it the boring one: a robot delivered there has
to settle and start the motion from scratch, which is the stop-and-restart the whole
experiment is trying to skip. The knee comes up somewhere after that half second, which at
50 Hz is frame 25 or so, and the frames around it are what `--frame` is for.

Nothing here says which frame is right. That is the choice this harness exists to make
visible, so it is left on the slider at the shared default until it has been watched.
"""

from mjlab.tasks.bridging.experiments.humanoid.tests.actors import FRONT_KICK, WALK
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import Couple, main

if __name__ == "__main__":
  main(Couple(leaving=WALK, entering=FRONT_KICK))
