"""Walk, press the button, bridge, jump.

The jump's clip opens on a stand and its crouch is about a second and a half in, so
`--frame` is the interesting slider here: a robot delivered to the crouch skips the
stand-up-and-settle a naive hand-over forces on it.
"""

from mjlab.tasks.bridging.experiments.humanoid.tests.actors import JUMP, WALK
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import Couple, main

if __name__ == "__main__":
  main(Couple(leaving=WALK, entering=JUMP))
