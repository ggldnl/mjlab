"""Walk, press the button, bridge, jump.

Run:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2jump

The jump has no precondition, so the switch fires on the panel button, or on --auto N
headless.

The jump's clip opens on a stand and its crouch is about a second and a half in, so
`--frame` is the interesting slider: a robot delivered to the crouch skips the
stand-up-and-settle a naive hand-over forces on it.
"""

from mjlab.tasks.bridging.experiments.humanoid.tests.actors import JUMP, WALK
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import Couple, main

if __name__ == "__main__":
  main(Couple(leaving=WALK, entering=JUMP))
