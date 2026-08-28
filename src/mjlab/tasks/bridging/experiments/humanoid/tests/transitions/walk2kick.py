"""Walk up to the ball, and kick it when it comes into reach.

No button: the ball is put a few metres out on the kicking foot's own line, and the switch
fires when walking has brought it into the box the kick was trained from. Steering is still
on the Drive sliders if the walk wanders off it.
"""

from mjlab.tasks.bridging.experiments.humanoid.tests.actors import KICK, WALK
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import Couple, main

if __name__ == "__main__":
  main(Couple(leaving=WALK, entering=KICK))
