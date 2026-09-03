"""Walk, press the button, bridge, punch combination. The reference couple.

Run:

    1. See what the hand-over is worth before there is a bridge to do it. No bridge
       checkpoint and no corpus needed: it measures handing over cold against handing over
       into a perfect arrival, and the gap between them is what a bridge has to earn.

       uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.handoff

    2. Check the entry state is one a robot can be in.
    2. Look at the entry states.
       uv run python -m ...selector.view --skill punch_combo

    3. Watch the real thing, once a bridge is trained.

       uv run python -m \
         mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2punch_combo

       # headless, firing on step 120
       uv run python -m ...transitions.walk2punch_combo --viewer none --auto 120

One clip, nothing on the floor, no precondition, so the switch fires on the panel button or
on --auto N.

##
# Why this couple first
##

Because the combination does not open from a stand. It is tracked from a LAFAN1 fight clip
and its reset puts the reference at frame zero, which is a guard: knees near seventy degrees,
hips folded unevenly, torso turned into the lead shoulder, elbows drawn in. Sixty-eight
degrees from the robot's default pose at the worst joint.

Hand over to walking or running instead and the entry is roughly the pose the robot is
already in when walking stops, so a bridge that has learned nothing at all still scores well.
Here it either reaches the guard or it does not, and the measurement says which.

Nothing is on the floor either, so a good arrival cannot be spoiled by an object being in the
wrong place. What is left is exactly the question the bridge exists to answer: did the robot
arrive in the pose and at the velocities the entry frame asks for.

##
# What the numbers mean
##

Measured on this couple with no bridge at all, walking interrupted at three seconds:

    handed over cold      fell after 17 steps, discounted return 0.013
    arrived exactly       ran the full window, discounted return 0.152

So the hand-over is real: cold, the combination cannot recover from a walking stride, and
from the entry state the same policy runs the clip cleanly. A trained bridge lands between
those two figures, and where it lands is the only number about it that means anything.

Harder than the front kick for the reason the skill is. A kick is one strike and a
combination is several, so a hand-over that lands out of phase does not cost one missed
moment, it costs every moment after. Which is why the score is discounted: a flat mean over
the whole combination hides a bad opening behind three good strikes, and the opening is the
part the bridge is responsible for.
"""

from mjlab.tasks.bridging.experiments.humanoid.tests.actors import PUNCH_COMBO, WALK
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import Couple, main

if __name__ == "__main__":
  main(Couple(leaving=WALK, entering=PUNCH_COMBO))
