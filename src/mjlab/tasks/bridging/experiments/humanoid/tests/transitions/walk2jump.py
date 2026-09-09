"""Walk, press the button, bridge, jump.

Run:

    1. Look at the window this aims into. The entry table has to exist first: the states
       the bridge aims at come out of it, and so do the frames the jump resumes at.

       uv run python -m ...selector.view --skill jump --gap 0.8

    2. Watch the transition.

       uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2jump

       # headless, firing the switch on step 120
       uv run python -m ...transitions.walk2jump --viewer none --auto 120

       # a particular state rather than the easiest one, and a longer window
       uv run python -m ...transitions.walk2jump --mode manual --entry 4 --duration-s 0.9

The jump has no object and no precondition, so the switch fires on the panel button, or on
--auto N headless. A robot can jump anywhere.

Nothing to tell it either, and that is the change. The skill here is Mjlab-G1-Jump: one
clip, jump_forward_level3, 1.54 m, tracked end to end at inference the way the front kick
and the punch combo are. It has no goal, no distance slider and no folder on the panel. The
continuous jump, which covers a range of distances and deploys a distilled student reading
only a goal, is not what this transition uses.

Which puts the entry frame back in charge of the hand-over. The selector hands back two
things about the state it picks, the state itself and the step of the jump's own trajectory
it was recorded at, and both are used: the bridge is aimed at the state, and the jump is
handed a clip wound to that frame. So the policy resumes mid-jump rather than restarting.
Entering at frame zero would replay the stand and the settle the clip opens with, after the
bridge has already delivered the robot into the crouch those lead to, and the reference
would be a second and a half behind the robot from the first step.

The winding happens once, in `anchor_clip` -> `JumpCommand.anchor_to_robot`, at the moment
the bridge is aimed and not at the hand-over. Placing the clip under the robot after it has
arrived would slide the reference onto wherever it actually got to and erase the arrival
error, which is the one thing this test must not do.

That leaves the arrival as the only thing this measures, and that is the point. A FAIL is
the bridge failing to deliver the robot into the state the entry asks for, in the pose and
at the velocities the clip holds at that frame, with nothing else mixed into it.

In auto the selector picks whichever of the six is easiest to reach from the stride the
robot is in when the button is pressed; the slider is for aiming at one on purpose. Every
state on it is one the jump really passed through, so a FAIL is the bridge failing to get
there, never the target being a state the jump could not have used. That separation is what
the whole selector exists for: before it, a bad hand-over and a badly chosen target looked
the same.

The window is frames 55 to 110, and the trail drawn at the switch is the odd one of the set.
The clip stands still until frame 90, so the first four ghosts land on the same tile and the
last two, the ones that have begun to lower, land on it too. That is not a drawing fault: a
jump goes nowhere until it leaves the ground, and what the line is showing is that the six
entries differ in posture and in momentum rather than in place. Compare it with the kick's,
which is a metre of run-up.

What to expect. The jump's best states are mostly a settled stand and a crouch, and the walk
arrives carrying about a metre per second, so the window has to shed nearly all of it. The
`cross` line prints the acceleration that implies and says when it is past what a body does.
Expect the verdict to read lower than it used to on the same hand-over: a discounted score is
a harder bar than a mean, and the ones it now fails are the ones it should have been failing.
"""

from mjlab.tasks.bridging.experiments.humanoid.tests.actors import JUMP, WALK
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import Couple, main

if __name__ == "__main__":
  main(Couple(leaving=WALK, entering=JUMP, duration_s=0.7))
