"""Walk at the ball, press the button, bridge, football kick.

Run:

    1. Look at the kick's window. Six states across fifty frames of run-up, drawn in the
       line the skill walks them in, which is the same line this transition draws in the
       world.

       uv run python -m mjlab.tasks.bridging.experiments.humanoid.selector.view --skill kick

    2. Watch the transition.

       uv run python -m mjlab.tasks.bridging.experiments.humanoid.tests.transitions.walk2kick

       # headless, firing the switch on step 130
       uv run python -m ...transitions.walk2kick --viewer none --auto 130

    3. Compare versions of the kick. Same arena, plus a dropdown that swaps which kick
       catches the robot and a slider that pins the step the switch fires on, so two
       watches differ in the policy and nothing else. A variant is a suffix on the
       experiment name: "base" is g1_kick, "robust" is what skills.finetune wrote to
       g1_kick_robust. Every flag below is taken there too, so --entry pins which of the
       six entries both versions are judged on.

       uv run python -m ...transitions.walk2kick_robust_test --variants "('base','robust')"

       # the same pair on one entry, headless
       uv run python -m ...transitions.walk2kick_robust_test --variants "('base','robust')" \
           --entry 3 --viewer none --auto 130

The ball goes out two and a half to four metres ahead on the striking foot's line, and the
walk sliders steer at it. The switch is the button, as it is for the jump and the strikes:
nothing here waits for the ball to reach a box.

##
# Why the ball places the target, and nothing else can
##

The kick is Mjlab-G1-Kick: one PAiD clip, tracked end to end, of a human running in a metre
and striking a ball. The ball is not scenery in it. dataset.py measured where the sole is
moving fastest and put the ball there, so inside the clip the ball and the swing are one
rigid arrangement, and the swing only connects if the clip is laid down with its ball on the
real one.

Which settles where the robot has to arrive. Anchoring winds the clip to the entry frame and
slides it until the root at that frame sits on the target, so the target is the one free
parameter and the ball fixes it:

    target  =  ball  -  R(heading) * (ball_in_clip - clip_root(entry frame))

`arrive_at_kick` is that line, and declaring it is all it takes: a skill that says where it
needs the robot is aimed there. The alternative, which every skill with nothing on the floor
still gets, puts the target where the robot's own momentum would carry it. That is a
perfectly good place to stand and has nothing to do with where the ball is: the swing goes
through empty floor and the arrival score says the hand-over was fine.

##
# Why the trail is the thing to watch
##

The kick's window is frames 95 to 145, which is not a stance but the last second of the
approach, speed running from 0.1 to 1.5 m/s across it. So an entry is a point on a run-up,
and where the robot must stand depends on which point: entering at 95 means standing a
metre back from the ball, entering at 145 means standing a stride from it. That is why
`Actor.arrive` takes a frame at all, and the kick is the only skill that reads it.

The whole window is drawn from the first step, not at the switch, and that is what makes
the button pressable. `arrive_at_kick` is evaluated at every entry's frame, so the six
ghosts stand where the robot has to be for each of them, and the ball does not move, so
neither do they: measured over a walk-up they sit at 1.62 to 2.21 m and stay within a
centimetre of it while the robot closes from 1.06 m away. The viewer's line counts the gap
down. `show the entry states` on the panel turns the ghosts off.

Which is the difference between hitting the ball and missing it, and the numbers say so:

    fired at step 130     0.08 m from where the kick wants the robot
    fired at step 180     0.68 m past it, travelled 0.24x, score 0.000

Both are the same bridge and the same entry. The second one is a robot that walked through
its own ball while the gap readout bottomed out at 0.33 m and started growing again.

What to look for once the timing is right is which of those ghosts the bridge is aimed at,
because that is the harness saying how much of the approach the hand-over is skipping and
how much speed it is therefore asking the bridge to have built by the time it arrives.

Aimed too late and the demand is a metre and a half per second out of a walk over half a
second, which `cross` prints as an acceleration and calls past what a body sustains. Aimed
too early and the kick has to do the accelerating itself, from a state the bridge could
reach comfortably. Neither is chosen here: the selector picks whichever entry is easiest to
reach from the stride the robot happens to be in when the button is pressed, and the line
shows what it picked out of.
"""

from mjlab.tasks.bridging.experiments.humanoid.tests.actors import KICK, WALK
from mjlab.tasks.bridging.experiments.humanoid.tests.stage import Couple, main

if __name__ == "__main__":
  main(Couple(leaving=WALK, entering=KICK))
