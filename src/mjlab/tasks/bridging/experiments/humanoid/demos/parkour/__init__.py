"""Parkour demo: a G1 walks a generated course, switching skills at every obstacle.

One robot, one locomotion skill, two traversal skills, and a bridge into each of them. The
course is drawn from a seed, a table of rules says what each obstacle asks for, and the
course is compiled into a flat list of actions before the robot moves. Nothing is retrained
per course and nothing is scripted per obstacle.

    config.yml     every number the course is drawn from
    course.py      what is on the course and where. Pure geometry
    pool.py        the skills, each wrapping one frozen policy and its knobs
    bridge.py      the bridge, aimed at a commanded pose
    arena.py       the environment, and a viewer for the course alone
    approach.py    where the robot has to stand before a traversal will work
    controller.py  the plan, and the loop that runs it one action at a time
    run.py         the entry point

Two kinds of obstacle, and the kind picks the skill:

    box       0.65 m, 1.48 m along the approach, 0.70 m across       -> climb
    hurdle    0.10 m tall and 0.40 m long                            -> jump

Both are solid, both are turned, and both are coloured from the palette. Going over one and
onto the other is the demo; ending up on the floor is how it stops.

Why the box is not a free parameter
-----------------------------------

Every box is the same, and its size comes out of the climb skill's own manifest rather than
out of config.yml. OmniRetarget retargeted the human motion and its obstacle together and
preserved the contacts between them, so the clip is only physical against that box at that
pose. A course that drew its own size would be asking a policy to climb something it has
never touched.

That is also why the approach is solved rather than tuned. The clip fixes where the box sits
relative to the robot, so the pose the robot must arrive in is that relationship inverted
onto the real obstacle. Arrive turned five degrees and the reference climbs a box five
degrees off the real one, so the pose is solved exactly and the bridge is aimed at it rather
than near it. See approach.solve.

The hurdle is the opposite case and its numbers were measured the same way. Its clip carries
no obstacle, so the bar has to be put where the jump goes: the feet clear the floor over
about half a metre and the lowest one peaks at 0.185 m, which is what sizes it. A hurdle
drawn to taste is one the robot lands on.

Three actions per obstacle, and a run out at the end:

    go_to -> cross -> traverse -> go_to -> cross -> traverse -> ... -> go_to

go_to walks to a point and stops there, cross runs the bridge into the pose the traversal
needs, and the traversal runs until the robot is back on the ground standing. There is no
bridge on the way out: both traversals end upright at about zero velocity, which is inside
the walk's own initiation set, so the walk takes over directly.

Needs

A checkpoint per skill in `controller.ROSTER`, and entry table rows for each. `run` checks
both before it builds anything and says what is missing.

Run

1. Look at the course. No robot, no policies, no simulation. Either viewer.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.demos.parkour.run \
      --scene True --seed 3 --count 6

2. Look at the plan the rules produce for it.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.demos.parkour.run \
      --dry True --seed 3 --count 6

3. Run it. --viewer takes viser, native or none.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.demos.parkour.run
    uv run python -m mjlab.tasks.bridging.experiments.humanoid.demos.parkour.run \
      --viewer native

4. Headless, for a number rather than a picture.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.demos.parkour.run \
      --viewer none

5. A different course, without touching any code.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.demos.parkour.run \
      --config my_course.yml --scene True
"""
