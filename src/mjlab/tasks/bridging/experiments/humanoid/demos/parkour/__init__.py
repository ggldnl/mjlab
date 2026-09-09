"""Parkour demo: a G1 walks a generated course, switching skills at every obstacle.

One robot, one locomotion skill, two traversal skills, and a bridge between every pair. The
course is drawn from a seed, a table of rules says what each obstacle asks for, the
controller solves the pose that skill needs the robot in, the selector says which of its
states is easiest to reach, and the bridge goes there. Nothing is retrained per course and
nothing is scripted per obstacle.

    config.yml     every number the course is drawn from
    course.py      what is on the course and where. Pure geometry
    pool.py        the skills, each wrapping one frozen policy and its knobs
    bridge.py      the bridge, aimed at a commanded pose
    arena.py       the environment, and a viewer for the course alone
    controller.py  the rules, the alignment, and the phase machine
    run.py         the entry point

Two kinds of obstacle, and the kind picks the skill:

    box       0.65 m, 1.48 m along the approach, 0.70 m across       -> climb
    hurdle    0.20 m tall and 1.00 m long                            -> jump

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
relative to the robot at frame zero, so the pose the robot must arrive in is that
relationship inverted onto the real obstacle: yaw first, then position. Arrive turned five
degrees and the reference climbs a box five degrees off the real one, which is why the
switch waits on alignment as well as on distance. See controller.approach_box.

Four phases per obstacle, two of them bridges:

    cruise -> bridge -> traverse -> bridge -> cruise

The return bridge is the half a two-skill test has no need of. On a course it matters as
much as the outbound one, because a robot that climbs a box and cannot resume walking has
stopped on top of it rather than cleared it.

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
