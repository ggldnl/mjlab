"""Parkour demo: a G1 runs a generated obstacle course, switching skills at every box.

One robot, one locomotion skill, three traversal skills, and a bridge between every pair.
The course is drawn from a seed, a table of rules says which skill each obstacle asks for,
the selector says which state of that skill is easiest to reach, and the bridge goes there.
Nothing is retrained per course and nothing is scripted per obstacle.

    course.py      what is on the course and where. Pure geometry, no mjlab
    pool.py        every policy, loaded once from its own checkpoint
    arena.py       the environment: bridge, every skill's machinery, the boxes
    controller.py  the rule table, and the phase machine that carries it out. Runnable

Four phases per obstacle, two of them bridges:

    cruise -> bridge -> traverse -> bridge -> cruise

The return bridge is the half a two-skill test has no need of. On a course it matters as
much as the outbound one, because a robot that clears a wall and cannot resume running has
stopped on the far side rather than cleared anything.

Needs

A checkpoint per skill in ROSTER, and entry table rows for each. The three traversal skills
do not exist yet: see `controller.STEP`, `VAULT` and `CLIMB` for what adding one costs.

Run

1. Look at a course, and at the plan the rules produce for it. No simulation and no
   checkpoints, so this works before anything is trained.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.demos.parkour.controller \
      --dry True --seed 3 --count 6

2. Run it.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.demos.parkour.controller

3. Headless, for a number rather than a picture.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.demos.parkour.controller \
      --viewer none

4. Move the boxes per environment, so the controller cannot be reading a course it was
   handed at build time.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.demos.parkour.controller \
      --jitter 0.5
"""
