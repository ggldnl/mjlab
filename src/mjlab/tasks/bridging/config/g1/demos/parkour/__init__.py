"""G1 parkour demo with walk, diffusion handoffs, jump and climb.

The controller repeats three readable states for every generated obstacle:

    go_to -> bridge -> traverse

Tall obstacles select climb, short obstacles select jump, then walk drives to the goal.
Course parameters, obstacle yaw and ignored side scenery live in config/config.yml. Tall
box dimensions always come from the climb motion manifest. The Viser Handoff panel changes
bridge distance and duration for the next crossing.

Run:
  uv run python -m mjlab.tasks.bridging.config.g1.demos.parkour.run
  uv run python -m mjlab.tasks.bridging.config.g1.demos.parkour.run --viewer none
  uv run python -m mjlab.tasks.bridging.config.g1.demos.parkour.run --dry True
"""
