"""
Humanoid robot running through a corridor with small obstacles.

Two skills implemented on a Booster T1 (`mjlab.asset_zoo.robots.booster_t1`):
- running: running at a given speed;
- jumping: performing a small jump (this is tricky).

The controller is based purely on the position of the robot along the corridor:
- the robot is within a given distance from an obstacle -> controller starts the
    jump skill;
- the robot surpasses the obstacle -> controller starts the running skill again.

To train the skills:

    uv run train Mjlab-Tracking-Booster-T1
    uv run train Mjlab-Jumping-Booster-T1

"""
