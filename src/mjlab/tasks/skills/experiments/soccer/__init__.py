"""
Humanoid robot playing soccer.

Three skills implemented on a Booster T1 (`mjlab.asset_zoo.robots.booster_t1`):
- running: running at a given speed in a given direction;
- kicking: kicking the ball;
- recovery: gain stability;

The controller is based purely on the position of the ball relative to the robot:
- the robot is within a given distance from the ball -> controller starts the
    kicking skill;
- the robot kicks the ball -> controller starts the recovery skill;
"""
