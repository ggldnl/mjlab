"""
Humanoid robot running through a corridor with small obstacles. Goal-conditioned
locomotion skills (walking, sprinting, jumping) are extracted from the LAFAN1
dataset.

Once we have the dataset, we train the individual skills with:

    TODO

The controller is based purely on the position of the robot along the corridor:
- the robot is within a given distance from an obstacle
    -> controller starts the jump skill;
- the robot surpasses the obstacle
    -> controller starts the walking skill again;
- the robot has nothing in front of it for a while
    -> controller starts sprinting.

TODO

"""
