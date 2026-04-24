"""Keyframes."""

from mjlab.entity import EntityCfg


# Robot standing initially, no need to learn how to get up.
# We need to manually tune the initial height. Using a kinematic
# model to derive this will defy the sole purpose of using DRL:
# it's difficult to have models for complex robots.
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.04),  # base at 4cm from ground
    joint_pos={
        "base_leg_[1-4]_coxa": 0.0,
        "leg_[1-4]_coxa_leg_[1-4]_femur": -0.25,
        "leg_[1-4]_femur_leg_[1-4]_tibia": -1.75,
    },
    joint_vel={".*": 0.0},
)