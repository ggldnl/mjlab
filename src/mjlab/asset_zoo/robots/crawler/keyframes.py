"""Crawler keyframes."""

from mjlab.entity import EntityCfg


# Default joint positions
_COXA_DEFAULT  =  0.00
_FEMUR_DEFAULT = -0.25
_TIBIA_DEFAULT = -1.75

# Robot standing initially, no need to learn how to get up.
# We need to manually tune the initial height. Using a kinematic
# model to derive this will defy the sole purpose of using DRL:
# it's difficult to have models for complex robots.
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.05),
    joint_pos={
        "base_leg_[1-4]_coxa": _COXA_DEFAULT,
        "leg_[1-4]_coxa_leg_[1-4]_femur": _FEMUR_DEFAULT,
        "leg_[1-4]_femur_leg_[1-4]_tibia": _TIBIA_DEFAULT,
    },
    joint_vel={".*": 0.0},
)