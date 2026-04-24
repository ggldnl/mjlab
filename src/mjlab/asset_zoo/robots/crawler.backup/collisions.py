"""Crawler collisions."""

from mjlab.utils.spec_config import CollisionCfg


# Friction coefficient of the material tibias and feet are made of
CRAWLER_FOOT_MATERIAL_FRICTION_COEFFICIENT = 0.45  # (I made up this value :) )

# Foot geoms are expected to be the terminal contact surface
_tibia_geom_regex = r"^leg_[1-4]_tibia_geom$"
_foot_geom_regex  = r"^leg_[1-4]_foot_geom$"

# Which geoms do we want non-default contact parameters on?
# The tibia regex should stay in geom_names_expr even when we have feet,
# because tibias can still hit the ground if the robot falls or folds badly,
# and we would want consistent friction on them.
CRAWLER_COLLISIONS = CollisionCfg(
    geom_names_expr=(
        r".*_geom$",
    ),  # match all the geometries to have consistent friction
    condim=3,  # 3d contacts only, normally it would be 6D which is overkill for this robot
    priority=1,
    friction=(CRAWLER_FOOT_MATERIAL_FRICTION_COEFFICIENT,),
    solimp={
        _foot_geom_regex: (0.9, 0.95, 0.005),  # soft landing on feet
        _tibia_geom_regex: (0.9, 0.95, 0.01),  # stiffer, minimal softness on tibia
    },
)