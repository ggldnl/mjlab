"""Crawler collisions."""

from mjlab.utils.spec_config import CollisionCfg


# Friction coefficient of the material tibias and feet are made of
CRAWLER_FOOT_MATERIAL_FRICTION_COEFFICIENT = 0.45  # (I made up this value :) )

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
# Note: in the URDF/MJCF we use a single _geom for both collision
#       geoms and visual geoms
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_geom",),
    condim={".*_foot_geom": 3, ".*_geom": 1},
    priority={".*_foot_geom": 1},
    friction={".*_foot_geom": (CRAWLER_FOOT_MATERIAL_FRICTION_COEFFICIENT,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
    geom_names_expr=(".*_geom",),
    contype=0,
    conaffinity=1,
    condim={".*_foot_geom": 3, ".*_geom": 1},
    priority={".*_foot_geom": 1},
    friction={".*_foot_geom": (CRAWLER_FOOT_MATERIAL_FRICTION_COEFFICIENT,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(".*_foot_geom",),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(CRAWLER_FOOT_MATERIAL_FRICTION_COEFFICIENT,),
)