"""Crawler collisions."""

from mjlab.utils.spec_config import CollisionCfg


# Friction coefficient of the material tibias and feet are made of
CRAWLER_FOOT_MATERIAL_FRICTION_COEFFICIENT = 0.45  # (I made up this value :) )

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
# solref (timeconst, dampratio)
# MuJoCo models each constraint (contact, joint limit, etc.) as a damped spring
# that drives the violation to zero. solref defines that spring.
#   timeconst is how fast the contact spring tries to close a violation
#   dampratio is the damping of that spring
#
# solimp (dmin, dmax, width, midpoint, power)
# Where solref defines the spring dynamics, solimp defines the impedance:
# how strongly the constraint force is applied as a function of penetration depth.
# how much are the bodies overlapping -> how hard is the contact pushing back.
#   dmin is the impedance at zero penetration.
#   dmax is the impedance at full penetration (beyond width).
#   width is the penetration depth over which the impedance ramps from dmin to dmax.
#   midpoint defines where along the width ramp the sigmoid's inflection point sits.
#   power is the exponent of the polynomial that shapes the ramp.
#
# Note: in the URDF/MJCF we use a single _geom for both collision
#       geoms and visual geoms
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_geom",),
    condim={".*_foot_geom": 3, ".*_geom": 1},
    priority={".*_foot_geom": 1},
    friction={".*_foot_geom": (CRAWLER_FOOT_MATERIAL_FRICTION_COEFFICIENT,)},
    solref={".*_foot_geom": (0.05, 2.0)},
    solimp={".*_foot_geom": (0.95, 0.99, 0.002, 0.5, 2)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
    geom_names_expr=(".*_geom",),
    contype=0,
    conaffinity=1,
    condim={".*_foot_geom": 3, ".*_geom": 1},
    priority={".*_foot_geom": 1},
    friction={".*_foot_geom": (CRAWLER_FOOT_MATERIAL_FRICTION_COEFFICIENT,)},
    solref={".*_foot_geom": (0.05, 2.0)},
    solimp={".*_foot_geom": (0.95, 0.99, 0.002, 0.5, 2)},
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
    solref={".*_foot_geom": (0.05, 2.0)},
    solimp={".*_foot_geom": (0.95, 0.99, 0.002, 0.5, 2)},
)