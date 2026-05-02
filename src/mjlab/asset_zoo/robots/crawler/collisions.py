"""Crawler collisions."""

from mjlab.utils.spec_config import CollisionCfg


# Friction coefficient of the material feet are made of
CRAWLER_FOOT_MATERIAL_FRICTION_COEFFICIENT = 0.9


# solref (timeconst, dampratio)
# MuJoCo models each constraint (contact, joint limit, etc.) as a damped spring
# that drives the violation to zero. solref defines that spring.
#   timeconst is how fast the contact spring tries to close a violation
#   dampratio is the damping of that spring
_FOOT_SOLREF = (0.01, 1.5)   # (timeconst, dampratio)

# solimp (dmin, dmax, width, midpoint, power)
# Where solref defines the spring dynamics, solimp defines the impedance:
# how strongly the constraint force is applied as a function of penetration depth.
# how much are the bodies overlapping -> how hard is the contact pushing back.
#   dmin is the impedance at zero penetration.
#   dmax is the impedance at full penetration (beyond width).
#   width is the penetration depth over which the impedance ramps from dmin to dmax.
#   midpoint defines where along the width ramp the sigmoid's inflection point sits.
#   power is the exponent of the polynomial that shapes the ramp.
_FOOT_SOLIMP = (0.85, 0.92, 0.006, 0.5, 2)  # (dmin, dmax, width, midpoint, power)


# This config only enables the foot spheres.  Leg capsules stay dormant.
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(".*_foot_geom",),
    contype=0,  # terrain can still touch foot
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(CRAWLER_FOOT_MATERIAL_FRICTION_COEFFICIENT,),
    solref=_FOOT_SOLREF,
    solimp=_FOOT_SOLIMP,
)

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(
        ".*_foot_geom",
        ".*_coxa_collision",
        ".*_femur_collision",
        ".*_tibia_collision",
    ),
    # Enable all listed geoms for active contact (contype=1 & conaffinity=1).
    contype=1,
    conaffinity=1,
    condim={
        ".*_foot_geom":       3,   # friction cone
        ".*_coxa_collision":  1,   # frictionless — just detects contact
        ".*_femur_collision": 1,
        ".*_tibia_collision": 1,
    },
    priority={".*_foot_geom": 1},
    friction={".*_foot_geom": (CRAWLER_FOOT_MATERIAL_FRICTION_COEFFICIENT,)},
    solref={".*_foot_geom": _FOOT_SOLREF},
    solimp={".*_foot_geom": _FOOT_SOLIMP},
)

# Same as FULL_COLLISION but self-collisions are disabled by clearing
# conaffinity on leg capsules so they cannot contact each other.
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
    geom_names_expr=(
        ".*_foot_geom",
        ".*_coxa_collision",
        ".*_femur_collision",
        ".*_tibia_collision",
    ),
    contype=1,
    # Leg capsules can be *hit* by terrain (conaffinity=1) but not by each
    # other (contype check fails for capsule-capsule pairs).
    conaffinity={
        ".*_foot_geom":       1,
        ".*_coxa_collision":  1,
        ".*_femur_collision": 1,
        ".*_tibia_collision": 1,
    },
    condim={
        ".*_foot_geom":       3,
        ".*_coxa_collision":  1,
        ".*_femur_collision": 1,
        ".*_tibia_collision": 1,
    },
    priority={".*_foot_geom": 1},
    friction={".*_foot_geom": (CRAWLER_FOOT_MATERIAL_FRICTION_COEFFICIENT,)},
    solref={".*_foot_geom": _FOOT_SOLREF},
    solimp={".*_foot_geom": _FOOT_SOLIMP},
)