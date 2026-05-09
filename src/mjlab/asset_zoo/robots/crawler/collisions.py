"""Crawler collisions."""

from mjlab.utils.spec_config import CollisionCfg


# Leg visuals, no collision

COXA_GEOM_NAMES = (
    "leg_1_coxa_geom",
    "leg_2_coxa_geom",
    "leg_3_coxa_geom",
    "leg_4_coxa_geom",
)

FEMUR_GEOM_NAMES = (
    "leg_1_femur_geom",
    "leg_2_femur_geom",
    "leg_3_femur_geom",
    "leg_4_femur_geom",
)

TIBIA_GEOM_NAMES = (
    "leg_1_tibia_geom",
    "leg_2_tibia_geom",
    "leg_3_tibia_geom",
    "leg_4_tibia_geom",
)

FOOT_GEOM_NAMES = (
    "leg_1_foot_geom",
    "leg_2_foot_geom",
    "leg_3_foot_geom",
    "leg_4_foot_geom",
)

ALL_GEOM_NAMES = (
    *COXA_GEOM_NAMES,
    *FEMUR_GEOM_NAMES,
    *TIBIA_GEOM_NAMES,
    *FOOT_GEOM_NAMES,
)

# Leg collisions

COXA_COLLISION_NAMES = (
    "leg_1_coxa_collision",
    "leg_2_coxa_collision",
    "leg_3_coxa_collision",
    "leg_4_coxa_collision",
)
COXA_COLLISION_REGEX = "*_coxa_collision"

FEMUR_COLLISION_NAMES = (
    "leg_1_femur_collision",
    "leg_2_femur_collision",
    "leg_3_femur_collision",
    "leg_4_femur_collision",
)
FEMUR_COLLISION_REGEX = "*_femur_collision"

TIBIA_COLLISION_NAMES = (
    "leg_1_tibia_collision",
    "leg_2_tibia_collision",
    "leg_3_tibia_collision",
    "leg_4_tibia_collision",
)
TIBIA_COLLISION_REGEX = "*_tibia_collision"

FOOT_COLLISION_NAMES = (
    "leg_1_foot_collision",
    "leg_2_foot_collision",
    "leg_3_foot_collision",
    "leg_4_foot_collision",
)
FOOT_COLLISION_REGEX = "*_foot_collision"

ALL_COLLISION_NAMES = (
    *COXA_COLLISION_NAMES,
    *FEMUR_COLLISION_NAMES,
    *TIBIA_COLLISION_NAMES,
    *FOOT_COLLISION_NAMES,
)

# Sites

FOOT_SITE_NAMES = (
    "leg_1_foot_site",
    "leg_2_foot_site",
    "leg_3_foot_site",
    "leg_4_foot_site",
)


# Bodies

FOOT_BODY_NAMES = (
    "leg_1_foot",
    "leg_2_foot",
    "leg_3_foot",
    "leg_4_foot",
)
FOOT_BODY_REGEX = "leg_[1-4]_foot"

# Base

BASE_NAME = "base"


# Friction coefficient of the material feet are made of
FOOT_MATERIAL_FRICTION_COEFFICIENT = 0.9

# solref (timeconst, dampratio)
# MuJoCo models each constraint (contact, joint limit, etc.) as a damped spring
# that drives the violation to zero. solref defines that spring.
#   timeconst is how fast the contact spring tries to close a violation
#   dampratio is the damping of that spring
FOOT_SOLREF = (0.01, 1.5)   # (timeconst, dampratio)

# solimp (dmin, dmax, width, midpoint, power)
# Where solref defines the spring dynamics, solimp defines the impedance:
# how strongly the constraint force is applied as a function of penetration depth.
# how much are the bodies overlapping -> how hard is the contact pushing back.
#   dmin is the impedance at zero penetration.
#   dmax is the impedance at full penetration (beyond width).
#   width is the penetration depth over which the impedance ramps from dmin to dmax.
#   midpoint defines where along the width ramp the sigmoid's inflection point sits.
#   power is the exponent of the polynomial that shapes the ramp.
FOOT_SOLIMP = (0.85, 0.92, 0.006, 0.5, 2)  # (dmin, dmax, width, midpoint, power)


# This config only enables the foot spheres. Leg capsules stay dormant.
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=FOOT_COLLISION_NAMES,
    contype=0,  # terrain can still touch foot
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(FOOT_MATERIAL_FRICTION_COEFFICIENT,),
    solref=FOOT_SOLREF,
    solimp=FOOT_SOLIMP,
)

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(
        FOOT_COLLISION_REGEX,
        COXA_COLLISION_REGEX,
        FEMUR_COLLISION_REGEX,
        TIBIA_COLLISION_REGEX
    ),
    # Enable all listed geoms for active contact (contype=1 & conaffinity=1).
    contype=1,
    conaffinity=1,
    condim={
        FOOT_COLLISION_REGEX: 3,
        COXA_COLLISION_REGEX: 1,
        FEMUR_COLLISION_REGEX: 1,
        TIBIA_COLLISION_REGEX: 1,
    },
    priority={FOOT_COLLISION_REGEX: 1},
    friction={FOOT_COLLISION_REGEX: (FOOT_MATERIAL_FRICTION_COEFFICIENT,)},
    solref={FOOT_COLLISION_REGEX: FOOT_SOLREF},
    solimp={FOOT_COLLISION_REGEX: FOOT_SOLIMP},
)

# Same as FULL_COLLISION but self-collisions are disabled by clearing
# conaffinity on leg capsules so they cannot contact each other.
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
    geom_names_expr=(
        FOOT_COLLISION_REGEX,
        COXA_COLLISION_REGEX,
        FEMUR_COLLISION_REGEX,
        TIBIA_COLLISION_REGEX
    ),
    contype={
        FOOT_COLLISION_REGEX: 1,
        COXA_COLLISION_REGEX: 0,
        FEMUR_COLLISION_REGEX: 0,
        TIBIA_COLLISION_REGEX: 0
    },
    conaffinity={
        FOOT_COLLISION_REGEX: 1,
        COXA_COLLISION_REGEX: 1,
        FEMUR_COLLISION_REGEX: 1,
        TIBIA_COLLISION_REGEX: 1
    },
    condim={
        FOOT_COLLISION_REGEX: 3,
        COXA_COLLISION_REGEX: 1,
        FEMUR_COLLISION_REGEX: 1,
        TIBIA_COLLISION_REGEX: 1
    },
    priority={FOOT_COLLISION_REGEX: 1},
    friction={FOOT_COLLISION_REGEX: (FOOT_MATERIAL_FRICTION_COEFFICIENT,)},
    solref={FOOT_COLLISION_REGEX: FOOT_SOLREF},
    solimp={FOOT_COLLISION_REGEX: FOOT_SOLIMP},
)