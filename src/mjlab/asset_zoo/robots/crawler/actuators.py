"""Crawler actuators: all hardware parameters."""

import math
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg


# Servo stall torque
STALL_TORQUE = 0.18  # Nm

# Physical joint range per side (half-range). A policy output of +1.0
# moves the joint +JOINT_RANGE from center, and -1.0 moves it -JOINT_RANGE,
# so the total mechanical excursion is 2*JOINT_RANGE.
JOINT_RANGE_DEG = {
    "coxa": 45,
    "femur": 90,
    "tibia": 90,
}
JOINT_RANGE = {j: math.radians(r) for j, r in JOINT_RANGE_DEG.items()}

# Policy controls this fraction of the mechanical range. Most policies
# output actions through a tanh or clipped linear activation, meaning
# the raw network output lives in (-1, 1). The action scale multiplies
# that output to get a joint target. If you map 1.0 to the full
# mechanical range, the policy can only reach the joint limits
# by saturating its output neuron. Saturated neurons have near-zero gradient,
# so the policy stops learning to use extreme positions. We trade off reachable
# range for gradient quality near the edges. ACTION_FRACTION = 0.5 means
# a saturated output reaches 50% of range — gradients stay healthy everywhere the policy actually operates.
ACTION_FRACTION = 0.50

LOAD_FACTOR     = 0.80  # at max action, use this fraction of stall torque (stay linear)
DAMPING_RATIO   = 0.70  # 1.0 = critically damped, <1 = slightly springy

# Numerical stability only, does not affect physical behavior
ARMATURE = 1e-4

# Derived action scales: how many radians a policy output of 1.0 maps to
ACTION_SCALES = {joint: ACTION_FRACTION * rng for joint, rng in JOINT_RANGE.items()}

# Stiffness: designed so that the largest action uses LOAD_FACTOR*stall_torque.
# Smaller joints automatically use less of the budget.
EFFORT_LIMIT = STALL_TORQUE
STIFFNESS = EFFORT_LIMIT * LOAD_FACTOR / max(ACTION_SCALES.values())

# Damping from the standard critically-damped PD formula
DAMPING = 2.0 * DAMPING_RATIO * math.sqrt(STIFFNESS * ARMATURE)
# 2 * 0.7 * sqrt(0.36 * 1e-4) ~= 0.0084 Nm·s/rad

CRAWLER_JOINT_NAMES = [
    "base_leg_1_coxa",  "leg_1_coxa_leg_1_femur",  "leg_1_femur_leg_1_tibia",
    "base_leg_2_coxa",  "leg_2_coxa_leg_2_femur",  "leg_2_femur_leg_2_tibia",
    "base_leg_3_coxa",  "leg_3_coxa_leg_3_femur",  "leg_3_femur_leg_3_tibia",
    "base_leg_4_coxa",  "leg_4_coxa_leg_4_femur",  "leg_4_femur_leg_4_tibia",
]

# Actual Izz values extracted directly from the MJCF fullinertia fields
_INERTIA_COXA  = 8.049e-7  # kg*m^2, from leg_N_coxa_geom inertial
_INERTIA_FEMUR = 4.509e-6  # kg*m^2, from leg_N_femur_geom inertial (dominant axis)
_INERTIA_TIBIA = 1.630e-6  # kg*m^2, from leg_N_tibia_geom inertial

DAMPING_PER_TYPE: dict[str, float] = {
    joint_type: 2.0 * DAMPING_RATIO * math.sqrt(STIFFNESS * inertia)
    for joint_type, inertia in {
        "coxa":  _INERTIA_COXA,
        "femur": _INERTIA_FEMUR,
        "tibia": _INERTIA_TIBIA,
    }.items()
}
# coxa:  2 * 0.7 * sqrt(0.146 * 8.05e-7) ~ 4.8e-4 Nm·s/rad
# femur: 2 * 0.7 * sqrt(0.146 * 4.51e-6) ~ 1.1e-3 Nm·s/rad
# tibia: 2 * 0.7 * sqrt(0.146 * 1.63e-6) ~ 6.7e-4 Nm·s/rad

# Three separate actuator configs, one per joint type
CRAWLER_ACTUATORS_COXA = BuiltinPositionActuatorCfg(
    target_names_expr=tuple(n for n in CRAWLER_JOINT_NAMES if n.endswith("coxa")),
    stiffness=STIFFNESS,
    damping=DAMPING_PER_TYPE["coxa"],
    effort_limit=EFFORT_LIMIT,
    armature=ARMATURE,
)

CRAWLER_ACTUATORS_FEMUR = BuiltinPositionActuatorCfg(
    target_names_expr=tuple(n for n in CRAWLER_JOINT_NAMES if n.endswith("femur")),
    stiffness=STIFFNESS,
    damping=DAMPING_PER_TYPE["femur"],
    effort_limit=EFFORT_LIMIT,
    armature=ARMATURE,
)

CRAWLER_ACTUATORS_TIBIA = BuiltinPositionActuatorCfg(
    target_names_expr=tuple(n for n in CRAWLER_JOINT_NAMES if n.endswith("tibia")),
    stiffness=STIFFNESS,
    damping=DAMPING_PER_TYPE["tibia"],
    effort_limit=EFFORT_LIMIT,
    armature=ARMATURE,
)

CRAWLER_ARTICULATIONS = EntityArticulationInfoCfg(
    actuators=(CRAWLER_ACTUATORS_COXA, CRAWLER_ACTUATORS_FEMUR, CRAWLER_ACTUATORS_TIBIA),
    soft_joint_pos_limit_factor=0.9,
)

# Per-joint action scales, looked up by joint type
CRAWLER_ACTION_SCALES: dict[str, float] = {
    name: ACTION_SCALES[
        "coxa"  if name.endswith("coxa")  else
        "femur" if name.endswith("femur") else
        "tibia"
    ]
    for name in CRAWLER_JOINT_NAMES
}