"""Crawler actuators: all hardware parameters."""

import math
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.utils.actuator import ElectricActuator
from mjlab.entity import EntityArticulationInfoCfg

# MG90S specification

# Brushed DC coreless motor: ~1.5g rotor, ~3mm radius
MG90S_ROTOR_INERTIA = 7e-9   # kg*m^2
MG90S_GEAR_RATIO    = 5.5    # total spur gear reduction (estimated, ~5/6)

# Simple gear-train reflection: J_out = J_rotor * N²
# (no planetary staging, so no two-stage planetary function needed)
MG90S_ARMATURE = MG90S_ROTOR_INERTIA * MG90S_GEAR_RATIO ** 2  # ~= 2.1e-7 kg*m^2

# Servo stall torque (from datasheet @ 6V, Kgf*cm)
MG90S_STALL_TORQUE_KGF_CM = 2.2
KGF_CM_TO_NM = 9.81 * 0.01
MG90S_EFFORT_LIMIT = MG90S_STALL_TORQUE_KGF_CM * KGF_CM_TO_NM  # ~= 0.216 Nm

# Velocity limit (from datasheet @ 6V, s/60°)
# It's the time in seconds for the servo to sweep 60 deg under no load, RC hobby convention
MG90S_NO_LOAD_SPEED_S_PER_60DEG = 0.08
DEG60_IN_RAD = math.pi / 3
MG90S_VELOCITY_LIMIT = DEG60_IN_RAD / MG90S_NO_LOAD_SPEED_S_PER_60DEG  # ~= 13.1 rad/s

# Actuator limits
ACTUATOR_MG90S = ElectricActuator(
    reflected_inertia=MG90S_ARMATURE,
    velocity_limit=MG90S_VELOCITY_LIMIT,
    effort_limit=MG90S_EFFORT_LIMIT,
)

# Stiffness & damping via natural frequency
# Hobby servos are slower/softer than brushless, ~5 Hz for the controller is realistic.
# This is the parameter most worth sweeping in domain randomization.
MG90S_NATURAL_FREQ = 5.0 * 2.0 * math.pi   # 5 Hz -> ~31.4 rad/s
MG90S_DAMPING_RATIO = 1.0                  # critically damped (typical for hobby servo tuning)

MG90S_STIFFNESS = MG90S_ARMATURE * MG90S_NATURAL_FREQ ** 2  # ~= 2.1e-4 N*m/rad

MG90S_DAMPING = 2.0 * MG90S_DAMPING_RATIO * MG90S_ARMATURE * MG90S_NATURAL_FREQ  # ~= 1.3e-5 N*m*s/rad

# Use the MG90S on the robot

CRAWLER_JOINT_NAMES = [
    "base_leg_1_coxa",  "leg_1_coxa_leg_1_femur",  "leg_1_femur_leg_1_tibia",
    "base_leg_2_coxa",  "leg_2_coxa_leg_2_femur",  "leg_2_femur_leg_2_tibia",
    "base_leg_3_coxa",  "leg_3_coxa_leg_3_femur",  "leg_3_femur_leg_3_tibia",
    "base_leg_4_coxa",  "leg_4_coxa_leg_4_femur",  "leg_4_femur_leg_4_tibia",
]

# Actuator config
CRAWLER_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*",),  # all joints have the same actuator
    stiffness=MG90S_STIFFNESS,
    damping=MG90S_DAMPING,
    effort_limit=ACTUATOR_MG90S.effort_limit,
    armature=ACTUATOR_MG90S.reflected_inertia,
)

CRAWLER_ARTICULATIONS = EntityArticulationInfoCfg(
    actuators=(CRAWLER_ACTUATOR, ),
    soft_joint_pos_limit_factor=0.9,  # clips the joint position limits inside the physics engine
)

# Policy related stuff

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
# a saturated output reaches 50% of range — gradients stay healthy everywhere
# the policy actually operates.
ACTION_FRACTION = 0.8  # The policy can only reach 80% of the URDF range when saturated

# Derived action scales: how many radians a policy output of 1.0 maps to
ACTION_SCALES = {joint: ACTION_FRACTION * rng for joint, rng in JOINT_RANGE.items()}

# Per-joint action scales, looked up by joint type
CRAWLER_ACTION_SCALE: dict[str, float] = {
    name: ACTION_SCALES[
        "coxa"  if name.endswith("coxa")  else
        "femur" if name.endswith("femur") else
        "tibia"
    ]
    for name in CRAWLER_JOINT_NAMES
}

"""
Note: the G1 defines ACTION_SCALES this way:

G1_ACTION_SCALE: dict[str, float] = {}
for a in G1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    G1_ACTION_SCALE[n] = 0.25 * e / s
    
This is saturation deflection - the position error at which the PD controller
hits its torque limit. Beyond e/s radians of error, commanding more position
does nothing - the actuator is already saturated. 0.25 * e/s keeps the policy
comfortably in the linear regime of the actuator, where torque responds
proportionally to position error. On the G1, we care about staying in the
linear torque regime.

On the crawler robot we have MG90S servos that are low-torque and have a
mechanical range as binding constraint, not actuator saturation. This
formulation is simpler and should work better.
"""