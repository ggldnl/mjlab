"""Crawler actuators: all hardware parameters."""

import math
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.entity import EntityCfg


LEG_1_JOINT_NAMES = (
    "base_leg_1_coxa",
    "leg_1_coxa_leg_1_femur",
    "leg_1_femur_leg_1_tibia"
)

LEG_2_JOINT_NAMES = (
    "base_leg_2_coxa",
    "leg_2_coxa_leg_2_femur",
    "leg_2_femur_leg_2_tibia"
)

LEG_3_JOINT_NAMES = (
    "base_leg_3_coxa",
    "leg_3_coxa_leg_3_femur",
    "leg_3_femur_leg_3_tibia"
)

LEG_4_JOINT_NAMES = (
    "base_leg_4_coxa",
    "leg_4_coxa_leg_4_femur",
    "leg_4_femur_leg_4_tibia"
)

COXA_JOINT_REGEX = "base_leg_[1-4]_coxa"
FEMUR_JOINT_REGEX = "leg_[1-4]_coxa_leg_[1-4]_femur"
TIBIA_JOINT_REGEX = "leg_[1-4]_femur_leg_[1-4]_tibia"

JOINT_NAMES = [
    *LEG_1_JOINT_NAMES,
    *LEG_2_JOINT_NAMES,
    *LEG_3_JOINT_NAMES,
    *LEG_4_JOINT_NAMES
]


# The MG90S is a brushed-DC servo with an internal PID position loop.
# We model it in MuJoCo as a position actuator (stiffness + damping + effort_limit).
# The three parameters below are the only externally observable quantities:
#   effort_limit  - stall torque
#   velocity_limit - no-load angular speed (used only for documentation here)
#   armature      - reflected rotor inertia

# Effort limit: 2.2 Kgf*cm at 6V (datasheet)
MG90S_STALL_TORQUE_KGF_CM = 2.2
KGF_CM_TO_NM = 9.81 * 0.01
MG90S_EFFORT_LIMIT = MG90S_STALL_TORQUE_KGF_CM * KGF_CM_TO_NM  # 0.216 N*m

# Velocity limit: 0.08 s/60° at 6V (datasheet) -> 13.1 rad/s
MG90S_NO_LOAD_SPEED_S_PER_60DEG = 0.08
MG90S_VELOCITY_LIMIT = (math.pi / 3.0) / MG90S_NO_LOAD_SPEED_S_PER_60DEG  # 13.1 rad/s

# Reflected rotor inertia
# Coreless brushed rotor: ~1.5 g, radius ~3 mm -> J_rotor ~= 7e-9 kg*m^2
# Spur gear train, N ~= 5.5 -> J_arm = J_rotor * N^2 ~= 2.1e-7 kg*m^2
MG90S_ROTOR_INERTIA = 7e-9  # kg*m^2
MG90S_GEAR_RATIO = 5.5
MG90S_ARMATURE = MG90S_ROTOR_INERTIA * MG90S_GEAR_RATIO ** 2  # 2.1e-7 kg*m^2

# Stiffness (Kp)
# Kp = effort_limit / delta_sat, where delta_sat is the position error at which
# the PD controller reaches stall torque.
#
# We choose delta_sat so that the tightest action range (tibia: +/-0.370 rad,
# see CRAWLER_ACTION_SCALE below) stays well inside the linear regime of the
# actuator.  With delta_sat = 0.6 rad, a max tibia action (0.370 rad) produces
# 0.370 / 0.6 ~= 62 % of stall torque — the servo responses proportionally and
# learning gradients flow everywhere.
#
# 0.6 rad (~34°) is physically reasonable for a spur-gear hobby servo with a
# plastic output shaft.
_DELTA_SAT = 0.6
MG90S_STIFFNESS = MG90S_EFFORT_LIMIT / _DELTA_SAT

# Damping (Kd)
MG90S_DAMPING = MG90S_EFFORT_LIMIT / MG90S_VELOCITY_LIMIT  # 0.0033 N*m*s/rad

# Default (standing) joint positions. They define "action = 0"
# (the neutral posture the policy holds when outputting zeros).
COXA_DEFAULT  =  0.00
FEMUR_DEFAULT = -0.20
TIBIA_DEFAULT = -1.45

# Joint hard limits
COXA_LIMS  = (-0.785, 0.785)
FEMUR_LIMS = (-1.571, 1.571)
TIBIA_LIMS = (-2.356, 2.356)
SMALLEST_ABS_LIM = min([
    abs(lim) for lim in [
        *COXA_LIMS,
        *FEMUR_LIMS,
        *TIBIA_LIMS
    ]
])

_SOFT = 0.9   # fraction of hard limits used as soft limits

# Robot standing initially, no need to learn how to get up.
# We need to manually tune the initial height. Using a kinematic
# model to derive this will defy the sole purpose of using DRL:
# it's difficult to have models for complex robots.
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.05),
    joint_pos={
        COXA_JOINT_REGEX: COXA_DEFAULT,
        FEMUR_JOINT_REGEX: FEMUR_DEFAULT,
        TIBIA_JOINT_REGEX: TIBIA_DEFAULT,
    },
    joint_vel={".*": 0.0},
)

# Actuator configuration
ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*", ),
    stiffness=MG90S_STIFFNESS,
    damping=MG90S_DAMPING,
    effort_limit=MG90S_EFFORT_LIMIT,
    armature=MG90S_ARMATURE,
    viscous_damping=0.0
)

ARTICULATIONS = EntityArticulationInfoCfg(
    actuators=(ACTUATOR,),
    soft_joint_pos_limit_factor=_SOFT,
)


def _range_center_and_scale(lims: tuple[float, float], soft: float) -> tuple[float, float]:
    """Center and half-width of the soft-limited range.
    action in [-1, 1] maps to [soft_lo, soft_hi] = center ± scale.
    """
    lo, hi = lims
    center = (lo + hi) / 2.0
    scale  = (hi - lo) / 2.0 * soft
    return center, scale

def _scale_from_default(
        lim: tuple[float, float],
        default: float,
        soft: float = _SOFT,
) -> float:
    """Maximum symmetric excursion from *default* that stays within soft limits."""
    lo, hi = lim
    half = (hi - lo) / 2.0
    centre = (lo + hi) / 2.0
    soft_lo = centre - half * soft
    soft_hi = centre + half * soft
    return min(default - soft_lo, soft_hi - default)

def _joint_value_dict(coxa_val: float, femur_val: float, tibia_val: float) -> dict[str, float]:
    return {
        name: (
            coxa_val  if name.endswith("coxa")  else
            femur_val if name.endswith("femur") else
            tibia_val
        )
        for name in JOINT_NAMES
    }


COXA_SCALE  = min(_scale_from_default(COXA_LIMS,  COXA_DEFAULT),  _DELTA_SAT)
FEMUR_SCALE = min(_scale_from_default(FEMUR_LIMS, FEMUR_DEFAULT), _DELTA_SAT)
TIBIA_SCALE = min(_scale_from_default(TIBIA_LIMS, TIBIA_DEFAULT), _DELTA_SAT)

ACTION_SCALE  = _joint_value_dict(COXA_SCALE, FEMUR_SCALE, TIBIA_SCALE)
ACTION_OFFSET = _joint_value_dict(COXA_DEFAULT, FEMUR_DEFAULT, TIBIA_DEFAULT)


if __name__ == "__main__":

    print("Joint ranges")
    print(f"{'Joint':35s} {'Type':5s}  {'lo':>7s}  {'hi':>7s}  scale")

    for name in JOINT_NAMES:

        offset = ACTION_OFFSET[name]
        scale  = ACTION_SCALE[name]
        lo, hi = offset - scale, offset + scale
        jt = ("COXA" if name.endswith("coxa") else
              "FEMUR" if name.endswith("femur") else "TIBIA")
        print(f"{name:35s} ({jt:5s})  {lo:7.3f}  {hi:7.3f}  {scale:.3f}")

    print("\nActuators specs")
    print(f"{'Parameter':35s} {'Value':5s}")

    params = {
        "Damping": MG90S_DAMPING,
        "Stiffness": MG90S_STIFFNESS,
        "Armature": MG90S_ARMATURE,
        "Velocity limit": MG90S_VELOCITY_LIMIT,
        "Effort limit": MG90S_EFFORT_LIMIT,
    }
    for name, value in params.items():
        print(f"{name:35s} {value:<7.3f}")