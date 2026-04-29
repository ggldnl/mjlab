"""Crawler actuators: all hardware parameters."""

import math
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg


# The MG90S is a brushed-DC servo with an internal PID position loop.
# We model it in MuJoCo as a position actuator (stiffness + damping + effort_limit).
# The three parameters below are the only externally observable quantities:
#   effort_limit  - stall torque
#   velocity_limit - no-load angular speed (used only for documentation here)
#   armature      - reflected rotor inertia

# Effort limit: 2.2 Kgf*cm at 6V (datasheet)
MG90S_STALL_TORQUE_KGF_CM = 2.2  # 2.2
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
# We choose delta_sat so that the tightest action range (tibia: ±0.370 rad,
# see CRAWLER_ACTION_SCALE below) stays well inside the linear regime of the
# actuator.  With delta_sat = 0.6 rad, a max tibia action (0.370 rad) produces
# 0.370 / 0.6 ~= 62 % of stall torque — the servo responds proportionally and
# learning gradients flow everywhere.
#
# 0.6 rad (~34°) is physically reasonable for a spur-gear hobby servo with a
# plastic output shaft.
_DELTA_SAT = 0.6
MG90S_STIFFNESS = MG90S_EFFORT_LIMIT / _DELTA_SAT  # 0.54 N*m/rad

# Damping (Kd): critical damping at the femur joint (largest effective inertia)
# I_eff accounts for tibia + foot rotating about the femur axis, plus armature:
#   tibia (~1.9 g at 40 mm from femur):  I ~= 3.0e-6 kg*m^2
#   foot  (~0.2 g at 70 mm from femur):  I ~= 1.0e-6 kg*m^2
#   armature (reflected rotor):              ~= 0.2e-6 kg*m^2
#   total: ~4.2e-6, rounded to 5e-6 for margin
_I_EFF = 5e-6  # kg*m^2
MG90S_DAMPING = 2.0 * math.sqrt(MG90S_STIFFNESS * _I_EFF)  # 0.0033 N*m*s/rad

# Joint names
CRAWLER_JOINT_NAMES = [
    "base_leg_1_coxa",  "leg_1_coxa_leg_1_femur",  "leg_1_femur_leg_1_tibia",
    "base_leg_2_coxa",  "leg_2_coxa_leg_2_femur",  "leg_2_femur_leg_2_tibia",
    "base_leg_3_coxa",  "leg_3_coxa_leg_3_femur",  "leg_3_femur_leg_3_tibia",
    "base_leg_4_coxa",  "leg_4_coxa_leg_4_femur",  "leg_4_femur_leg_4_tibia",
]

# Actuator configuration
CRAWLER_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(r".*_coxa$", r".*_femur$", r".*_tibia$"),
    stiffness=MG90S_STIFFNESS,
    damping=MG90S_DAMPING,
    effort_limit=MG90S_EFFORT_LIMIT,
    armature=MG90S_ARMATURE,
)

_SOFT = 0.9

CRAWLER_ARTICULATIONS = EntityArticulationInfoCfg(
    actuators=(CRAWLER_ACTUATOR,),
    soft_joint_pos_limit_factor=_SOFT,
)

# Joint hard limits
_COXA_LIM  = (-0.785,  0.785)
_FEMUR_LIM = (-1.571,  1.571)
_TIBIA_LIM = (-2.356,  2.356)

def _scale_offset_from_limits(
    lim: tuple[float, float],
    soft: float = _SOFT,
) -> tuple[float, float]:
    lo, hi = lim
    center = (lo + hi) / 2.0
    half_range = (hi - lo) / 2.0
    scale = half_range * soft  # shrink toward center by soft margin
    return scale, center

def _joint_value_dict(coxa_val: float, femur_val: float, tibia_val: float) -> dict[str, float]:
    return {
        name: (
            coxa_val  if name.endswith("coxa")  else
            femur_val if name.endswith("femur") else
            tibia_val
        )
        for name in CRAWLER_JOINT_NAMES
    }

_COXA_SCALE,  _COXA_OFFSET  = _scale_offset_from_limits(_COXA_LIM)
_FEMUR_SCALE, _FEMUR_OFFSET = _scale_offset_from_limits(_FEMUR_LIM)
_TIBIA_SCALE, _TIBIA_OFFSET = _scale_offset_from_limits(_TIBIA_LIM)

CRAWLER_ACTION_SCALE = _joint_value_dict(_COXA_SCALE,  _FEMUR_SCALE,  _TIBIA_SCALE)
CRAWLER_ACTION_OFFSET = _joint_value_dict(_COXA_OFFSET, _FEMUR_OFFSET, _TIBIA_OFFSET)


if __name__ == "__main__":

    for name in CRAWLER_JOINT_NAMES:
        lo = CRAWLER_ACTION_OFFSET[name] - CRAWLER_ACTION_SCALE[name]
        hi = CRAWLER_ACTION_OFFSET[name] + CRAWLER_ACTION_SCALE[name]

        if name.endswith("coxa"):
            jt = "COXA"
        elif name.endswith("femur"):
            jt = "FEMUR"
        else:
            jt = "TIBIA"

        print(f"{name:30s} ({jt:5s}): {lo:.3f} -> {hi:.3f}")