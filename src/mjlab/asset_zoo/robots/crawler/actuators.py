"""Crawler actuators: all hardware parameters."""

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.entity import EntityCfg
import torch
import math


LEG_1_JOINT_NAMES = (
  "base_leg_1_coxa",
  "leg_1_coxa_leg_1_femur",
  "leg_1_femur_leg_1_tibia",
)

LEG_2_JOINT_NAMES = (
  "base_leg_2_coxa",
  "leg_2_coxa_leg_2_femur",
  "leg_2_femur_leg_2_tibia",
)

LEG_3_JOINT_NAMES = (
  "base_leg_3_coxa",
  "leg_3_coxa_leg_3_femur",
  "leg_3_femur_leg_3_tibia",
)

LEG_4_JOINT_NAMES = (
  "base_leg_4_coxa",
  "leg_4_coxa_leg_4_femur",
  "leg_4_femur_leg_4_tibia",
)

COXA_JOINT_REGEX = "base_leg_[1-4]_coxa"
FEMUR_JOINT_REGEX = "leg_[1-4]_coxa_leg_[1-4]_femur"
TIBIA_JOINT_REGEX = "leg_[1-4]_femur_leg_[1-4]_tibia"

JOINT_NAMES = [
  *LEG_1_JOINT_NAMES,
  *LEG_2_JOINT_NAMES,
  *LEG_3_JOINT_NAMES,
  *LEG_4_JOINT_NAMES,
]

# Diagonal-pair trot offsets: legs 0 and 2 are in phase, legs 1 and 3 are anti-phase.
# To use a different gait (walk, bound), change these offsets only — nothing else.
LEG_PHASE_OFFSETS = torch.tensor([0.0, math.pi, 0.0, math.pi])

# Scale actuator parameters: a small robot is difficult to simulate in mjlab
SCALE = 1.0
MASS_SCALE = SCALE**3
INERTIA_SCALE = SCALE**5
TORQUE_SCALE = SCALE**4

# The MG90S is a brushed-DC servo with an internal PID position loop.
# We model it in MuJoCo as a position actuator (stiffness + damping + effort_limit).

# Effort limit: 2.2 Kgf*cm at 6V (datasheet)
MG90S_STALL_TORQUE_KGF_CM = 2.2
KGF_CM_TO_NM = 9.81 * 0.01
MG90S_EFFORT_LIMIT = MG90S_STALL_TORQUE_KGF_CM * KGF_CM_TO_NM  # 0.216 N*m
MG90S_EFFORT_LIMIT *= TORQUE_SCALE

# Velocity limit: 0.08 s/60° at 6V (datasheet) -> 13.1 rad/s
MG90S_NO_LOAD_SPEED_S_PER_60DEG = 0.08
MG90S_VELOCITY_LIMIT = (math.pi / 3.0) / MG90S_NO_LOAD_SPEED_S_PER_60DEG  # 13.1 rad/s
MG90S_JOINT_DAMPING = MG90S_EFFORT_LIMIT / MG90S_VELOCITY_LIMIT

# Reflected rotor inertia
# Coreless brushed rotor: ~1.5 g, radius ~3 mm -> J_rotor ~= 7e-9 kg*m^2
# Spur gear train, N ~= 5.5 -> J_arm = J_rotor * N^2 ~= 2.1e-7 kg*m^2
MG90S_ROTOR_INERTIA = 7e-9  # kg*m^2
MG90S_GEAR_RATIO = 5.5
MG90S_ARMATURE = MG90S_ROTOR_INERTIA * MG90S_GEAR_RATIO**2  # 2.1e-7 kg*m^2
MG90S_ARMATURE *= INERTIA_SCALE

# Effective inetias per joint type
# These are the load inertias seen at the joint output shaft, obtained by
# summing the contributions of all downstream rigid bodies projected onto
# the joint rotation axis.  Estimated from the MJCF inertial tags:
#
#   tibia  <- tibia body + foot body
#   femur  <- femur body + tibia body + foot body
#   coxa   <- coxa body + femur body + tibia body + foot body
EFFECTIVE_INERTIAS = {
  "coxa": 9.0e-6 * INERTIA_SCALE,  # kg*m^2
  "femur": 5.0e-6 * INERTIA_SCALE,  # kg*m^2
  "tibia": 8.0e-7 * INERTIA_SCALE,  # kg*m^2
}

NATURAL_FREQ = 50.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0


def _pd_gains(joint_type: str) -> tuple[float, float]:
  """Return (stiffness, damping) for the given joint type."""
  j = EFFECTIVE_INERTIAS[joint_type]
  k = j * NATURAL_FREQ**2
  d = 2.0 * DAMPING_RATIO * j * NATURAL_FREQ
  return k, d


_coxa_k, _coxa_d = _pd_gains("coxa")
_femur_k, _femur_d = _pd_gains("femur")
_tibia_k, _tibia_d = _pd_gains("tibia")

COXA_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(COXA_JOINT_REGEX,),
  stiffness=_coxa_k,
  damping=_coxa_d,
  effort_limit=MG90S_EFFORT_LIMIT,
  armature=MG90S_ARMATURE,
)

FEMUR_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(FEMUR_JOINT_REGEX,),
  stiffness=_femur_k,
  damping=_femur_d,
  effort_limit=MG90S_EFFORT_LIMIT,
  armature=MG90S_ARMATURE,
)

TIBIA_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(TIBIA_JOINT_REGEX,),
  stiffness=_tibia_k,
  damping=_tibia_d,
  effort_limit=MG90S_EFFORT_LIMIT,
  armature=MG90S_ARMATURE,
)


# Default (standing) joint positions. They define "action = 0"
# (the neutral posture the policy holds when outputting zeros).
#
# These must place the feet *below* the base so the robot stands on them.
# On this model that requires a POSITIVE femur: femur=+0.5 puts the feet ~4.8 cm
# below the base, giving an upright stance with all four feet planted and no
# body/leg ground contact (verified in sim). Negative femur folds the legs
# upward (feet above the base) and the robot rests on its hips -- do not use it.
COXA_DEFAULT = 0.00
FEMUR_DEFAULT = 0.50
TIBIA_DEFAULT = -1.00

# Base height (m) the robot holds at the neutral stance above, measured in sim.
# Used as the height-reward target and to set a sane spawn height.
NOMINAL_BASE_HEIGHT = 0.048 * SCALE

# Joint hard limits
COXA_LIMS = (-0.785, 0.785)
FEMUR_LIMS = (-1.571, 1.571)
TIBIA_LIMS = (-2.356, 2.356)
SMALLEST_ABS_LIM = min([abs(lim) for lim in [*COXA_LIMS, *FEMUR_LIMS, *TIBIA_LIMS]])

_SOFT = 0.9  # fraction of hard limits used as soft limits

# Robot standing initially, no need to learn how to get up.
# Spawn just above the neutral stance height so the feet settle onto the
# ground instead of free-falling the body onto it each reset.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, NOMINAL_BASE_HEIGHT + 0.005),
  joint_pos={
    COXA_JOINT_REGEX: COXA_DEFAULT,
    FEMUR_JOINT_REGEX: FEMUR_DEFAULT,
    TIBIA_JOINT_REGEX: TIBIA_DEFAULT,
  },
  joint_vel={".*": 0.0},
)


ARTICULATIONS = EntityArticulationInfoCfg(
  actuators=(COXA_ACTUATOR, FEMUR_ACTUATOR, TIBIA_ACTUATOR),
  soft_joint_pos_limit_factor=_SOFT,
)


def _range_center_and_scale(
  lims: tuple[float, float], soft: float
) -> tuple[float, float]:
  """Center and half-width of the soft-limited range.
  action in [-1, 1] maps to [soft_lo, soft_hi] = center ± scale.
  """
  lo, hi = lims
  center = (lo + hi) / 2.0
  scale = (hi - lo) / 2.0 * soft
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


def _joint_value_dict(
  coxa_val: float, femur_val: float, tibia_val: float
) -> dict[str, float]:
  return {
    name: (
      coxa_val
      if name.endswith("coxa")
      else femur_val
      if name.endswith("femur")
      else tibia_val
    )
    for name in JOINT_NAMES
  }


# Per-joint action ranges (rad), i.e. the max excursion from the neutral stance
# that action = +/-1 commands. These are set PER JOINT from what a real gait
# needs, not as one uniform fraction:
#
#   coxa  - the propulsion joint. Its swing is ~horizontal (dz~=0), so a large
#           range barely affects tipping but directly sets stride length. An
#           open-loop trot needs ~+/-0.5 rad of coxa to walk at ~0.11 m/s; the
#           earlier uniform 0.35x fraction capped it at 0.247 rad, starving
#           propulsion and pushing the policy toward drag/vibration exploits.
#   femur - the lift joint. It moves the foot vertically and shifts the CoM, so
#           it dominates tipping; keep it moderate (the open-loop gait only needs
#           ~0.4 rad of lift).
#   tibia - secondary shaping; moderate.
#
# All stay within the soft joint limits from their defaults
# (coxa 0+/-0.5 < 0.707, femur 0.5+/-0.45 in [0.05,0.95] < 1.41, tibia
# -1.0+/-0.45 in [-1.45,-0.55] < 2.12).
COXA_SCALE = 0.50
FEMUR_SCALE = 0.45
TIBIA_SCALE = 0.45

ACTION_SCALE = _joint_value_dict(COXA_SCALE, FEMUR_SCALE, TIBIA_SCALE)
ACTION_OFFSET = _joint_value_dict(COXA_DEFAULT, FEMUR_DEFAULT, TIBIA_DEFAULT)


if __name__ == "__main__":
  print("Joint ranges")
  print(f"{'Joint':35s} {'Type':5s}  {'lo':>7s}  {'hi':>7s}  scale")

  for name in JOINT_NAMES:
    offset = ACTION_OFFSET[name]
    scale = ACTION_SCALE[name]
    lo, hi = offset - scale, offset + scale
    jt = (
      "COXA"
      if name.endswith("coxa")
      else "FEMUR"
      if name.endswith("femur")
      else "TIBIA"
    )
    print(f"{name:35s} ({jt:5s})  {lo:7.3f}  {hi:7.3f}  {scale:.3f}")

  print("\nActuator specs")
  print(f"\t{'Parameter':35s} {'Value':>12s}")

  for jt, (k, d) in [
    ("coxa", (_coxa_k, _coxa_d)),
    ("femur", (_femur_k, _femur_d)),
    ("tibia", (_tibia_k, _tibia_d)),
  ]:
    print(f"\t[{jt}]")
    print(f"\t{'Stiffness (N*m/rad)':33s} {k:12.6f}")
    print(f"\t{'Damping (N*m*s/rad)':33s} {d:12.6f}")
    wn = math.sqrt(k / EFFECTIVE_INERTIAS[jt])
    print(f"\t{'Implied w (rad/s)':33s} {wn:12.2f}  ({wn / (2 * math.pi):.2f} Hz)")

  print(f"\n{'Effort limit (N*m)':35s} {MG90S_EFFORT_LIMIT:12.4f}")
  print(f"{'Velocity limit (rad/s)':35s} {MG90S_VELOCITY_LIMIT:12.4f}")
  print(f"{'Armature (kg*m^2)':35s} {MG90S_ARMATURE:12.4e}")
