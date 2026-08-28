"""
Worked example: identify actuator parameters for a hobby servo.

Run this file directly once you have:
  1. A subclass of ActuatorInterface wired to your real motor.
  2. A rough idea of the load inertia (measure it or estimate from CAD).
  3. The stall torque from the motor spec sheet.

The script will:
  - collect excitation trajectories from the real motor
  - run trajectory-matching identification via MJX
  - print and save an mjlab BuiltinPositionActuatorCfg
"""

import math

import numpy as np
from sysid.collect import CollectionConfig
from sysid.export import write_config_file
from sysid.hardware.base import ActuatorInterface
from sysid.identify import IdentConfig, identify
from sysid.model import build_mj_model

# --- Replace this stub with your real hardware interface ---------------------


class MyServo(ActuatorInterface):
  """Placeholder — replace with your actual protocol (Dynamixel, ODrive, etc.)."""

  def __init__(self, port: str = "/dev/ttyUSB0", motor_id: int = 1):
    self._port = port
    self._motor_id = motor_id
    self._q = 0.0
    self._qdot = 0.0
    # Open serial / CAN / whatever here

  def enable(self) -> None:
    print("[hw] enable torque")
    # e.g. dxl.write1ByteTxRx(id, ADDR_TORQUE_ENABLE, 1)

  def disable(self) -> None:
    print("[hw] disable torque")

  def set_position_target(self, q_des: float) -> None:
    # Convert radians → encoder ticks and write to the motor
    # e.g. ticks = int((q_des + math.pi) / (2*math.pi) * 4096)
    pass

  def read_state(self) -> tuple[float, float]:
    # Read position and velocity from the motor driver
    # e.g. ticks, vel = dxl.read(...)
    # Simulate random noise for this stub
    noise = np.random.default_rng().normal(0, 1e-4)
    return self._q + noise, self._qdot

  @property
  def position_limits(self) -> tuple[float, float]:
    return -math.pi / 2, math.pi / 2  # ±90° for a typical servo

  @property
  def velocity_limit(self) -> float:
    return 5.24  # rad/s  (~50 RPM — typical hobby servo no-load speed)

  @property
  def effort_limit(self) -> float:
    return 0.18  # Nm — stall torque from spec sheet


# --- Configuration -----------------------------------------------------------

# Rigid bar load dimensions and mass (measure these from your test rig)
BAR_MASS_KG = 0.24
BAR_LENGTH_M = 0.352
# Moment of inertia of a thin rod rotating about one end: I = mL²/3
BAR_INERTIA_KGM2 = BAR_MASS_KG * BAR_LENGTH_M**2 / 3.0

STALL_TORQUE_NM = 0.18  # from the servo spec sheet
CTRL_TIMESTEP_S = 0.002  # 500 Hz — match your real controller rate

JOINT_NAMES = [
  "base_leg_1_coxa",
  "base_leg_2_coxa",
  "base_leg_3_coxa",
  "base_leg_4_coxa",
]


# --- Step 1: collect data from the real motor --------------------------------

actuator = MyServo(port="/dev/ttyUSB0", motor_id=1)

collect_cfg = CollectionConfig(
  duration_train=40.0,
  duration_test=10.0,
  sample_rate=1.0 / CTRL_TIMESTEP_S,
  n_modes=10,
  freq_min=0.5,
  freq_max=5.0,
  amp_fraction=0.7,
)

# Comment the next two lines out and load from disk on subsequent runs:
# data = collect(actuator, collect_cfg, save_path="sysid_coxa.npz")

# Load previously collected data instead:
import os
import sys

if os.path.exists("sysid_coxa.npz"):
  data = dict(np.load("sysid_coxa.npz"))
else:
  print("No data file found — run collect() on real hardware first.")
  sys.exit(0)


# --- Step 2: build the single-joint simulation model ------------------------

mj_model = build_mj_model(
  inertia_kgm2=BAR_INERTIA_KGM2,
  effort_limit_nm=STALL_TORQUE_NM,
  timestep=CTRL_TIMESTEP_S,
  q_limits_rad=actuator.position_limits,
  gravity=False,  # set True if the bar hangs under gravity during the test
  load_mass_kg=BAR_MASS_KG,
)


# --- Step 3: identify --------------------------------------------------------

ident_cfg = IdentConfig(
  segment_len=3,  # paper default; try 2-5
  overlap=1,
  n_epochs=3000,
  lr=1e-2,
  batch_size=2000,
  patience=200,
  init_kp=2.0,  # rough initial guess for this servo class
  init_kv=0.02,
  init_armature=1e-4,
  identify_inertia=False,  # set True if bar inertia is uncertain
)

result = identify(data, mj_model, ident_cfg)


# --- Step 4: export ----------------------------------------------------------

write_config_file(
  path="actuator_coxa_identified.py",
  result=result,
  effort_limit_nm=STALL_TORQUE_NM,
  joint_names=JOINT_NAMES,
  actuator_name="COXA",
  dr_spread=0.20,  # ±20% domain randomization around the identified values
)

# The generated file will contain something like:
#
# COXA_ACTUATOR = BuiltinPositionActuatorCfg(
#     target_names_expr=("base_leg_1_coxa", ...),
#     stiffness=3.684123,
#     damping=0.00055200,
#     effort_limit=0.180000,
#     armature=3.210000e-03,
# )
#
# COXA_ARTICULATIONS = EntityArticulationInfoCfg(
#     actuators=(COXA_ACTUATOR,),
#     soft_joint_pos_limit_factor=0.9,
# )
#
# STIFFNESS_RANGE = (2.947298, 4.420947)   # Nm/rad
# DAMPING_RANGE   = (0.00044160, 0.00066240)  # Nm·s/rad
# ARMATURE_RANGE  = (2.568e-03, 3.852e-03)  # kg·m²
