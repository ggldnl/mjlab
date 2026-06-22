"""Constants shared by the bridge's training and deployment halves.

The training environment and the deployed LearnedBridge must agree on what the
policy sees, how its action becomes a twist, and what counts as success.
"""

from __future__ import annotations

import math

# Control rate. Control dt = TIMESTEP * DECIMATION = 0.02 s (50 Hz), and the
# harvested rollouts use the same rate so their states match the trained dynamics
TIMESTEP = 0.005
DECIMATION = 4

# Twist action mapping. The policy emits a raw 2-vector (roughly in [-1, 1]); it
# maps to a body twist (forward speed v, yaw rate omega) and is clamped to physical
# limits
V_OFFSET = 1.0
V_SCALE = 1.5
OMEGA_SCALE = 4.0
V_MIN, V_MAX = -0.5, 2.5
OMEGA_MAX = 4.0

# Wheel-velocity servo, matching DiffDrive: torque = KV * (target - actual), clamped
# to the actuator's ctrlrange
KV = 3.0
TORQUE_LIMIT = 0.6

# Goal-closeness weights
W_POS = 1.0
W_HEAD = 0.3
W_SPEED = 0.3

# Success: the robot has reached the next skill's tube and can hand over
POS_TOL = 0.15
HEAD_TOL = math.radians(20.0)
SPEED_TOL = 0.4

# Margins for the smooth tracking reward (the reward is near 1 within the margin,
# decaying outside it)
M_POS = 0.6
M_HEAD = math.radians(45.0)
M_SPEED = 1.0

# Rollout harvesting. One duration, in seconds, sets both saved windows: the next
# skill's early tube (the bridge's candidate goals) and the previous skill's approach to
# the junction (the states leading up to an interrupt). Seconds, not a raw tick count, so
# the horizon stays meaningful independent of the control rate.
CONTROL_DT = TIMESTEP * DECIMATION  # seconds per control tick (0.02 s)
WINDOW_SECONDS = 1.0  # duration of each saved window
WINDOW_STEPS = round(WINDOW_SECONDS / CONTROL_DT)  # control ticks per window
N_INTERRUPTS = 50  # interrupt states harvested per corridor transition

# The target tube is harvested as a family of rollouts, one per sampled initiation
# state, then reduced to a single representative the bridge aims to join. A complex robot
# has no single perfect start, only an initiation set, so the tube is a family, not a line.
WINDOW_SAMPLES = 24  # initiation-set rollouts forming the target tube family
WINDOW_REPRESENTATIVE = "medoid"  # how to reduce the family: "medoid" or "mean"

OBS_DIM = 7
ACTION_DIM = 2
