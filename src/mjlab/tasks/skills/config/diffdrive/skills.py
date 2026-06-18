"""Diff-drive skill primitives. Each is a policy ``state -> wheel torques``.

    drive_straight  ->  accelerate to a high cruise speed and hold heading. It does
                        not decelerate -- there is no "slow down" command.
    turn_left/right ->  a slow-speed in-place turn: apply yaw only, with NO forward
                        speed control. At low speed friction keeps it roughly in
                        place; from high speed it cannot bleed the momentum, so it
                        skids forward in a wide arc -- it "only knows how to turn
                        slowly".

Each skill is its own small controller: it picks a target body twist and the shared
tracker (``DiffDrive.twist_to_torque``) maps it to wheel torques with skill-specific
gains. The key asymmetry lives in those gains: ``drive`` has strong forward
authority, ``turn`` has none (``kp_v = 0``), so neither can gracefully bleed speed
-- that deceleration is exactly the bridge's job.
"""

from __future__ import annotations

import numpy as np

from mjlab.tasks.skills.config.diffdrive.dynamics import DiffDrive

_DD = DiffDrive()
V_CRUISE = 2.5  # drive_straight target speed (m/s).
OMEGA_TURN = 3.0  # turn_left / turn_right yaw rate (rad/s).


def drive_straight(state: np.ndarray) -> np.ndarray:
  """Accelerate to cruise speed and hold heading (never decelerates)."""
  return _DD.twist_to_torque(state, V_CRUISE, 0.0, kp_v=12.0, kp_w=120.0)


def turn_left(state: np.ndarray) -> np.ndarray:
  """Slow-speed turn left: yaw torque only, no forward braking (kp_v=0)."""
  return _DD.twist_to_torque(state, 0.0, OMEGA_TURN, kp_v=0.0, kp_w=120.0)


def turn_right(state: np.ndarray) -> np.ndarray:
  """Slow-speed turn right: yaw torque only, no forward braking (kp_v=0)."""
  return _DD.twist_to_torque(state, 0.0, -OMEGA_TURN, kp_v=0.0, kp_w=120.0)


SKILLS = {
  "drive_straight": drive_straight,
  "turn_left": turn_left,
  "turn_right": turn_right,
}
