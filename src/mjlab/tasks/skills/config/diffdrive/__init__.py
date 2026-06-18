"""Differential-drive configuration for the skill-bridging project.

* ``diffdrive.xml`` -- the MuJoCo model (chassis + two driven wheels + caster).
* ``dynamics`` -- physical parameters, the wheel-torque mapping, the twist tracker
  (``twist_to_torque``) and state extraction (``state_from_mjdata``).
* ``skills`` -- the analytic primitives ``drive_straight``, ``turn_left``,
  ``turn_right`` (each a constant body twist).
"""
