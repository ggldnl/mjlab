"""Differential-drive bridging testbed.

Contents:

* ``diffdrive.xml`` -- the MuJoCo model (chassis + two driven wheels + caster).
* ``dynamics`` -- physical parameters + the wheel-torque mapping the skills use
  (and a small numpy simulator for offline analysis).
* ``skills`` -- the analytic skills ``drive`` and ``goal``, plus a viser viewer
  (``python -m mjlab.tasks.diffdrive.skills``) to watch one at a time.

No RL task is registered yet: the skills are analytic, so there is nothing to
train here. The learned bridge controller (and the env config it needs) comes
later.
"""
