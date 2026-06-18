"""Differential-drive parameters and the torque-driven unicycle model.

Two roles:

* It is the **plant model the analytic skills are tuned against** -- the skills in
  :mod:`mjlab.tasks.diffdrive.skills` read physical parameters (mass, inertia, the
  wheel-torque mapping) from a :class:`DiffDrive` instance. These parameters mirror
  ``diffdrive.xml``, so a skill tuned here behaves the same on the MuJoCo model.
* It is a small, vectorized numpy simulator (:meth:`DiffDrive.step` /
  :meth:`DiffDrive.rollout`) for any offline analysis that does not need full
  MuJoCo contact -- it is *not* used by the live viewer, which steps the real
  ``diffdrive.xml`` model.

State ``s = [x, y, theta, v, omega]``:

* ``x``, ``y`` -- chassis position in the world plane (m).
* ``theta`` -- heading (rad), ``0`` = ``+x``.
* ``v`` -- forward (body-x) speed (m/s).
* ``omega`` -- yaw rate (rad/s).

Input ``u = [tau_l, tau_r]``: left/right wheel torques (N*m), each saturated to
``+/-torque_max``. The wheels map to a forward force and a yaw moment::

    F = (tau_l + tau_r) / r
    M = (tau_r - tau_l) * b / r

with ``r`` the wheel radius and ``b`` the half-axle width. Bounded torque is what
makes some fast states unrecoverable in time -- the source of a bounded funnel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# State indices.
X, Y, THETA, V, OMEGA = 0, 1, 2, 3, 4
STATE_DIM = 5


def wrap_angle(a: np.ndarray) -> np.ndarray:
  """Wrap angle(s) to ``(-pi, pi]``."""
  return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class DiffDrive:
  """Torque-driven unicycle. All methods are vectorized over a leading batch."""

  wheel_radius: float = 0.06  # r (m), matches diffdrive.xml wheel size.
  half_axle: float = 0.11  # b (m), wheel offset from the center line.
  mass: float = 2.4  # chassis + wheels (kg).
  inertia: float = 0.03  # yaw inertia about the center (kg*m^2).
  lin_damping: float = 0.5  # forward viscous damping (N*s/m).
  ang_damping: float = 0.02  # yaw viscous damping (N*m*s/rad).
  torque_max: float = 2.0  # per-wheel torque limit, matches ctrlrange in XML.
  dt: float = 0.01  # analysis step (s).

  # Input handling.

  def saturate(self, u: np.ndarray) -> np.ndarray:
    """Clip wheel torques to the actuator limit -- the source of bounded funnels."""
    return np.clip(u, -self.torque_max, self.torque_max)

  def wheel_to_body(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map ``[tau_l, tau_r]`` to forward force ``F`` and yaw moment ``M``."""
    tau_l, tau_r = u[..., 0], u[..., 1]
    f = (tau_l + tau_r) / self.wheel_radius
    m = (tau_r - tau_l) * self.half_axle / self.wheel_radius
    return f, m

  def body_to_wheel(self, f: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Inverse of :meth:`wheel_to_body`: desired ``F``, ``M`` -> wheel torques.

    A controller may request more than the actuator can deliver; the plant clips
    it (here :meth:`saturate`, in MuJoCo the actuator ``ctrlrange``). That clipping
    is precisely what makes some fast states unrecoverable in time.
    """
    half = 0.5 * self.wheel_radius
    tau_l = half * (f - m / self.half_axle)
    tau_r = half * (f + m / self.half_axle)
    return np.stack([tau_l, tau_r], axis=-1)

  # Numpy simulator (offline analysis only; the viewer uses MuJoCo).

  def deriv(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Continuous-time state derivative ``s_dot`` for state ``s``, input ``u``."""
    f, m = self.wheel_to_body(self.saturate(u))
    theta, v, omega = s[..., THETA], s[..., V], s[..., OMEGA]
    ds = np.zeros_like(s)
    ds[..., X] = v * np.cos(theta)
    ds[..., Y] = v * np.sin(theta)
    ds[..., THETA] = omega
    ds[..., V] = (f - self.lin_damping * v) / self.mass
    ds[..., OMEGA] = (m - self.ang_damping * omega) / self.inertia
    return ds

  def step(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Advance one ``dt`` with explicit midpoint (RK2); the input is held over dt."""
    ds = self.deriv(s, u)
    s_mid = s + 0.5 * self.dt * ds
    s_next = s + self.dt * self.deriv(s_mid, u)
    s_next[..., THETA] = wrap_angle(s_next[..., THETA])
    return s_next

  def rollout(self, s0: np.ndarray, controller, steps: int) -> np.ndarray:
    """Roll ``controller`` (state -> wheel torques) from ``s0`` for ``steps``.

    Returns the trajectory of shape ``(steps + 1, ..., STATE_DIM)`` including the
    initial state.
    """
    s = np.asarray(s0, dtype=float).copy()
    traj = np.empty((steps + 1, *s.shape), dtype=float)
    traj[0] = s
    for t in range(steps):
      s = self.step(s, controller(s))
      traj[t + 1] = s
    return traj
