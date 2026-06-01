"""Tests for the ballistic jump abstraction math."""

import math

import torch

from mjlab.tasks.jump.mdp.abstractions import (
  _ballistic_point,
  _progress_weight,
  _solve_ballistic,
)

_GRAVITY = 9.81


def test_progress_weight_endpoints_and_monotonicity() -> None:
  p = torch.linspace(0.0, 1.0, 11)
  for rate in (0.0, 3.0, 6.0):
    w = _progress_weight(p, rate)
    # Rises from 0 at takeoff to 1 at landing.
    torch.testing.assert_close(w[0], torch.tensor(0.0), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(w[-1], torch.tensor(1.0), atol=1e-6, rtol=0.0)
    # Monotonically increasing.
    assert torch.all(w[1:] >= w[:-1])


def test_progress_weight_backloaded() -> None:
  # A positive rate concentrates weight near the end: at the midpoint the
  # exponential ramp is below the linear ramp.
  mid = torch.tensor([0.5])
  assert _progress_weight(mid, 3.0).item() < _progress_weight(mid, 0.0).item()


def test_ballistic_reaches_target() -> None:
  start = torch.tensor([[0.0, 0.0, 0.665], [1.0, -2.0, 0.665]], dtype=torch.float32)
  target = torch.tensor([[1.5, 0.3, 0.665], [2.4, -2.0, 0.665]], dtype=torch.float32)
  apex = torch.tensor([0.4, 0.3], dtype=torch.float32)

  v0, flight_time = _solve_ballistic(start, target, apex, gravity=_GRAVITY)

  landing = _ballistic_point(start, v0, flight_time, gravity=_GRAVITY)
  torch.testing.assert_close(landing, target, atol=1e-4, rtol=0.0)


def test_apex_height_matches() -> None:
  start = torch.tensor([[0.0, 0.0, 0.665]], dtype=torch.float32)
  target = torch.tensor([[1.5, 0.0, 0.665]], dtype=torch.float32)
  apex = torch.tensor([0.4], dtype=torch.float32)

  v0, _ = _solve_ballistic(start, target, apex, gravity=_GRAVITY)

  # Vertical launch speed implies apex = vz0^2 / (2 g).
  vz0 = v0[:, 2]
  measured_apex = vz0**2 / (2 * _GRAVITY)
  torch.testing.assert_close(measured_apex, apex, atol=1e-5, rtol=0.0)

  # Time to apex.
  t_up = vz0 / _GRAVITY
  apex_point = _ballistic_point(start, v0, t_up, gravity=_GRAVITY)
  expected_z = start[:, 2] + apex
  torch.testing.assert_close(apex_point[:, 2], expected_z, atol=1e-5, rtol=0.0)


def test_apex_clamped_above_target() -> None:
  # Requesting an apex below the height drop should still be feasible: the
  # solver clamps the apex above the higher endpoint.
  start = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float32)
  target = torch.tensor([[1.0, 0.0, 1.2]], dtype=torch.float32)  # Higher target.
  apex = torch.tensor([0.1], dtype=torch.float32)  # Too low to reach the target.

  v0, flight_time = _solve_ballistic(start, target, apex, gravity=_GRAVITY)

  assert torch.isfinite(flight_time).all()
  assert (flight_time > 0).all()
  landing = _ballistic_point(start, v0, flight_time, gravity=_GRAVITY)
  torch.testing.assert_close(landing, target, atol=1e-4, rtol=0.0)


def test_symmetric_flight_time() -> None:
  # Equal-height endpoints: flight time is symmetric, T = 2 * sqrt(2 h / g).
  start = torch.tensor([[0.0, 0.0, 0.665]], dtype=torch.float32)
  target = torch.tensor([[1.5, 0.0, 0.665]], dtype=torch.float32)
  apex = torch.tensor([0.45], dtype=torch.float32)

  _, flight_time = _solve_ballistic(start, target, apex, gravity=_GRAVITY)
  expected = 2.0 * math.sqrt(2.0 * 0.45 / _GRAVITY)
  torch.testing.assert_close(flight_time, torch.tensor([expected]), atol=1e-5, rtol=0.0)
