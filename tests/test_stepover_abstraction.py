"""Tests for the step-over swing-foot clearance signal."""

import torch

from mjlab.tasks.stepover.mdp.abstractions import _clearance_signal

_KW = dict(crossing_band=0.25, clearance_std=0.1, air_threshold=0.03)


def test_clearance_zero_when_no_foot_over_barrier() -> None:
  # Both feet far from the barrier x: nothing is crossing, reward is 0.
  foot_x = torch.tensor([[0.0, 0.0]])
  foot_z = torch.tensor([[0.2, 0.2]])
  barrier_x = torch.tensor([[2.0]])
  target_z = torch.tensor([[0.3]])
  out = _clearance_signal(foot_x, foot_z, barrier_x, target_z, **_KW)
  torch.testing.assert_close(out, torch.tensor([0.0]))


def test_clearance_rewards_foot_at_via_point() -> None:
  # One foot over the barrier, lifted to the via-point height: reward ~1.
  foot_x = torch.tensor([[2.0, 0.0]])
  foot_z = torch.tensor([[0.3, 0.05]])
  barrier_x = torch.tensor([[2.0]])
  target_z = torch.tensor([[0.3]])
  out = _clearance_signal(foot_x, foot_z, barrier_x, target_z, **_KW)
  torch.testing.assert_close(out, torch.tensor([1.0]), atol=1e-6, rtol=0.0)


def test_clearance_penalizes_foot_below_via_point() -> None:
  # A foot over the barrier but too low scores less than one at the via-point;
  # clearing higher than the via-point is not penalized.
  barrier_x = torch.tensor([[2.0]])
  target_z = torch.tensor([[0.3]])
  low = _clearance_signal(
    torch.tensor([[2.0, 0.0]]), torch.tensor([[0.15, 0.05]]), barrier_x, target_z, **_KW
  )
  at = _clearance_signal(
    torch.tensor([[2.0, 0.0]]), torch.tensor([[0.30, 0.05]]), barrier_x, target_z, **_KW
  )
  high = _clearance_signal(
    torch.tensor([[2.0, 0.0]]), torch.tensor([[0.45, 0.05]]), barrier_x, target_z, **_KW
  )
  assert (low < at).all()
  torch.testing.assert_close(high, at, atol=1e-6, rtol=0.0)


def test_clearance_ignores_grounded_foot() -> None:
  # A foot over the barrier x but still on the ground (below air_threshold) does
  # not count as swinging, so it contributes no clearance reward.
  foot_x = torch.tensor([[2.0, 0.0]])
  foot_z = torch.tensor([[0.01, 0.05]])
  barrier_x = torch.tensor([[2.0]])
  target_z = torch.tensor([[0.3]])
  out = _clearance_signal(foot_x, foot_z, barrier_x, target_z, **_KW)
  torch.testing.assert_close(out, torch.tensor([0.0]))
