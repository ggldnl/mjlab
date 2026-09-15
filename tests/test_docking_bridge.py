"""Small checks for the docking bridge state machine."""

import torch
from torch import nn

from mjlab.tasks.bridging.experiments.humanoid.bridges.docking.bridge import (
  DockingBridge,
  Tolerances,
  channel_errors,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.docking.command import (
  DockingCommandCfg,
  _align_target_route,
  _capture_age,
  encode_sequence,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.docking.env_cfg import (
  docking_env_cfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.interface import Bridge
from mjlab.utils.lab_api.math import quat_apply, quat_from_angle_axis, quat_mul


class ConstantPolicy(nn.Module):
  def __init__(self, value: float) -> None:
    super().__init__()
    self.value = value

  def forward(
    self, history: torch.Tensor, target: torch.Tensor, time_left: torch.Tensor
  ) -> torch.Tensor:
    del target, time_left
    return history.new_full((history.shape[0], 2), self.value)


def state() -> torch.Tensor:
  value = torch.zeros(1, 17)
  value[:, 3] = 1.0
  return value


def test_play_config_enables_target_visualization() -> None:
  command = docking_env_cfg(play=True).commands["bridge"]
  assert isinstance(command, DockingCommandCfg)
  assert command.debug_vis


def test_no_op_bridge_hands_off_without_applying_zeros() -> None:
  bridge = Bridge(action_dim=2)
  current = state()[:, None]
  target = state()[:, None].repeat(1, 3, 1)
  output = bridge(current, target, torch.ones(1))
  assert output.handoff.item()


def test_target_action_stays_at_entry_until_capture() -> None:
  age = _capture_age(
    torch.tensor((30, 30)),
    torch.tensor((-1, 27)),
    torch.tensor((False, True)),
  )
  torch.testing.assert_close(age, torch.tensor((0, 3)))


def test_target_route_is_placed_at_independent_start() -> None:
  route_start = state()
  route_start[:, 0] = 2.0
  actual_start = state()
  actual_start[:, 0] = 10.0
  targets = route_start[:, None].repeat(1, 3, 1)
  targets[0, :, 0] += torch.tensor((0.5, 1.0, 1.5))
  placed = _align_target_route(targets, route_start, actual_start)
  torch.testing.assert_close(placed[0, :, 0], torch.tensor((10.5, 11.0, 11.5)))


def test_channel_errors_use_worst_joint() -> None:
  actual = state()
  target = state()
  actual[:, 13:15] = torch.tensor((0.1, 0.4))
  actual[:, 15:17] = torch.tensor((0.2, 0.7))
  errors = channel_errors(actual, target)
  torch.testing.assert_close(errors[0, 4:], torch.tensor((0.4, 0.7)))


def test_capture_is_latched_until_blended_handoff() -> None:
  bridge = DockingBridge(
    ConstantPolicy(1.0),
    ConstantPolicy(2.0),
    action_dim=2,
    tolerances=Tolerances(),
    blend_steps=2,
  )
  current = state()[:, None].repeat(1, 2, 1)
  target = state()[:, None].repeat(1, 3, 1)
  first = bridge(current, target, torch.ones(1))
  assert first.captured.item()
  assert not first.handoff.item()
  torch.testing.assert_close(first.action, torch.full((1, 2), 2.0))

  current[:, -1, 0] = 10.0
  second = bridge(current, target, torch.ones(1))
  assert second.captured.item()
  assert second.handoff.item()
  assert second.blend.item() == 1.0


def test_timeout_hands_off_even_without_capture() -> None:
  bridge = DockingBridge(ConstantPolicy(1.0), ConstantPolicy(2.0), action_dim=2)
  current = state()[:, None]
  target = state()[:, None].repeat(1, 3, 1)
  target[:, 1, 0] = 10.0
  output = bridge(current, target, torch.zeros(1))
  assert output.handoff.item()
  assert not output.captured.item()
  assert output.blend.item() == 1.0


def test_sequence_encoding_ignores_world_yaw_and_translation() -> None:
  sequence = state()[:, None].repeat(1, 3, 1)
  sequence[0, :, 0] = torch.tensor((0.0, 0.1, 0.3))
  sequence[0, :, 7] = torch.tensor((0.1, 0.2, 0.4))
  anchor = sequence[:, -1]

  axis = torch.tensor(((0.0, 0.0, 1.0),))
  rotation = quat_from_angle_axis(torch.tensor((1.2,)), axis)
  turn = rotation[:, None].expand(-1, sequence.shape[1], -1)
  moved = sequence.clone()
  moved[..., 0:3] = quat_apply(turn, sequence[..., 0:3]) + torch.tensor(
    (4.0, -2.0, 0.0)
  )
  moved[..., 3:7] = quat_mul(turn, sequence[..., 3:7])
  moved[..., 7:10] = quat_apply(turn, sequence[..., 7:10])
  moved[..., 10:13] = quat_apply(turn, sequence[..., 10:13])

  torch.testing.assert_close(
    encode_sequence(sequence, anchor),
    encode_sequence(moved, moved[:, -1]),
    atol=1e-6,
    rtol=1e-6,
  )
