"""Reward supported foot swings instead of dragging or hopping."""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor, TerrainHeightSensor


def supported_swing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  height_sensor_name: str,
  command_name: str = "twist",
  target_height: float = 0.06,
  std: float = 0.02,
) -> torch.Tensor:
  contact = env.scene[sensor_name]
  height = env.scene[height_sensor_name]
  assert isinstance(contact, ContactSensor)
  assert isinstance(height, TerrainHeightSensor)
  assert contact.data.found is not None
  in_air = contact.data.found == 0
  single_support = in_air.sum(dim=-1) == 1
  clearance = torch.exp(-((height.data.heights - target_height) / std).square())
  command = env.command_manager.get_command(command_name)
  assert command is not None
  moving = command[:, :2].norm(dim=-1) > 0.05
  progress = (env.scene["robot"].data.root_link_lin_vel_b[:, 0] / 0.2).clamp(0.0, 1.0)
  swing = (clearance * in_air).sum(dim=-1) * single_support * moving * progress
  return swing - 2.0 * in_air.all(dim=-1).float()
