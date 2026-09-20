"""Equal-sized parallel environments with one tracker teacher per source."""

from typing import cast

import torch
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from torch import nn

from mjlab.rl import RslRlVecEnvWrapper


class PooledEnv(VecEnv):
  def __init__(self, groups: tuple[RslRlVecEnvWrapper, ...]) -> None:
    if (
      not groups
      or groups[0].num_envs < 1
      or len({group.num_envs for group in groups}) != 1
    ):
      raise ValueError("CVAE teacher groups must have equal nonzero sizes")
    if len({group.num_actions for group in groups}) != 1:
      raise ValueError("CVAE teacher groups must have the same action space")
    self.groups = groups
    self.group_size = groups[0].num_envs
    self.num_envs = sum(group.num_envs for group in groups)
    self.num_actions = groups[0].num_actions
    self.device = groups[0].device
    self.max_episode_length = groups[0].max_episode_length
    self.cfg = groups[0].cfg

  @property
  def unwrapped(self):
    return self.groups[0].unwrapped

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return torch.cat([group.episode_length_buf for group in self.groups])

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:
    for index, group in enumerate(self.groups):
      start = index * self.group_size
      group.episode_length_buf = value[start : start + self.group_size]

  def get_observations(self) -> TensorDict:
    return TensorDict.cat([group.get_observations() for group in self.groups], dim=0)

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    results = [
      group.step(action)
      for group, action in zip(self.groups, actions.split(self.group_size), strict=True)
    ]
    extras: dict = {
      "time_outs": torch.cat([result[3]["time_outs"] for result in results])
    }
    for key in ("episode", "log"):
      present = [result[3][key] for result in results if key in result[3]]
      if present:
        extras[key] = {
          name: torch.stack(
            [
              torch.as_tensor(item[name], device=self.device).float().mean()
              for item in present
              if name in item
            ]
          ).mean()
          for name in set().union(*(item.keys() for item in present))
        }
    return (
      TensorDict.cat([result[0] for result in results], dim=0),
      torch.cat([result[1] for result in results]),
      torch.cat([result[2] for result in results]),
      extras,
    )

  def close(self) -> None:
    for group in self.groups[1:]:
      group.close()


class RoutedTeacher(nn.Module):
  def __init__(self, teachers: list[MLPModel], group_size: int) -> None:
    super().__init__()
    self.teachers = nn.ModuleList(teachers)
    self.group_size = group_size

  def forward(self, obs: TensorDict) -> torch.Tensor:
    return torch.cat(
      [
        teacher(obs[index * self.group_size : (index + 1) * self.group_size])
        for index, teacher in enumerate(self.teachers)
      ]
    )

  def reset(self, dones: torch.Tensor) -> None:
    for teacher, group_dones in zip(
      self.teachers, dones.split(self.group_size), strict=True
    ):
      cast(MLPModel, teacher).reset(group_dones)
