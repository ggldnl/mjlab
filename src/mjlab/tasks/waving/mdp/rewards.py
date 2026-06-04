"""Reward terms for the waving (greeting) task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.waving.mdp.observations import wave_phase_value
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class wave_arm:
  """Reward tracking the waving arm toward a phase-driven greeting motion.

  The target for each waving joint is its default standing pose plus a fixed
  ``center`` offset (raising the arm into a greeting posture) plus an
  ``amplitude * sin(phase)`` term that produces the back-and-forth wave. Joint
  targets are resolved against the default pose once at construction.

  The per-step value is the negative mean squared tracking error. Unlike an
  ``exp(-error)`` kernel, this does not saturate to zero when the arm starts far
  from a raised target, so there is always a gradient pulling the arm up toward
  the wave - which is what lets the policy learn a large motion from the default
  standing pose.

  Implemented as a class (like :class:`mjlab.envs.mdp.rewards.posture`) so the
  per-joint ``center`` and ``amplitude`` dicts can be resolved into tensors once.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    joint_ids, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    self.joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)

    _, _, center = resolve_matching_names_values(
      data=cfg.params["center"], list_of_strings=joint_names
    )
    _, _, amplitude = resolve_matching_names_values(
      data=cfg.params["amplitude"], list_of_strings=joint_names
    )
    self.center = torch.tensor(center, device=env.device, dtype=torch.float32)
    self.amplitude = torch.tensor(amplitude, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    frequency: float,
    center: dict[str, float],
    amplitude: dict[str, float],
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    del center, amplitude  # Resolved into tensors at construction.
    asset: Entity = env.scene[asset_cfg.name]
    phase = wave_phase_value(env, frequency)  # (num_envs,)
    osc = torch.sin(phase).unsqueeze(-1)  # (num_envs, 1)
    target = (
      self.default_joint_pos[:, self.joint_ids] + self.center + self.amplitude * osc
    )
    current = asset.data.joint_pos[:, self.joint_ids]
    return -torch.mean(torch.square(current - target), dim=1)
