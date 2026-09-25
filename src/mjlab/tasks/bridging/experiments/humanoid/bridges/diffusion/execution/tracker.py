"""UniTracker adapter for generated G1 state paths."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.envs.mdp.actions.actions import BaseAction
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.command import (
  Tolerances,
  channel_errors,
  upper_body_mask,
)
from mjlab.tasks.unitracker.checkpoint import OBSERVATION_SIZE, G1UniTrackerPolicy
from mjlab.tasks.unitracker.config.g1.env_cfgs import G1_POLICY_JOINT_NAMES
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


class UniTrackerExecutor:
  """Turn a five-frame generated reference into G1 actions."""

  horizon = 5

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    policy: G1UniTrackerPolicy | None = None,
  ) -> None:
    if env.num_envs != 1:
      raise ValueError("The released UniTracker checkpoint supports one environment")
    self.env = env
    self.robot: Entity = env.scene["robot"]
    indices, _ = self.robot.find_joints(G1_POLICY_JOINT_NAMES, preserve_order=True)
    self.joint_ids = torch.tensor(indices, device=env.device, dtype=torch.long)
    action_term = env.action_manager.get_term("joint_pos")
    if not isinstance(action_term, BaseAction):
      raise TypeError("UniTracker needs a joint action term")
    target_names = tuple(action_term.target_names)
    if set(target_names) != set(G1_POLICY_JOINT_NAMES):
      raise ValueError(
        "Environment action joints do not match the UniTracker checkpoint"
      )
    self.policy_from_action = torch.tensor(
      [target_names.index(name) for name in G1_POLICY_JOINT_NAMES],
      device=env.device,
    )
    self.action_from_policy = torch.tensor(
      [G1_POLICY_JOINT_NAMES.index(name) for name in target_names],
      device=env.device,
    )
    # The released TextOp deployment feeds body 0 (the pelvis) as its anchor, despite
    # the public training config naming torso_link.  Using the torso makes the exact
    # demonstrated trajectories diverge; pelvis reproduces the checkpoint contract.
    self.anchor_id = self.robot.body_names.index("pelvis")
    self.upper_body = upper_body_mask(tuple(self.robot.joint_names), env.device)
    self.tolerances = Tolerances().tensor(env.device)
    self.policy = policy or G1UniTrackerPolicy()

  def observation(self, reference: torch.Tensor) -> torch.Tensor:
    if reference.shape[0] != 1 or reference.shape[1] != self.horizon:
      raise ValueError("UniTracker needs one five-frame reference")
    joints = self.robot.data.joint_pos.shape[1]
    desired_pos, desired_quat = reference[..., :3], reference[..., 3:7]
    current_pos = self.robot.data.body_link_pos_w[:, self.anchor_id]
    current_quat = self.robot.data.body_link_quat_w[:, self.anchor_id]
    relative_pos, relative_quat = subtract_frame_transforms(
      current_pos[:, None].expand(-1, self.horizon, -1),
      current_quat[:, None].expand(-1, self.horizon, -1),
      desired_pos,
      desired_quat,
    )
    desired_joint_pos = reference[..., 13 : 13 + joints][..., self.joint_ids]
    desired_joint_vel = reference[..., 13 + joints :][..., self.joint_ids]
    default_pos = self.robot.data.default_joint_pos
    default_vel = self.robot.data.default_joint_vel
    assert default_pos is not None and default_vel is not None
    observation = torch.cat(
      (
        desired_joint_pos.flatten(1),
        desired_joint_vel.flatten(1),
        relative_pos.flatten(1),
        matrix_from_quat(relative_quat)[..., :2].flatten(1),
        self.robot.data.projected_gravity_b,
        self.robot.data.root_link_lin_vel_b,
        self.robot.data.root_link_ang_vel_b,
        self.robot.data.joint_pos[:, self.joint_ids] - default_pos[:, self.joint_ids],
        self.robot.data.joint_vel[:, self.joint_ids] - default_vel[:, self.joint_ids],
        self.env.action_manager.action[:, self.policy_from_action],
      ),
      dim=-1,
    )
    if observation.shape != (1, OBSERVATION_SIZE):
      raise ValueError(
        f"UniTracker observation is {observation.shape}, expected (1, {OBSERVATION_SIZE})"
      )
    return observation

  def __call__(self, history: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if history.shape[0] != reference.shape[0]:
      raise ValueError("history and reference batch sizes differ")
    observation = self.observation(reference)
    action = self.policy(
      np.ascontiguousarray(observation.detach().cpu().numpy(), dtype=np.float32)
    )
    return torch.as_tensor(action, device=reference.device)[:, self.action_from_policy]

  def captured(self, history: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return whether the simulated state is safe for policy handoff."""
    if target.shape[1] < 1:
      raise ValueError("target must contain B")
    errors = channel_errors(history[:, -1], target[:, 0], self.upper_body)
    return (errors <= self.tolerances).all(-1)
