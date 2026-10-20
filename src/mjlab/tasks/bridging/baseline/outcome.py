"""Binary skill outcomes used to label proximity training examples."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.bridging.config.g1.skills.kick import mdp as kick_mdp
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.tasks.bridging.baseline.scene import TransitionScene

KICK_MIN_SPEED_M_S = 0.5


class SkillOutcome:
  def __init__(self, scene: TransitionScene) -> None:
    self.scene = scene
    self.touched = torch.zeros_like(scene.failed)
    self.early_contact = torch.zeros_like(scene.failed)
    self.peak_speed = torch.zeros_like(scene.failed, dtype=torch.float32)
    self.direction = None
    if scene.target_skill == "kick":
      local = scene.run.motion.kick_direction
      self.direction = quat_apply(
        scene.run.motion.anchor_yaw_quat,
        torch.cat((local, torch.zeros_like(local[:, :1])), -1),
      )
      self.early_contact |= kick_mdp.touching_ball(scene.env)

  def update(self, active: torch.Tensor, incoming: torch.Tensor) -> None:
    if self.direction is None:
      return
    contact = kick_mdp.touching_ball(self.scene.env)
    self.early_contact |= contact & active
    self.touched |= contact & incoming
    speed = (kick_mdp.ball_vel_w(self.scene.env) * self.direction).sum(-1)
    eligible = incoming & self.touched
    self.peak_speed = torch.where(
      eligible, torch.maximum(self.peak_speed, speed), self.peak_speed
    )

  def success(self) -> torch.Tensor:
    success = ~self.scene.failed
    if self.direction is not None:
      success = (
        success
        & self.touched
        & ~self.early_contact
        & torch.isfinite(self.peak_speed)
        & (self.peak_speed > KICK_MIN_SPEED_M_S)
      )
    return success
