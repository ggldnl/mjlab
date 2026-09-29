"""Masked DDPM training and DDIM trajectory inpainting."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import torch
from torch import nn

from mjlab.tasks.bridging.bridges.diffusion.dataset.motions import (
  Layout,
  Normalizer,
  quat_from_rot6d,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.model import (
  Denoiser,
)
from mjlab.tasks.bridging.config import get_robot
from mjlab.utils.lab_api.math import quat_apply, quat_from_angle_axis, quat_mul


@dataclass(frozen=True)
class ProcessCfg:
  steps: int = 100
  sample_steps: int = 50
  continuity_weight: float = 0.5


class RobotFootKinematics(nn.Module):
  """Differentiable forward kinematics for a robot's two sole sites."""

  body_pos: torch.Tensor
  body_quat: torch.Tensor
  joint_pos: torch.Tensor
  joint_axis: torch.Tensor
  joint_index: torch.Tensor
  joint_ref: torch.Tensor
  site_pos: torch.Tensor

  def __init__(self, robot: str = "g1") -> None:
    super().__init__()
    selected = get_robot(robot)
    model = selected.get_spec().compile()
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, selected.base_body_name)
    chains: list[list[int]] = []
    sites: list[list[float]] = []
    for side in ("left", "right"):
      site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot")
      body = int(model.site_bodyid[site])
      chain = []
      while body != pelvis:
        chain.append(body)
        body = int(model.body_parentid[body])
      chains.append(chain[::-1])
      sites.append(model.site_pos[site].tolist())
    if not chains[0] or len(chains[0]) != len(chains[1]):
      raise ValueError("Robot feet must have equal nonempty kinematic chains")

    body_pos = []
    body_quat = []
    joint_pos = []
    joint_axis = []
    joint_index = []
    joint_ref = []
    for chain in chains:
      positions = []
      quaternions = []
      anchors = []
      axes = []
      indexes = []
      references = []
      for body in chain:
        if model.body_jntnum[body] != 1:
          raise ValueError("Robot leg links must each have one joint")
        joint = int(model.body_jntadr[body])
        address = int(model.jnt_qposadr[joint])
        positions.append(model.body_pos[body].tolist())
        quaternions.append(model.body_quat[body].tolist())
        anchors.append(model.jnt_pos[joint].tolist())
        axes.append(model.jnt_axis[joint].tolist())
        indexes.append(address - 7)
        references.append(float(model.qpos0[address]))
      body_pos.append(positions)
      body_quat.append(quaternions)
      joint_pos.append(anchors)
      joint_axis.append(axes)
      joint_index.append(indexes)
      joint_ref.append(references)
    self.register_buffer("body_pos", torch.tensor(body_pos, dtype=torch.float32))
    self.register_buffer("body_quat", torch.tensor(body_quat, dtype=torch.float32))
    self.register_buffer("joint_pos", torch.tensor(joint_pos, dtype=torch.float32))
    self.register_buffer("joint_axis", torch.tensor(joint_axis, dtype=torch.float32))
    self.register_buffer("joint_index", torch.tensor(joint_index, dtype=torch.long))
    self.register_buffer("joint_ref", torch.tensor(joint_ref, dtype=torch.float32))
    self.register_buffer("site_pos", torch.tensor(sites, dtype=torch.float32))
    self.joints = model.nq - 7
    self.steps = len(chains[0])

  @staticmethod
  def _expand(value: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    return value.view(*([1] * (states.ndim - 1)), *value.shape).expand(
      *states.shape[:-1], *value.shape
    )

  def forward(self, states: torch.Tensor) -> torch.Tensor:
    if states.shape[-1] != 13 + 2 * self.joints:
      raise ValueError("State width does not match the selected robot")
    position = states[..., None, :3].expand(*states.shape[:-1], 2, 3)
    orientation = states[..., None, 3:7].expand(*states.shape[:-1], 2, 4)
    for step in range(self.steps):
      offset = self._expand(self.body_pos[:, step], states)
      position = position + quat_apply(orientation, offset)
      link_quat = self._expand(self.body_quat[:, step], states)
      orientation = quat_mul(orientation, link_quat)
      anchor = self._expand(self.joint_pos[:, step], states)
      axis = self._expand(self.joint_axis[:, step], states)
      indexes = self.joint_index[:, step]
      reference = self._expand(self.joint_ref[:, step], states)
      angle = states[..., 13 + indexes] - reference
      joint_quat = quat_from_angle_axis(angle, axis)
      rotated = quat_mul(orientation, joint_quat)
      position = (
        position + quat_apply(orientation, anchor) - quat_apply(rotated, anchor)
      )
      orientation = rotated
    return position + quat_apply(orientation, self._expand(self.site_pos, states))


class RobotFootSlipLoss(nn.Module):
  """Penalize horizontal sole velocity on demonstrated contact edges."""

  mean: torch.Tensor
  std: torch.Tensor

  def __init__(
    self,
    robot: str,
    normalizer: Normalizer,
    layout: Layout,
    fps: float,
    weight: float,
    contact_height: float,
    contact_speed: float,
  ) -> None:
    super().__init__()
    if min(fps, contact_height, contact_speed) <= 0 or weight < 0:
      raise ValueError("invalid foot slip loss configuration")
    self.layout = layout
    self.fps = fps
    self.weight = weight
    self.contact_height = contact_height
    self.contact_speed = contact_speed
    self.register_buffer("mean", normalizer.mean)
    self.register_buffer("std", normalizer.std)
    self.feet = RobotFootKinematics(robot)
    if layout.joints != self.feet.joints:
      raise ValueError("Layout and robot joint counts differ")

  def _states(self, normalized: torch.Tensor) -> torch.Tensor:
    features = normalized * self.std + self.mean
    return torch.cat(
      (
        features[..., :3],
        quat_from_rot6d(features[..., 3:9]),
        features[..., 9:15],
        features[..., self.layout.joint_positions],
        features[..., self.layout.joint_velocities],
      ),
      dim=-1,
    )

  def forward(
    self,
    predicted: torch.Tensor,
    clean: torch.Tensor,
    active_edges: torch.Tensor,
  ) -> torch.Tensor:
    clean_feet = self.feet(self._states(clean))
    predicted_feet = self.feet(self._states(predicted))
    clean_delta = clean_feet[:, 1:, :, :2] - clean_feet[:, :-1, :, :2]
    clean_speed = torch.linalg.vector_norm(clean_delta, dim=-1) * self.fps
    contact = (
      (clean_feet[:, 1:, :, 2] <= self.contact_height)
      & (clean_feet[:, :-1, :, 2] <= self.contact_height)
      & (clean_speed <= self.contact_speed)
      & active_edges.any(-1)[..., None]
    )
    slip = (predicted_feet[:, 1:, :, :2] - predicted_feet[:, :-1, :, :2]) * self.fps
    if not bool(contact.any()):
      return slip.sum() * 0.0
    return self.weight * slip.square().sum(-1)[contact].mean()


def cosine_schedule(steps: int) -> torch.Tensor:
  if steps < 2:
    raise ValueError("steps must be at least two")
  ticks = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
  curve = torch.cos((ticks + 0.008) / 1.008 * math.pi / 2).square()
  return (curve[1:] / curve[0]).clamp(1e-5, 0.9999).float()


class Diffusion(nn.Module):
  def __init__(self, denoiser: Denoiser, cfg: ProcessCfg):
    super().__init__()
    if not 1 <= cfg.sample_steps <= cfg.steps:
      raise ValueError("sample_steps must be between one and steps")
    if cfg.continuity_weight < 0:
      raise ValueError("continuity_weight cannot be negative")
    self.denoiser = denoiser
    self.cfg = cfg
    self.register_buffer("alphas", cosine_schedule(cfg.steps))

  @property
  def schedule(self) -> torch.Tensor:
    assert isinstance(self.alphas, torch.Tensor)
    return self.alphas

  def loss(
    self,
    clean: torch.Tensor,
    known: torch.Tensor,
    valid: torch.Tensor | None = None,
    foot_slip: RobotFootSlipLoss | None = None,
  ) -> torch.Tensor:
    if clean.shape != known.shape:
      raise ValueError("clean window and known mask must match")
    if valid is None:
      valid = torch.ones_like(known)
    elif valid.shape == clean.shape[:2]:
      valid = valid[..., None].expand_as(clean)
    elif valid.shape != clean.shape:
      raise ValueError("valid mask must cover time or every feature")
    step = torch.randint(self.cfg.steps, (clean.shape[0],), device=clean.device)
    alpha = self.schedule[step, None, None]
    noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * torch.randn_like(clean)
    noisy = torch.where(known, clean, noisy)
    predicted = self.denoiser(noisy, known, step)
    unknown = ~known & valid
    reconstruction = (predicted - clean).square()[unknown].mean()
    full = torch.where(known, clean, predicted)
    target_delta = clean[:, 1:] - clean[:, :-1]
    predicted_delta = full[:, 1:] - full[:, :-1]
    active_edges = (unknown[:, 1:] | unknown[:, :-1]) & (valid[:, 1:] & valid[:, :-1])
    continuity = (predicted_delta - target_delta).square()[active_edges].mean()
    total = reconstruction + self.cfg.continuity_weight * continuity
    if foot_slip is not None:
      total = total + foot_slip(full, clean, active_edges)
    return total

  @torch.no_grad()
  def sample(
    self,
    known_values: torch.Tensor,
    known: torch.Tensor,
    trace: list[torch.Tensor] | None = None,
  ) -> torch.Tensor:
    """Fill free channels; copy all conditioned channels at every denoising step.

    A trace list collects one clean prediction per rung of the ladder, for viewers.
    Its last entry is what is returned.
    """
    if known_values.shape != known.shape:
      raise ValueError("known values and mask must match")
    ladder = (
      torch.linspace(
        self.cfg.steps - 1, 0, self.cfg.sample_steps, device=known_values.device
      )
      .round()
      .long()
      .unique_consecutive()
    )
    noisy = torch.where(known, known_values, torch.randn_like(known_values))
    for index, tick in enumerate(ladder):
      step = tick.expand(noisy.shape[0])
      clean = self.denoiser(noisy, known, step)
      clean = torch.where(known, known_values, clean)
      if trace is not None:
        trace.append(clean)
      if index == len(ladder) - 1:
        return clean
      alpha = self.schedule[tick]
      next_alpha = self.schedule[ladder[index + 1]]
      noise = (noisy - alpha.sqrt() * clean) / (1 - alpha).sqrt()
      noisy = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * noise
      noisy = torch.where(known, known_values, noisy)
    raise RuntimeError("empty denoising schedule")


class G1FootKinematics(RobotFootKinematics):
  """Backward-compatible G1 foot kinematics."""

  def __init__(self) -> None:
    super().__init__("g1")


class G1FootSlipLoss(RobotFootSlipLoss):
  """Backward-compatible G1 foot slip loss."""

  def __init__(
    self,
    normalizer: Normalizer,
    layout: Layout,
    fps: float,
    weight: float,
    contact_height: float,
    contact_speed: float,
  ) -> None:
    super().__init__(
      "g1",
      normalizer,
      layout,
      fps,
      weight,
      contact_height,
      contact_speed,
    )
