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
  integrate,
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


class PathLoss(nn.Module):
  """Losses on the path rebuilt from predicted steps.

  endpoint   gap between the summed steps and B, in typical steps times sqrt(duration)
  foot slip  horizontal sole speed on edges where the demonstration has a planted foot
  """

  mean: torch.Tensor
  std: torch.Tensor
  step_std: torch.Tensor

  def __init__(
    self,
    robot: str,
    normalizer: Normalizer,
    layout: Layout,
    history: int,
    fps: float,
    endpoint_weight: float,
    foot_slip_weight: float,
    contact_height: float,
    contact_speed: float,
  ) -> None:
    super().__init__()
    if min(fps, contact_height, contact_speed) <= 0 or history < 1:
      raise ValueError("invalid path loss configuration")
    if min(endpoint_weight, foot_slip_weight) < 0:
      raise ValueError("path loss weights cannot be negative")
    self.layout = layout
    self.history = history
    self.fps = fps
    self.endpoint_weight = endpoint_weight
    self.foot_slip_weight = foot_slip_weight
    self.contact_height = contact_height
    self.contact_speed = contact_speed
    self.register_buffer("mean", normalizer.mean)
    self.register_buffer("std", normalizer.std)
    self.register_buffer(
      "step_std",
      torch.cat((normalizer.std[layout.root_step], normalizer.std[layout.joint_steps])),
    )
    self.feet = RobotFootKinematics(robot)
    if layout.joints != self.feet.joints:
      raise ValueError("Layout and robot joint counts differ")

  def _endpoint(self, features: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
    layout = self.layout
    anchor = self.history - 1
    batch = torch.arange(features.shape[0], device=features.device)
    steps = torch.cat(
      (features[..., layout.root_step], features[..., layout.joint_steps]), dim=-1
    )
    known = torch.cat(
      (features[..., layout.root_position], features[..., layout.joint_positions]),
      dim=-1,
    )
    time = torch.arange(features.shape[1], device=features.device)[None]
    inside = (time > anchor) & (time <= anchor + duration[:, None])
    travelled = (steps * inside[..., None]).sum(dim=1)
    gap = known[batch, anchor + duration] - known[:, anchor] - travelled
    scale = self.step_std * duration.float().sqrt()[:, None]
    return (gap / scale).square().mean()

  def _states(self, pose: torch.Tensor) -> torch.Tensor:
    return torch.cat(
      (
        pose[..., :3],
        quat_from_rot6d(pose[..., 3:9]),
        pose.new_zeros((*pose.shape[:-1], 6)),
        pose[..., 9:],
        pose.new_zeros((*pose.shape[:-1], self.layout.joints)),
      ),
      dim=-1,
    )

  def _foot_slip(
    self,
    predicted: torch.Tensor,
    clean: torch.Tensor,
    active_edges: torch.Tensor,
    duration: torch.Tensor,
  ) -> torch.Tensor:
    clean_pose = integrate(clean, self.layout, self.history, duration)
    predicted_pose = integrate(predicted, self.layout, self.history, duration)
    clean_feet = self.feet(self._states(clean_pose))
    predicted_feet = self.feet(self._states(predicted_pose))
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
    return slip.square().sum(-1)[contact].mean()

  def forward(
    self,
    predicted: torch.Tensor,
    clean: torch.Tensor,
    active_edges: torch.Tensor,
    duration: torch.Tensor,
  ) -> torch.Tensor:
    predicted = predicted * self.std + self.mean
    clean = clean * self.std + self.mean
    total = self.endpoint_weight * self._endpoint(predicted, duration)
    if self.foot_slip_weight:
      total = total + self.foot_slip_weight * self._foot_slip(
        predicted, clean, active_edges, duration
      )
    return total


def cosine_schedule(steps: int) -> torch.Tensor:
  if steps < 2:
    raise ValueError("steps must be at least two")
  ticks = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
  curve = torch.cos((ticks + 0.008) / 1.008 * math.pi / 2).square()
  return (curve[1:] / curve[0]).clamp(1e-5, 0.9999).float()


class Diffusion(nn.Module):
  """Masked diffusion over the predicted channels of a Layout.

  Condition channels are never noised or predicted. They hold their value where
  known and zero elsewhere.
  """

  predicted: torch.Tensor

  def __init__(self, denoiser: Denoiser, cfg: ProcessCfg, layout: Layout):
    super().__init__()
    if not 1 <= cfg.sample_steps <= cfg.steps:
      raise ValueError("sample_steps must be between one and steps")
    if cfg.continuity_weight < 0:
      raise ValueError("continuity_weight cannot be negative")
    if denoiser.features != layout.width:
      raise ValueError("denoiser width does not match the layout")
    self.denoiser = denoiser
    self.cfg = cfg
    self.register_buffer("alphas", cosine_schedule(cfg.steps))
    predicted = torch.zeros(layout.width, dtype=torch.bool)
    predicted[layout.predicted] = True
    self.register_buffer("predicted", predicted)

  @property
  def schedule(self) -> torch.Tensor:
    assert isinstance(self.alphas, torch.Tensor)
    return self.alphas

  def _inputs(
    self, free: torch.Tensor, values: torch.Tensor, known: torch.Tensor
  ) -> torch.Tensor:
    """Known values where known, free values on predicted channels, zero elsewhere."""
    return torch.where(
      known, values, torch.where(self.predicted, free, torch.zeros_like(free))
    )

  def loss(
    self,
    clean: torch.Tensor,
    known: torch.Tensor,
    valid: torch.Tensor | None = None,
    path_loss: PathLoss | None = None,
    duration: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if clean.shape != known.shape:
      raise ValueError("clean window and known mask must match")
    if path_loss is not None and duration is None:
      raise ValueError("path_loss needs the A to B durations")
    if valid is None:
      valid = torch.ones_like(known)
    elif valid.shape == clean.shape[:2]:
      valid = valid[..., None].expand_as(clean)
    elif valid.shape != clean.shape:
      raise ValueError("valid mask must cover time or every feature")
    step = torch.randint(self.cfg.steps, (clean.shape[0],), device=clean.device)
    alpha = self.schedule[step, None, None]
    noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * torch.randn_like(clean)
    predicted = self.denoiser(self._inputs(noisy, clean, known), known, step)
    unknown = ~known & valid & self.predicted
    reconstruction = (predicted - clean).square()[unknown].mean()
    full = torch.where(known | ~self.predicted, clean, predicted)
    target_delta = clean[:, 1:] - clean[:, :-1]
    predicted_delta = full[:, 1:] - full[:, :-1]
    active_edges = (unknown[:, 1:] | unknown[:, :-1]) & (valid[:, 1:] & valid[:, :-1])
    continuity = (predicted_delta - target_delta).square()[active_edges].mean()
    total = reconstruction + self.cfg.continuity_weight * continuity
    if path_loss is not None:
      assert duration is not None
      total = total + path_loss(full, clean, active_edges, duration)
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
    noisy = self._inputs(torch.randn_like(known_values), known_values, known)
    for index, tick in enumerate(ladder):
      step = tick.expand(noisy.shape[0])
      clean = self._inputs(self.denoiser(noisy, known, step), known_values, known)
      if trace is not None:
        trace.append(clean)
      if index == len(ladder) - 1:
        return clean
      alpha = self.schedule[tick]
      next_alpha = self.schedule[ladder[index + 1]]
      noise = (noisy - alpha.sqrt() * clean) / (1 - alpha).sqrt()
      noisy = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * noise
      noisy = self._inputs(noisy, known_values, known)
    raise RuntimeError("empty denoising schedule")


class G1FootKinematics(RobotFootKinematics):
  """Backward-compatible G1 foot kinematics."""

  def __init__(self) -> None:
    super().__init__("g1")
