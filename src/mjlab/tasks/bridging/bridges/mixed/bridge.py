"""Diffusion bridge conditioned on footsteps.

Same planner as bridges.diffusion: it inpaints root and joint steps between an exact
A history and an exact B continuation. The window also carries footstep channels,
see footsteps.py. At inference they come from the heuristic footstep planner.

The model only learns to respect footsteps, so every denoising rung is followed by
two projections on the integrated pose:

    floor       lift the root where a sole goes under the floor
    plant       a few damped least squares steps on the leg joints pulling each
                planted sole onto its footstep, carried smoothly through swings

Both write their change back into the steps and vanish at A and B, so the
boundaries stay exact and later rungs adapt to the correction.
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from mjlab.tasks.bridging.bridges.diffusion.dataset.motions import (
  Normalizer,
  encode,
  encode_pose,
  integrate,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.bridge import (
  DiffusionBridge,
  GeneratedPath,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.model import Denoiser, ModelCfg
from mjlab.tasks.bridging.bridges.diffusion.planner.process import (
  Diffusion,
  PathLoss,
  ProcessCfg,
  RobotFootKinematics,
  pose_states,
)
from mjlab.tasks.bridging.bridges.mixed.footsteps import (
  ContactCfg,
  FootstepLayout,
  FootstepPlanner,
  Gait,
  deployment_mask,
  extract,
  foot_values,
  sole_frames,
)
from mjlab.tasks.bridging.config import get_robot

CHECKPOINT_FORMAT = "mjlab-footstep-diffusion-v1"


def planner_experiment(robot: str) -> str:
  get_robot(robot)
  return f"{robot}_footstep_diffusion_planner"


@dataclass(frozen=True)
class PlantCfg:
  iterations: int = 3
  damping: float = 1e-3
  max_joint_step: float = 0.2
  smoothing: int = 3
  """Half width in frames of the triangular kernel smoothing the correction."""


@dataclass
class FootstepPlan:
  contact: torch.Tensor
  """(B, columns, 2) planted flags."""

  channels: torch.Tensor
  """(B, columns, 2, 6) footstep channels."""

  known: torch.Tensor
  """(B, columns, 2) frames whose footstep channels are given to the model."""


class PlantLoss(PathLoss):
  """PathLoss plus the distance between predicted soles and the given footsteps.

  Applies between A and B on frames whose footstep is known and planted. Position
  error is measured in units of scale, yaw error as 1 minus the cosine.
  """

  def __init__(
    self,
    robot: str,
    normalizer: Normalizer,
    layout: FootstepLayout,
    history: int,
    fps: float,
    endpoint_weight: float,
    foot_slip_weight: float,
    contact_height: float,
    contact_speed: float,
    plant_weight: float,
    scale: float = 0.05,
  ) -> None:
    super().__init__(
      robot,
      normalizer,
      layout,
      history,
      fps,
      endpoint_weight,
      foot_slip_weight,
      contact_height,
      contact_speed,
    )
    if plant_weight < 0 or scale <= 0:
      raise ValueError("plant weight cannot be negative and scale must be positive")
    self.footstep_layout = layout
    self.plant_weight = plant_weight
    self.scale = scale

  def forward(
    self,
    predicted: torch.Tensor,
    clean: torch.Tensor,
    active_edges: torch.Tensor,
    duration: torch.Tensor,
    known: torch.Tensor | None = None,
  ) -> torch.Tensor:
    total = super().forward(predicted, clean, active_edges, duration)
    if not self.plant_weight or known is None:
      return total
    layout = self.footstep_layout
    features = predicted * self.std + self.mean
    pose = integrate(features, layout, self.history, duration)
    position, yaw = sole_frames(self.feet, pose)
    target = foot_values(layout, features)
    place_known = known[..., [channel + 1 for channel in layout.contact_channels]]
    time = torch.arange(pose.shape[1], device=pose.device)[None]
    anchor = self.history - 1
    inside = (time > anchor) & (time < anchor + duration[:, None])
    active = place_known & (target[..., 0] > 0.5) & inside[..., None]
    if not bool(active.any()):
      return total + position.sum() * 0.0
    distance = (position - target[..., 1:4]).square().sum(-1) / self.scale**2
    heading = 1 - (yaw.cos() * target[..., 4] + yaw.sin() * target[..., 5])
    return total + self.plant_weight * (distance + heading)[active].mean()


def fill_gaps(values: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
  """Linear interpolation in time of values (B, T, C) between frames where fixed (B, T).

  The first and last frames must be fixed.
  """
  frames = values.shape[1]
  time = torch.arange(frames, device=values.device)[None].expand_as(fixed)
  before = torch.where(fixed, time, -1).cummax(1).values
  after = torch.where(fixed, time, frames).flip(1).cummin(1).values.flip(1)
  span = (after - before).clamp_min(1)
  weight = ((time - before) / span).float()[..., None]
  low = values.gather(1, before[..., None].expand_as(values))
  high = values.gather(1, after[..., None].expand_as(values))
  return low + weight * (high - low)


def smooth(values: torch.Tensor, half_width: int) -> torch.Tensor:
  """Triangular kernel smoothing of values (B, T, C) along time, edges replicated."""
  if half_width < 1:
    return values
  kernel = torch.cat(
    (
      torch.arange(1, half_width + 2, device=values.device),
      torch.arange(half_width, 0, -1, device=values.device),
    )
  ).float()
  kernel = (kernel / kernel.sum()).view(1, 1, -1)
  channels = values.shape[-1]
  flat = values.transpose(1, 2).reshape(-1, 1, values.shape[1])
  flat = torch.nn.functional.pad(flat, (half_width, half_width), mode="replicate")
  out = torch.nn.functional.conv1d(flat, kernel)
  return out.reshape(values.shape[0], channels, -1).transpose(1, 2)


def plant_projection(
  feet: RobotFootKinematics,
  layout: FootstepLayout,
  normalizer: Normalizer,
  history: int,
  duration: torch.Tensor,
  plan: FootstepPlan,
  cfg: PlantCfg,
) -> Callable[[torch.Tensor], torch.Tensor]:
  """Projection pulling planted soles onto their footsteps with leg IK.

  The Jacobian of each sole against its own leg joints comes from finite
  differences, one forward kinematics pass per leg joint for both legs at once.

  Correcting planted frames alone makes the correction a step function in time,
  which showed up as joint accelerations six times the recorded ones. So the
  correction is interpolated through swing frames, smoothed with a short
  triangular kernel and tapered to zero at A and B.
  """
  anchor = history - 1
  time = torch.arange(plan.contact.shape[1], device=duration.device)[None]
  inside = (time > anchor) & (time < anchor + duration[:, None])
  planted = plan.contact & plan.known & inside[..., None]
  active = planted.float()[..., None]
  fixed = planted | ~inside[..., None]
  target = plan.channels[..., 1:4]
  legs = feet.joint_index
  epsilon = 1e-3

  def soles(pose: torch.Tensor) -> torch.Tensor:
    return feet(pose_states(pose))

  def project(normalized: torch.Tensor) -> torch.Tensor:
    if not bool(active.any()):
      return normalized
    features = normalizer.denormalize(normalized)
    pose = integrate(features, layout, history, duration)
    delta = torch.zeros_like(pose[..., 9:])
    for _ in range(cfg.iterations):
      current = pose.clone()
      current[..., 9:] += delta
      here = soles(current)
      error = (target - here) * active
      columns = []
      for step in range(legs.shape[1]):
        moved = current.clone()
        moved[..., 9 + legs[:, step]] += epsilon
        columns.append((soles(moved) - here) / epsilon)
      jacobian = torch.stack(columns, dim=-1)
      gram = jacobian @ jacobian.transpose(-1, -2)
      gram = gram + cfg.damping * torch.eye(3, device=gram.device)
      solved = torch.linalg.solve(gram, error[..., None])
      change = (jacobian.transpose(-1, -2) @ solved)[..., 0]
      change = change.clamp(-cfg.max_joint_step, cfg.max_joint_step)
      for foot in range(2):
        delta[..., legs[foot]] += change[..., foot, :]
    for foot in range(2):
      delta[..., legs[foot]] = fill_gaps(delta[..., legs[foot]], fixed[..., foot])
    delta = smooth(delta, cfg.smoothing) * inside[..., None]
    joint_steps = torch.diff(delta, dim=1, prepend=delta[:, :1])
    features[..., layout.joint_steps] += joint_steps
    return normalizer.normalize(features)

  return project


class FootstepBridge(DiffusionBridge):
  """Diffusion bridge that inpaints around footsteps from the heuristic planner."""

  layout: FootstepLayout

  def __init__(
    self,
    *args,
    gait: Gait,
    contact: ContactCfg,
    plant: PlantCfg | None = None,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.gait = gait
    self.contact = contact
    self.plant = plant or PlantCfg()
    self._plan_feet: RobotFootKinematics | None = None

  @property
  def feet(self) -> RobotFootKinematics:
    if self._plan_feet is None:
      device = self.normalizer.mean.device
      self._plan_feet = RobotFootKinematics(self.robot).to(device)
    return self._plan_feet

  @classmethod
  def load(
    cls, checkpoint: Path, device: str = "cpu", sample_steps: int | None = None
  ) -> FootstepBridge:
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved.get("format") != CHECKPOINT_FORMAT:
      raise ValueError(f"{checkpoint} is not a footstep diffusion checkpoint")
    layout = FootstepLayout(**saved["layout"])
    history = int(saved["history"])
    future = int(saved["future"])
    cfg = ProcessCfg(**saved["process_cfg"])
    if sample_steps is not None:
      cfg = ProcessCfg(cfg.steps, sample_steps, cfg.continuity_weight)
    columns = history + int(saved["max_steps"]) + future - 1
    model = Denoiser(layout.width, columns, ModelCfg(**saved["model_cfg"]))
    process = Diffusion(model, cfg, layout).to(device)
    process.denoiser.load_state_dict(saved["ema"])
    process.requires_grad_(False)
    norm = saved["normalizer"]
    return cls(
      process,
      Normalizer(norm["mean"].to(device), norm["std"].to(device)),
      layout,
      history,
      future,
      float(saved["fps"]),
      int(saved["min_steps"]),
      str(saved["robot"]),
      gait=Gait.from_dict(saved["gait"]),
      contact=ContactCfg(**saved["contact"]),
    )

  def plan(
    self, history: torch.Tensor, target: torch.Tensor, duration: torch.Tensor
  ) -> FootstepPlan:
    """Heuristic footsteps for dynamic state boundaries."""
    planner = FootstepPlanner(self.feet, self.gait, self.contact, self.fps)
    columns = self.process.denoiser.columns
    contact, channels = planner(history, target, duration, columns)
    return FootstepPlan(contact, channels, torch.ones_like(contact))

  def clip_plan(self, states: torch.Tensor) -> FootstepPlan:
    """Footsteps read from a recorded window of dynamic states, A at history - 1."""
    pose = encode_pose(states, states[:, self.history - 1])
    contact, channels, _ = extract(self.feet, pose, self.fps, self.contact)
    return FootstepPlan(contact, channels, torch.ones_like(contact))

  def plant_error(
    self, path: GeneratedPath, plan: FootstepPlan, anchor: torch.Tensor
  ) -> torch.Tensor:
    """Mean sole distance to planned footsteps between A and B, per window.

    NaN for windows without a planted frame between A and B.
    """
    frames = path.states.shape[1]
    position, _ = sole_frames(self.feet, encode_pose(path.states, anchor))
    start = self.history - 1
    target = plan.channels[:, start : start + frames, :, 1:4]
    planted = plan.contact[:, start : start + frames]
    time = torch.arange(frames, device=path.states.device)[None]
    inside = (time > 0) & (time < path.duration[:, None])
    active = (planted & inside[..., None]).float()
    distance = (position - target).norm(dim=-1) * active
    return distance.sum((1, 2)) / active.sum((1, 2))

  @torch.no_grad()
  def generate(
    self,
    history: torch.Tensor,
    target: torch.Tensor,
    duration: torch.Tensor,
    trace: list[GeneratedPath] | None = None,
    plan: FootstepPlan | None = None,
  ) -> GeneratedPath:
    """Fill the gap between pre A and post B states around footsteps.

    Without a plan the heuristic planner is used. A plan with nothing known runs
    the model without footsteps.
    """
    batch = history.shape[0]
    if duration.shape != (batch,) or bool(
      ((duration < self.min_steps) | (duration > self.max_steps)).any()
    ):
      raise ValueError("duration lies outside the checkpoint's trained range")
    if history.shape[1] != self.history or target.shape[1] != self.future:
      raise ValueError("history or target has the wrong number of states")
    anchor = history[:, -1]
    rungs: list[torch.Tensor] | None = [] if trace is not None else None
    denoised = self.denoise(history, target, duration, rungs, plan)
    if trace is not None and rungs is not None:
      trace.extend(self.assemble(rung, anchor, target, duration) for rung in rungs)
    return self.assemble(denoised, anchor, target, duration)

  def denoise(
    self,
    history: torch.Tensor,
    target: torch.Tensor,
    duration: torch.Tensor,
    rungs: list[torch.Tensor] | None = None,
    plan: FootstepPlan | None = None,
  ) -> torch.Tensor:
    """Normalized denoised features around footsteps, heuristic ones by default."""
    if plan is None:
      plan = self.plan(history, target, duration)
    layout = self.layout
    batch = history.shape[0]
    columns = self.process.denoiser.columns
    mask = deployment_mask(
      layout, columns, self.history, self.future, duration, plan.contact, plan.known
    )
    anchor = history[:, -1]
    values = history.new_zeros((batch, columns, layout.width))
    values[:, : self.history, : layout.feet.start] = encode(history, anchor)
    rows = self.history - 1 + duration
    offsets = torch.arange(self.future, device=history.device)
    indexes = torch.arange(batch, device=history.device)[:, None]
    values[indexes, rows[:, None] + offsets, : layout.feet.start] = encode(
      target, anchor
    )
    values[..., layout.feet] = plan.channels.flatten(-2)
    normalized = self.normalizer.normalize(values)
    return self.process.sample(normalized, mask, rungs, self.projection(duration, plan))

  def projection(
    self, duration: torch.Tensor, plan: FootstepPlan
  ) -> Callable[[torch.Tensor], torch.Tensor]:
    plant = plant_projection(
      self.feet,
      self.layout,
      self.normalizer,
      self.history,
      duration,
      plan,
      self.plant,
    )
    floor = self.floor(duration)
    if floor is None:
      return plant
    return lambda normalized: plant(floor(normalized))


def load_bridge(
  checkpoint: Path, device: str = "cpu", sample_steps: int | None = None
) -> DiffusionBridge:
  """Plain or footstep diffusion bridge, picked by the checkpoint's format."""
  saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
  saved = saved.get("planner", saved)
  if saved.get("format") == CHECKPOINT_FORMAT:
    return FootstepBridge.load(checkpoint, device, sample_steps)
  return DiffusionBridge.load(checkpoint, device, sample_steps)


def checkpoint_metadata(
  model_cfg: ModelCfg,
  process_cfg: ProcessCfg,
  layout: FootstepLayout,
  normalizer: Normalizer,
  history: int,
  future: int,
  min_steps: int,
  max_steps: int,
  fps: float,
  ema: dict[str, torch.Tensor],
  iteration: int,
  robot: str,
  gait: Gait,
  contact: ContactCfg,
) -> dict:
  return {
    "format": CHECKPOINT_FORMAT,
    "model_cfg": asdict(model_cfg),
    "process_cfg": asdict(process_cfg),
    "layout": asdict(layout),
    "normalizer": {"mean": normalizer.mean.cpu(), "std": normalizer.std.cpu()},
    "history": history,
    "future": future,
    "min_steps": min_steps,
    "max_steps": max_steps,
    "fps": fps,
    "ema": {name: value.cpu() for name, value in ema.items()},
    "iteration": iteration,
    "robot": robot,
    "gait": gait.to_dict(),
    "contact": asdict(contact),
  }
