"""Generate an exact-boundary kinematic plan between dynamic states."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from mjlab.tasks.bridging.bridges.diffusion.config import planner_experiment
from mjlab.tasks.bridging.bridges.diffusion.dataset.motions import (
  Layout,
  Normalizer,
  bridge_mask,
  decode,
  encode,
  integrate,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.model import (
  Denoiser,
  ModelCfg,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.process import (
  Diffusion,
  ProcessCfg,
  RobotFootKinematics,
  pose_states,
)

EXPERIMENT = planner_experiment("g1")
CHECKPOINT_FORMAT = "mjlab-kinematic-diffusion-v6"


@dataclass
class GeneratedPath:
  states: torch.Tensor
  """Kinematic reference with exact dynamic boundaries."""

  duration: torch.Tensor
  """Control ticks from A to B."""


class DiffusionBridge:
  """Inpaint root pose and joint angles between exact A and B boundaries."""

  def __init__(
    self,
    process: Diffusion,
    normalizer: Normalizer,
    layout: Layout,
    history: int,
    future: int,
    fps: float,
    min_steps: int = 3,
    robot: str = "g1",
    floor_height: float | None = 0.0,
  ):
    self.process = process.eval()
    self.normalizer = normalizer
    self.layout = layout
    self.history = history
    self.future = future
    self.fps = fps
    self.min_steps = min_steps
    self.robot = robot
    self.floor_height = floor_height
    """Lowest allowed sole height between A and B, None turns the floor off"""
    self._feet: RobotFootKinematics | None = None

  @property
  def max_steps(self) -> int:
    return self.process.denoiser.columns - self.history - self.future + 1

  @classmethod
  def load(
    cls, checkpoint: Path, device: str = "cpu", sample_steps: int | None = None
  ) -> DiffusionBridge:
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    saved = saved.get("planner", saved)
    if saved.get("format") != CHECKPOINT_FORMAT:
      raise ValueError(
        f"{checkpoint} is not a kinematic diffusion checkpoint; retrain it"
      )
    layout = Layout(**saved["layout"])
    history = int(saved["history"])
    future = int(saved["future"])
    cfg = ProcessCfg(**saved["process_cfg"])
    if sample_steps is not None:
      cfg = ProcessCfg(
        steps=cfg.steps,
        sample_steps=sample_steps,
        continuity_weight=cfg.continuity_weight,
      )
    columns = history + int(saved["max_steps"]) + future - 1
    model = Denoiser(layout.width, columns, ModelCfg(**saved["model_cfg"]))
    process = Diffusion(model, cfg, layout).to(device)
    process.denoiser.load_state_dict(saved["ema"])
    process.requires_grad_(False)
    norm = saved["normalizer"]
    normalizer = Normalizer(norm["mean"].to(device), norm["std"].to(device))
    return cls(
      process,
      normalizer,
      layout,
      history,
      future,
      float(saved["fps"]),
      int(saved["min_steps"]),
      str(saved.get("robot", "g1")),
    )

  @torch.no_grad()
  def generate(
    self,
    history: torch.Tensor,
    target: torch.Tensor,
    duration: torch.Tensor,
    trace: list[GeneratedPath] | None = None,
  ) -> GeneratedPath:
    """Fill the gap between real pre-A and post-B state sequences."""
    batch = history.shape[0]
    state_width = 13 + 2 * self.layout.joints
    if history.shape != (batch, self.history, state_width):
      raise ValueError("history has the wrong shape")
    if target.shape != (batch, self.future, state_width):
      raise ValueError(f"target must contain {self.future} consecutive states")
    if duration.shape != (batch,) or duration.dtype not in (torch.int32, torch.int64):
      raise ValueError("duration must contain one integer tick count per path")
    if history.device != target.device or history.device != duration.device:
      raise ValueError("history, target and duration must share a device")
    if not bool(torch.isfinite(history).all() and torch.isfinite(target).all()):
      raise ValueError("boundary states must be finite")
    if bool(((duration < self.min_steps) | (duration > self.max_steps)).any()):
      raise ValueError("duration lies outside the checkpoint's trained range")

    anchor = history[:, -1]
    rungs: list[torch.Tensor] | None = [] if trace is not None else None
    denoised = self.denoise(history, target, duration, rungs)
    if trace is not None and rungs is not None:
      trace.extend(self.assemble(rung, anchor, target, duration) for rung in rungs)
    return self.assemble(denoised, anchor, target, duration)

  def denoise(
    self,
    history: torch.Tensor,
    target: torch.Tensor,
    duration: torch.Tensor,
    rungs: list[torch.Tensor] | None = None,
  ) -> torch.Tensor:
    """Normalized denoised features, before the steps are added up into a path."""
    batch = history.shape[0]
    columns = self.process.denoiser.columns
    mask = bridge_mask(batch, columns, self.layout, self.history, self.future, duration)
    anchor = history[:, -1]
    values = history.new_zeros((batch, columns, self.layout.width))
    values[:, : self.history] = encode(history, anchor)
    rows = self.history - 1 + duration
    offsets = torch.arange(self.future, device=history.device)
    indexes = torch.arange(batch, device=history.device)[:, None]
    values[indexes, rows[:, None] + offsets] = encode(target, anchor)
    normalized = self.normalizer.normalize(values)
    return self.process.sample(normalized, mask, rungs, self.floor(duration))

  def floor(
    self, duration: torch.Tensor
  ) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Projection that keeps both soles above the floor between A and B.

    Lifts the root on frames where the shipped path puts a sole under the floor.
    The lift goes into the root height steps and sums to zero over the bridge, so
    the gap left at B, and the raw path seen by viewers, keep their meaning.
    """
    if self.floor_height is None:
      return None
    floor_height = self.floor_height
    if self._feet is None:
      self._feet = RobotFootKinematics(self.robot).to(duration.device)
    feet = self._feet
    anchor = self.history - 1
    height = self.layout.root_step.start + 2

    def project(normalized: torch.Tensor) -> torch.Tensor:
      features = self.normalizer.denormalize(normalized)
      pose = integrate(features, self.layout, self.history, duration)
      # pose height is world height, A's heading frame only turns and shifts in xy
      soles = feet(pose_states(pose))[..., 2].amin(-1)
      time = torch.arange(pose.shape[1], device=pose.device)[None]
      inside = (time > anchor) & (time < anchor + duration[:, None])
      lift = (floor_height - soles).clamp_min(0.0) * inside
      features[..., height] += torch.diff(lift, dim=1, prepend=lift[:, :1])
      return self.normalizer.normalize(features)

    return project

  def assemble(
    self,
    denoised: torch.Tensor,
    anchor: torch.Tensor,
    target: torch.Tensor,
    duration: torch.Tensor,
  ) -> GeneratedPath:
    """Add up the steps, decode states and copy A and B exactly."""
    features = self.normalizer.denormalize(denoised)
    pose = integrate(features, self.layout, self.history, duration)
    states = decode(pose, anchor, self.fps)
    states = states[:, self.history - 1 :]
    states[:, 0] = anchor
    offsets = torch.arange(self.future, device=states.device)
    indexes = torch.arange(states.shape[0], device=states.device)[:, None]
    states[indexes, duration[:, None] + offsets] = target
    last = duration + self.future - 1
    time = torch.arange(states.shape[1], device=states.device)[None]
    states = torch.where((time > last[:, None])[..., None], target[:, -1:, :], states)
    return GeneratedPath(states, duration)


def checkpoint_metadata(
  model_cfg: ModelCfg,
  process_cfg: ProcessCfg,
  layout: Layout,
  normalizer: Normalizer,
  history: int,
  future: int,
  min_steps: int,
  max_steps: int,
  fps: float,
  ema: dict[str, torch.Tensor],
  iteration: int,
  robot: str = "g1",
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
  }
