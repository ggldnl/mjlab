"""Load a position-only diffusion checkpoint and generate boundary-pinned paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.data import (
  Layout,
  Normalizer,
  bridge_mask,
  decode,
  encode,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.model import (
  Denoiser,
  ModelCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.process import (
  Diffusion,
  ProcessCfg,
)
from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_conjugate, quat_mul

EXPERIMENT = "g1_diffusion_bridge"


@dataclass
class GeneratedPath:
  states: torch.Tensor
  """Dynamic reference from A through the known post-B continuation."""

  duration: torch.Tensor
  """Control ticks from A to B."""


class DiffusionBridge:
  """Generate kinematic paths; a separate feedback executor tracks them."""

  def __init__(
    self,
    process: Diffusion,
    normalizer: Normalizer,
    layout: Layout,
    history: int,
    future: int,
    fps: float,
    min_steps: int = 3,
  ):
    self.process = process.eval()
    self.normalizer = normalizer
    self.layout = layout
    self.history = history
    self.future = future
    self.fps = fps
    self.min_steps = min_steps

  @property
  def max_steps(self) -> int:
    return self.process.denoiser.columns - self.history - self.future + 1

  @classmethod
  def load(
    cls, checkpoint: Path, device: str = "cpu", sample_steps: int | None = None
  ) -> DiffusionBridge:
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved.get("format") != "mjlab-diffusion-bridge-v4":
      raise ValueError(f"{checkpoint} is not a position-only v4 checkpoint; retrain it")
    layout = Layout(**saved["layout"])
    history = int(saved["history"])
    future = int(saved["future"])
    cfg = ProcessCfg(**saved["process_cfg"])
    if sample_steps is not None:
      cfg = ProcessCfg(steps=cfg.steps, sample_steps=sample_steps)
    columns = history + int(saved["max_steps"]) + future - 1
    model = Denoiser(layout.width, columns, ModelCfg(**saved["model_cfg"]))
    process = Diffusion(model, cfg).to(device)
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

    columns = self.process.denoiser.columns
    mask = bridge_mask(
      batch,
      columns,
      self.layout,
      self.history,
      self.future,
      duration,
    )
    anchor = history[:, -1]
    values = history.new_zeros((batch, columns, self.layout.width))
    values[:, : self.history] = encode(history, anchor)
    target_features = encode(target, anchor)
    rows = self.history - 1 + duration
    offsets = torch.arange(self.future, device=history.device)
    indexes = torch.arange(batch, device=history.device)[:, None]
    values[indexes, rows[:, None] + offsets] = target_features
    normalized = self.normalizer.normalize(values)
    rungs: list[torch.Tensor] | None = [] if trace is not None else None
    denoised = self.process.sample(normalized, mask, rungs)
    if trace is not None and rungs is not None:
      trace.extend(self.assemble(rung, anchor, target, duration) for rung in rungs)
    return self.assemble(denoised, anchor, target, duration)

  def assemble(
    self,
    denoised: torch.Tensor,
    anchor: torch.Tensor,
    target: torch.Tensor,
    duration: torch.Tensor,
  ) -> GeneratedPath:
    """Decode positions, derive velocities, and copy dynamic boundaries exactly."""
    poses = decode(self.normalizer.denormalize(denoised), anchor, self.layout)
    poses = poses[:, self.history - 1 :]
    position = poses[..., :3]
    orientation = poses[..., 3:7]
    joint_position = poses[..., 7:]
    linear = (position[:, 1:] - position[:, :-1]) * self.fps
    angular = (
      axis_angle_from_quat(
        quat_mul(orientation[:, 1:], quat_conjugate(orientation[:, :-1]))
      )
      * self.fps
    )
    joint_velocity = (joint_position[:, 1:] - joint_position[:, :-1]) * self.fps
    states = torch.cat(
      (
        position,
        orientation,
        torch.cat((linear, linear[:, -1:]), dim=1),
        torch.cat((angular, angular[:, -1:]), dim=1),
        joint_position,
        torch.cat((joint_velocity, joint_velocity[:, -1:]), dim=1),
      ),
      dim=-1,
    )
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
) -> dict:
  return {
    "format": "mjlab-diffusion-bridge-v4",
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
  }
