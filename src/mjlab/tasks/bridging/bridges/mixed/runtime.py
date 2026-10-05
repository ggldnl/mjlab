"""Plan once with the footstep diffusion bridge and execute with the tracker."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from pathlib import Path

import torch

from mjlab.tasks.bridging.bridges.dataset.dataset import find_checkpoint
from mjlab.tasks.bridging.bridges.diffusion.execution.runtime import (
  DiffusionRuntime,
)
from mjlab.tasks.bridging.bridges.mixed.bridge import (
  CHECKPOINT_FORMAT,
  FootstepBridge,
  planner_experiment,
)


class MixedRuntime(DiffusionRuntime):
  """DiffusionRuntime whose planner places footsteps before inpainting the body."""

  robot = "g1"

  def latest_checkpoint(self) -> Path:
    return find_checkpoint(
      (planner_experiment(self.robot),), hint=" Train one with bridges.mixed.train."
    )

  def load(self, device: torch.device | str) -> FootstepBridge:
    if self._bridge is None:
      path = find_checkpoint(
        (planner_experiment(self.robot),),
        str(self.checkpoint) if self.checkpoint is not None else None,
        hint=" Train one with bridges.mixed.train.",
      )
      print(f"bridge   {path}")
      self._bridge = FootstepBridge.load(path, str(device), self.sample_steps)
    assert isinstance(self._bridge, FootstepBridge)
    return self._bridge


def runtime_kind(bridge: str, checkpoint: Path | None) -> type[DiffusionRuntime]:
  """Plain or footstep diffusion runtime for a checkpoint.

  A given checkpoint decides by its format, so a footstep checkpoint also runs
  under bridge "diffusion" and a plain one under "mixed". Without a checkpoint,
  bridge picks whose newest checkpoint is used.
  """
  wanted = MixedRuntime if bridge == "mixed" else DiffusionRuntime
  if checkpoint is None:
    return wanted
  saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
  footsteps = saved.get("planner", saved).get("format") == CHECKPOINT_FORMAT
  kind = MixedRuntime if footsteps else DiffusionRuntime
  if kind is not wanted:
    name = "footstep" if footsteps else "plain"
    print(f"bridge   {checkpoint} is a {name} diffusion checkpoint, loading it as such")
  return kind
