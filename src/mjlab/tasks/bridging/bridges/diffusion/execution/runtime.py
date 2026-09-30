"""Plan once with diffusion and execute the path with feedback tracking."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import torch

from mjlab.tasks.bridging.bridges.dataset.dataset import (
  find_checkpoint,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.bridge import (
  EXPERIMENT,
  DiffusionBridge,
  GeneratedPath,
)
from mjlab.tasks.bridging.bridges.interface import (
  Bridge,
  BridgeOutput,
)

PathExecutor = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
EndpointBoxTest = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class DiffusionRuntime(Bridge):
  """Generate a kinematic path and execute it with a feedback tracker."""

  def __init__(
    self,
    action_dim: int,
    checkpoint: Path | None = None,
    sample_steps: int | None = None,
    executor: PathExecutor | None = None,
    endpoint_box_test: EndpointBoxTest | None = None,
    executor_horizon: int = 5,
  ) -> None:
    super().__init__(action_dim)
    if executor_horizon < 1:
      raise ValueError("executor_horizon must be positive")
    self.checkpoint = checkpoint
    self.sample_steps = sample_steps
    self.executor = executor
    self.endpoint_box_test = endpoint_box_test
    self.executor_horizon = executor_horizon
    self._bridge: DiffusionBridge | None = None
    self.path: GeneratedPath | None = None
    self._index: torch.Tensor | None = None
    self._active: torch.Tensor | None = None

  def set_executor(
    self, executor: PathExecutor, endpoint_box_test: EndpointBoxTest | None = None
  ) -> None:
    """Install the tracker used to turn path references into actions."""
    self.executor = executor
    self.endpoint_box_test = endpoint_box_test
    self.executor_horizon = int(getattr(executor, "horizon", self.executor_horizon))

  def load(self, device: torch.device | str) -> DiffusionBridge:
    if self._bridge is None:
      path = find_checkpoint(
        (EXPERIMENT,),
        str(self.checkpoint) if self.checkpoint is not None else None,
        hint=" Train one with bridges.diffusion.planner.train.",
      )
      print(f"bridge   {path}")
      bridge = DiffusionBridge.load(path, str(device), self.sample_steps)
      self._bridge = bridge
    return self._bridge

  def reset(self, done: torch.Tensor | None = None) -> None:
    reset = getattr(self.executor, "reset", None)
    if callable(reset):
      reset(done)
    if done is None:
      self.path = None
      self._index = None
      self._active = None
    elif self._active is not None:
      if done.shape != self._active.shape:
        raise ValueError("done must have one flag per environment")
      self._active[done] = False

  def _reference(self) -> torch.Tensor:
    assert self.path is not None and self._index is not None
    batch = self.path.states.shape[0]
    offsets = torch.arange(self.executor_horizon, device=self.path.states.device)
    rows = torch.minimum(self._index[:, None] + offsets, self.path.duration[:, None])
    return self.path.states[torch.arange(batch, device=rows.device)[:, None], rows]

  def step(
    self, history: torch.Tensor, target: torch.Tensor, time_left: torch.Tensor
  ) -> BridgeOutput:
    """Execute one control tick from a path with a fixed total duration."""
    if self.executor is None:
      raise RuntimeError(
        "DiffusionRuntime needs a feedback path executor. "
        "Install one with set_executor()."
      )
    current = history[:, -1]
    bridge = self.load(current.device)
    executor_fps = float(getattr(self.executor, "fps", bridge.fps))
    if not math.isclose(executor_fps, bridge.fps):
      raise ValueError(
        f"planner runs at {bridge.fps:g} Hz but executor runs at {executor_fps:g} Hz"
      )
    batch = current.shape[0]
    if target.shape[1] < bridge.future:
      raise ValueError(
        f"diffusion checkpoint requires {bridge.future} consecutive target states"
      )
    if self._active is None:
      self._index = torch.zeros(batch, device=current.device, dtype=torch.long)
      self._active = torch.zeros(batch, device=current.device, dtype=torch.bool)
    assert self._index is not None
    assert self._active is not None
    if self._active.shape[0] != batch:
      raise ValueError("batch size changed; reset the runtime before reusing it")
    new = ~self._active
    if bool(new.any()):
      ticks = (time_left[new] * bridge.fps).round().long()
      if history.shape[1] < bridge.history:
        raise ValueError(
          f"diffusion checkpoint requires {bridge.history} observed history states"
        )
      generated = bridge.generate(
        history[new, -bridge.history :], target[new, : bridge.future], ticks
      )
      if self.path is None:
        self.path = GeneratedPath(
          current.new_zeros((batch, *generated.states.shape[1:])),
          torch.zeros(batch, device=current.device, dtype=torch.long),
        )
      self.path.states[new] = generated.states
      self.path.duration[new] = ticks
      self._index[new] = 0
      self._active[new] = True
    assert self.path is not None
    set_plan = getattr(self.executor, "set_plan", None)
    if callable(set_plan):
      set_plan(self.path.states, self.path.duration, self._index)
    action = self.executor(history, self._reference())
    if action.shape != (batch, self.action_dim):
      raise ValueError("path executor returned the wrong action shape")
    played = self._index.clone()
    self._index[:] = torch.minimum(self._index + 1, self.path.duration)
    within_endpoint_box = (
      self.endpoint_box_test(history, target)
      if self.endpoint_box_test is not None
      else torch.zeros_like(played, dtype=torch.bool)
    )
    if within_endpoint_box.shape != (batch,):
      raise ValueError("endpoint box test returned the wrong shape")
    return BridgeOutput(
      action=action,
      handoff=played >= self.path.duration,
      within_endpoint_box=within_endpoint_box,
      blend=(played.float() / self.path.duration.float()).clamp(max=1.0),
    )
