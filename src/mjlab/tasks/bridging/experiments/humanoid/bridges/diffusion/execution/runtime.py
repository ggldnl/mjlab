"""Plan once with diffusion and execute the path with feedback tracking."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch

from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  find_checkpoint,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.planner.bridge import (
  EXPERIMENT,
  DiffusionBridge,
  GeneratedPath,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.interface import (
  Bridge,
  BridgeOutput,
)

PathExecutor = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
CaptureTest = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class DiffusionRuntime(Bridge):
  """Generate a kinematic path and pass a moving reference to UniTracker."""

  def __init__(
    self,
    action_dim: int,
    checkpoint: Path | None = None,
    sample_steps: int | None = None,
    executor: PathExecutor | None = None,
    capture_test: CaptureTest | None = None,
    executor_horizon: int = 5,
  ) -> None:
    super().__init__(action_dim)
    if executor_horizon < 1:
      raise ValueError("executor_horizon must be positive")
    self.checkpoint = checkpoint
    self.sample_steps = sample_steps
    self.executor = executor
    self.capture_test = capture_test
    self.executor_horizon = executor_horizon
    self._bridge: DiffusionBridge | None = None
    self.path: GeneratedPath | None = None
    self._ticks: torch.Tensor | None = None
    self._index: torch.Tensor | None = None
    self._active: torch.Tensor | None = None

  def set_executor(
    self, executor: PathExecutor, capture_test: CaptureTest | None = None
  ) -> None:
    """Install the tracker used to turn path references into actions."""
    self.executor = executor
    self.capture_test = capture_test
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
      self._ticks = None
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
    rows = (self._index[:, None] + offsets).clamp_max(self.path.states.shape[1] - 1)
    return self.path.states[torch.arange(batch, device=rows.device)[:, None], rows]

  def step(
    self, history: torch.Tensor, target: torch.Tensor, time_left: torch.Tensor
  ) -> BridgeOutput:
    if self.executor is None:
      raise RuntimeError(
        "DiffusionRuntime needs UniTracker or another feedback path executor. "
        "Install one with set_executor()."
      )
    current = history[:, -1]
    bridge = self.load(current.device)
    batch = current.shape[0]
    if target.shape[1] < bridge.future:
      raise ValueError(
        f"diffusion checkpoint requires {bridge.future} consecutive target states"
      )
    if self._active is None:
      self._ticks = torch.zeros(batch, device=current.device, dtype=torch.long)
      self._index = torch.zeros_like(self._ticks)
      self._active = torch.zeros(batch, device=current.device, dtype=torch.bool)
    assert self._ticks is not None and self._index is not None
    assert self._active is not None
    if self._active.shape[0] != batch:
      raise ValueError("batch size changed; reset the runtime before reusing it")
    new = ~self._active
    if bool(new.any()):
      ticks = (
        (time_left[new] * bridge.fps)
        .round()
        .long()
        .clamp(bridge.min_steps, bridge.max_steps)
      )
      if history.shape[1] < bridge.history:
        raise ValueError(
          f"diffusion checkpoint requires {bridge.history} observed history states"
        )
      generated = bridge.generate(
        history[new, -bridge.history :], target[new, : bridge.future], ticks
      )
      if self.path is None:
        self.path = GeneratedPath(
          current.new_zeros((batch, *generated.states.shape[1:])), self._ticks
        )
      self.path.states[new] = generated.states
      self.path.duration[new] = ticks
      self._ticks[new] = ticks
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
    self._index += 1
    captured = (
      self.capture_test(history, target)
      if self.capture_test is not None
      else torch.zeros_like(played, dtype=torch.bool)
    )
    if captured.shape != (batch,):
      raise ValueError("capture test returned the wrong shape")
    handoff = played >= self._ticks
    if self.capture_test is not None:
      handoff &= captured
    return BridgeOutput(
      action=action,
      handoff=handoff,
      captured=captured,
      blend=(played.float() / self._ticks.float()).clamp(max=1.0),
    )
