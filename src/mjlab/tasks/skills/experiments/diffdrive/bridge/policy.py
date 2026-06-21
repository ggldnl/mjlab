"""The learned bridge at deployment: load the trained policy and drive a switch.

LearnedBridge wraps the ONNX policy exported during training so it fits the Bridge
interface the controller expects. At the moment of a switch it picks the goal exactly
as training did, namely the next skill's window state closest to the interrupt state,
then on every tick it feeds the goal-relative observation to the policy and returns
the twist it asks for, reporting done once the goal is reached.

The goal is selected once, on the first tick after a reset, because that first tick
is when the interrupt state (where the previous skill left the robot) is known.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch

from mjlab.tasks.skills.experiments.diffdrive.bridge import features, rollouts
from mjlab.tasks.skills.experiments.diffdrive.gridworld import GridWorld
from mjlab.tasks.skills.experiments.diffdrive.skills import CorridorSkill
from mjlab.tasks.skills.interfaces import Bridge, Command, Skill, State


class LearnedBridge(Bridge):
  """Deploy the trained bridge policy across a skill switch."""

  def __init__(
    self,
    checkpoint: str,
    world: GridWorld,
    speeds: dict[int, float],
    mode: str = "cruise",
  ) -> None:
    import onnxruntime  # local: keep onnxruntime off the package import path

    self._session = onnxruntime.InferenceSession(
      checkpoint, providers=["CPUExecutionProvider"]
    )
    self._windows = {
      cid: torch.as_tensor(window, dtype=torch.float32)
      for cid, window in rollouts.harvest_windows(world, speeds, mode).items()
    }
    self._target: int | None = None
    self._goal: torch.Tensor | None = None

  def reset(self, from_skill: Skill, to_skill: Skill) -> None:
    self._target = cast("CorridorSkill", to_skill).cid
    self._goal = None  # selected on the first step, from the interrupt state

  def step(self, state: State) -> tuple[Command, bool]:
    assert self._target is not None
    s = torch.as_tensor(np.asarray(state, float), dtype=torch.float32)
    if self._goal is None:
      window = self._windows[self._target]
      self._goal = window[features.goal_distance(s, window).argmin()]
    raw = self._infer(features.observation(s, self._goal))
    v, omega = features.twist_from_action(raw)
    done = bool(features.success(s, self._goal))
    return np.array([float(v), float(omega)]), done

  def _infer(self, obs: torch.Tensor) -> torch.Tensor:
    inputs = {"obs": obs.unsqueeze(0).numpy()}
    action = np.asarray(self._session.run(["actions"], inputs)[0])[0]
    return torch.as_tensor(action, dtype=torch.float32)
