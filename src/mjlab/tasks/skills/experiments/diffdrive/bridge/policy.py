"""The learned bridge at deployment: pick a merge frame, then drive onto it.

LearnedBridge wraps the policy trained for the Mjlab-Bridge-Diffdrive task so it fits the
Bridge interface the controller expects. At the moment of a switch it reduces skill2's tube
to one representative line (the medoid) and asks the trained selector, from the two windows,
which frame of that line to merge into. Then on every tick it feeds the goal-relative
observation (the chosen frame), a short history of recent motion, and the two windows to the
actor and returns the twist it asks for, reporting done once the robot has reached the merge
frame, where the controller hands control to skill2 to follow the rest.

The actor and the selector are the two ONNX files exported during training (the selector
file is the actor file with a _selector suffix). Skill2's tube is reharvested here the same
way training built it, so deployment needs only the checkpoints plus the skills.

At a switch the bridge does not have skill1's real approach window (it only starts running
once the switch fires), so the selector's skill1 window is approximated by the current
state, held for the window's length. The interrupt state, which carries the motion the
merge has to be compatible with, is exact.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch

from mjlab.tasks.skills.experiments.diffdrive.bridge import features
from mjlab.tasks.skills.experiments.diffdrive.bridge.rollouts import (
  representative,
  window_family,
)
from mjlab.tasks.skills.experiments.diffdrive.experiment import CONFIG, build_model
from mjlab.tasks.skills.experiments.diffdrive.gridworld import GridWorld
from mjlab.tasks.skills.experiments.diffdrive.robot import OMEGA, DiffDrive, V
from mjlab.tasks.skills.experiments.diffdrive.skills import (
  CorridorSkill,
  corridor_skills,
)
from mjlab.tasks.skills.interfaces import Bridge, Command, Skill, State


class LearnedBridge(Bridge):
  """Deploy the trained bridge policy across a skill switch."""

  def __init__(
    self, checkpoint: str, world: GridWorld, speeds: dict[int, float]
  ) -> None:
    import onnxruntime  # local: keep onnxruntime off the package import path

    selector_path = checkpoint.replace(".onnx", "_selector.onnx")
    self._actor = onnxruntime.InferenceSession(
      checkpoint, providers=["CPUExecutionProvider"]
    )
    self._selector = onnxruntime.InferenceSession(
      selector_path, providers=["CPUExecutionProvider"]
    )

    # Each skill's tube, reduced to one representative line (the candidate merge frames),
    # plus the per-dimension scale that makes "on the tube" a dimensionless test.
    robot = DiffDrive()
    model = build_model(world, robot)
    skills = corridor_skills(world, speeds)
    rng = np.random.default_rng(0)
    self._tube: dict[int, torch.Tensor] = {}
    self._scale: dict[int, torch.Tensor] = {}
    for cid, skill in skills.items():
      family = window_family(
        model, robot, skill, CONFIG.window_steps, CONFIG.couples_per_junction, rng
      )
      self._tube[cid] = torch.as_tensor(
        representative(family, CONFIG.representative), dtype=torch.float32
      )
      self._scale[cid] = features.tube_scale(
        torch.as_tensor(np.stack(family), dtype=torch.float32)
      )

    self._target: int | None = None
    self._merge: int | None = (
      None  # merge frame, picked by the selector on the first step
    )
    self._history: torch.Tensor | None = None
    self._window: torch.Tensor | None = (
      None  # the two windows encoded, fixed per switch
    )

  def reset(self, from_skill: Skill, to_skill: Skill) -> None:
    self._target = cast("CorridorSkill", to_skill).cid
    self._merge = None  # selected on the first step, from the interrupt state
    self._history = None
    self._window = None

  def step(self, state: State) -> tuple[Command, bool]:
    assert self._target is not None
    tube = self._tube[self._target]
    scale = self._scale[self._target]
    s = torch.as_tensor(np.asarray(state, float), dtype=torch.float32)

    if (
      self._merge is None
    ):  # first tick: seed history, encode the windows, pick the frame
      self._history = s[[V, OMEGA]].unsqueeze(0).repeat(CONFIG.history_len, 1)
      end_window = s.unsqueeze(0).repeat(len(tube), 1)  # skill1 window approximated
      self._window = features.window_features(end_window, tube)
      scores = self._infer(self._selector, self._window)
      self._merge = int(np.argmax(scores))

    goal = tube[self._merge]
    raw = torch.as_tensor(self._infer(self._actor, self._obs(s, goal)))
    v, omega = features.twist_from_action(raw)
    reached = bool(
      features.tube_distance(s, tube, scale)[self._merge] < CONFIG.merge_tol
    )
    assert self._history is not None
    self._history = torch.cat([self._history[1:], s[[V, OMEGA]].unsqueeze(0)], dim=0)
    return np.array([float(v), float(omega)]), reached

  def _obs(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    assert self._history is not None and self._window is not None
    return torch.cat(
      [features.observation(state, goal), self._history.reshape(-1), self._window]
    )

  def _infer(self, session, vec: torch.Tensor) -> np.ndarray:
    inputs = {"obs": vec.unsqueeze(0).numpy().astype(np.float32)}
    return np.asarray(session.run(None, inputs)[0])[0]
