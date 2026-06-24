"""The learned bridge at deployment: select a merge target, then drive onto skill2's tube.

LearnedBridge wraps the policy trained for the Mjlab-Bridge-Diffdrive task so it fits the
Bridge interface the controller expects. At the moment of a switch it reduces skill2's tube
to one representative line (the medoid) and reads the critic's value over that line to pick
where to join, exactly the value-readout training set up. Then on every tick it feeds the
goal-relative observation (plus a short history of the robot's recent motion) to the actor
and returns the twist it asks for, reporting done once the robot is on the tube, where the
controller hands control to skill2 to follow the rest.

The actor and critic are the two ONNX files exported during training (the critic file is
the actor file with a _critic suffix). Skill2's tube is reharvested here the same way
training built it, so deployment needs only the checkpoints plus the skills.
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

    critic_path = checkpoint.replace(".onnx", "_critic.onnx")
    self._actor = onnxruntime.InferenceSession(
      checkpoint, providers=["CPUExecutionProvider"]
    )
    self._critic = onnxruntime.InferenceSession(
      critic_path, providers=["CPUExecutionProvider"]
    )

    # Each skill's tube, reduced to one representative line (the medoid the bridge aims at),
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
    self._r: int | None = None  # reference index along the tube, set on the first step
    self._history: torch.Tensor | None = None

  def reset(self, from_skill: Skill, to_skill: Skill) -> None:
    self._target = cast("CorridorSkill", to_skill).cid
    self._r = None  # merge point selected on the first step, from the interrupt state
    self._history = None

  def step(self, state: State) -> tuple[Command, bool]:
    assert self._target is not None
    tube = self._tube[self._target]
    scale = self._scale[self._target]
    s = torch.as_tensor(np.asarray(state, float), dtype=torch.float32)
    if (
      self._r is None
    ):  # first tick: seed history with current motion, pick the join point
      self._history = s[[V, OMEGA]].unsqueeze(0).repeat(CONFIG.history_len, 1)
      self._r = self._select(s, tube)

    goal = tube[self._r]
    raw = self._infer(self._actor, self._obs(s, goal))
    v, omega = features.twist_from_action(raw)
    on_tube = bool(features.tube_distance(s, tube, scale)[self._r] < CONFIG.track_tol)
    if on_tube and self._r < len(tube) - 1:
      self._r += 1
    assert self._history is not None
    self._history = torch.cat([self._history[1:], s[[V, OMEGA]].unsqueeze(0)], dim=0)
    return np.array([float(v), float(omega)]), on_tube

  def _obs(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    assert self._history is not None
    return torch.cat([features.observation(state, goal), self._history.reshape(-1)])

  def _select(self, state: torch.Tensor, tube: torch.Tensor) -> int:
    """Pick the tube state with the highest critic value from the current state."""
    values = [self._value(self._obs(state, tube[k])) for k in range(len(tube))]
    return int(np.argmax(values))

  def _value(self, obs: torch.Tensor) -> float:
    out = self._critic.run(["actions"], {"obs": obs[None].numpy().astype(np.float32)})[
      0
    ]
    return float(np.asarray(out).reshape(-1)[0])

  def _infer(self, session, obs: torch.Tensor) -> torch.Tensor:
    inputs = {"obs": obs.unsqueeze(0).numpy().astype(np.float32)}
    action = np.asarray(session.run(["actions"], inputs)[0])[0]
    return torch.as_tensor(action, dtype=torch.float32)
