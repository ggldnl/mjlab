"""The learned bridge at deployment: select a merge target, then drive to it.

LearnedBridge wraps the policy trained for the Mjlab-Bridge-Diffdrive task so it fits the
Bridge interface the controller expects. At the moment of a switch it selects where to
join the next skill's tube by reading the critic's value over the tube's candidate states
and taking the best one, exactly the value-readout the training set up. Then on every tick
it feeds the goal-relative observation to the actor and returns the twist it asks for,
reporting done once the robot is on the tube.

The actor and critic are the two ONNX files exported during training (the critic file is
the actor file with a _critic suffix). The tube of each skill is reharvested here, the
same way training built it, so deployment needs only the checkpoints plus the skills.
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
from mjlab.tasks.skills.experiments.diffdrive.robot import DiffDrive
from mjlab.tasks.skills.experiments.diffdrive.skills import (
  CorridorSkill,
  corridor_skills,
)
from mjlab.tasks.skills.interfaces import Bridge, Command, Skill, State


class LearnedBridge(Bridge):
  """Deploy the trained bridge policy across a skill switch."""

  def __init__(
    self,
    checkpoint: str,
    world: GridWorld,
    speeds: dict[int, float],
    mode: str = "hold",
  ) -> None:
    import onnxruntime  # local: keep onnxruntime off the package import path

    critic_path = checkpoint.replace(".onnx", "_critic.onnx")
    self._actor = onnxruntime.InferenceSession(
      checkpoint, providers=["CPUExecutionProvider"]
    )
    self._critic = onnxruntime.InferenceSession(
      critic_path, providers=["CPUExecutionProvider"]
    )

    # Each skill's tube, reduced to one representative line, plus the per-dimension scale
    # that makes "on the tube" a dimensionless test. Harvested as in training.
    robot = DiffDrive()
    model = build_model(world, robot)
    skills = corridor_skills(world, speeds, mode=mode)
    rng = np.random.default_rng(0)
    self._tube: dict[int, torch.Tensor] = {}
    self._scale: dict[int, torch.Tensor] = {}
    for cid, skill in skills.items():
      family = window_family(
        model, robot, skill, CONFIG.window_steps, CONFIG.window_samples, rng
      )
      self._tube[cid] = torch.as_tensor(
        representative(family, CONFIG.representative), dtype=torch.float32
      )
      self._scale[cid] = features.tube_scale(
        torch.as_tensor(np.stack(family), dtype=torch.float32)
      )

    self._target: int | None = None
    self._goal: torch.Tensor | None = None

  def reset(self, from_skill: Skill, to_skill: Skill) -> None:
    self._target = cast("CorridorSkill", to_skill).cid
    self._goal = None  # selected on the first step, from the interrupt state

  def step(self, state: State) -> tuple[Command, bool]:
    assert self._target is not None
    tube = self._tube[self._target]
    s = torch.as_tensor(np.asarray(state, float), dtype=torch.float32)
    if self._goal is None:
      self._goal = tube[self._select(s, tube)]
    raw = self._infer(self._actor, features.observation(s, self._goal))
    v, omega = features.twist_from_action(raw)
    on_tube = features.tube_distance(s, tube, self._scale[self._target]).amin()
    done = bool(on_tube < CONFIG.arrival_threshold)
    return np.array([float(v), float(omega)]), done

  def _select(self, state: torch.Tensor, tube: torch.Tensor) -> int:
    """Pick the tube state with the highest critic value from the current state.

    The exported critic fixes its batch dimension to 1, so the candidate tube states
    are scored one at a time rather than in a single batched call.
    """
    obs = features.observation(state.expand(tube.shape[0], 5), tube)  # [L, 7]
    values = [
      self._critic.run(["actions"], {"obs": row[None].numpy().astype(np.float32)})[0]
      for row in obs
    ]
    return int(np.asarray(values).reshape(-1).argmax())

  def _infer(self, session, obs: torch.Tensor) -> torch.Tensor:
    inputs = {"obs": obs.unsqueeze(0).numpy().astype(np.float32)}
    action = np.asarray(session.run(["actions"], inputs)[0])[0]
    return torch.as_tensor(action, dtype=torch.float32)
