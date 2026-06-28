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

The selector also needs skill1's approach window. The controller hands every state to the
bridge through observe, even while skill1 is active, so the bridge keeps a rolling buffer of
the last states; at a switch its last frame is the interrupt and the buffer is skill1's end
window, the same shape training used.
"""

from __future__ import annotations

from collections import deque
from typing import cast

import numpy as np
import torch

from mjlab.tasks.skills.experiments.diffdrive.bridge import features
from mjlab.tasks.skills.experiments.diffdrive.bridge.rollouts import (
  representative,
  start_window_family,
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
      family = start_window_family(
        model, robot, skill, CONFIG.couples_per_junction, CONFIG.window_steps, rng
      )
      self._tube[cid] = torch.as_tensor(
        representative(list(family), CONFIG.representative), dtype=torch.float32
      )
      self._scale[cid] = features.tube_scale(
        torch.as_tensor(np.asarray(family), dtype=torch.float32)
      )

    # The last states before a switch, fed every tick by the controller via observe. Its
    # last frame is the interrupt, so it stands in for skill1's end window for the selector.
    self._buffer: deque = deque(maxlen=CONFIG.window_steps + 1)
    self._target: int | None = None
    self._merge: int | None = (
      None  # merge frame, picked by the selector on the first step
    )
    self._history: torch.Tensor | None = None
    self._window: torch.Tensor | None = (
      None  # the two windows encoded, fixed per switch
    )
    self._reached = False  # latched once the robot is on the tube, hands over to skill2
    # TEMP(diffdrive): the on-tube reach test never fires, so the bridge never reports
    # done and skill2 never resumes. Until the tube formalization is reworked, force the
    # handover after a fixed budget of ticks so the current approach can still be tested
    # end to end. Remove this (and the _steps bookkeeping below) once reach works.
    self._steps = 0

  def observe(self, state: State) -> None:
    """Record the latest state each tick (called even while idle) for skill1's window."""
    self._buffer.append(np.asarray(state, float))

  def reset(self, from_skill: Skill, to_skill: Skill) -> None:
    # The buffer is not cleared: it carries skill1's approach into the switch.
    self._target = cast("CorridorSkill", to_skill).cid
    self._merge = None  # selected on the first step, from the interrupt state
    self._history = None
    self._window = None
    self._reached = False
    self._steps = 0  # TEMP(diffdrive): ticks since this switch began

  def step(self, state: State) -> tuple[Command, bool]:
    assert self._target is not None
    tube = self._tube[self._target]
    scale = self._scale[self._target]
    s = torch.as_tensor(np.asarray(state, float), dtype=torch.float32)

    if (
      self._merge is None
    ):  # first tick: seed history, encode the windows, pick the frame
      self._history = s[[V, OMEGA]].unsqueeze(0).repeat(CONFIG.history_len, 1)
      self._window = features.window_features(self._end_window(len(tube)), tube)
      scores = self._infer(self._selector, self._window)
      self._merge = int(np.argmax(scores))

    goal = tube[self._merge]
    raw = torch.as_tensor(self._infer(self._actor, self._obs(s, goal)))
    v, omega = features.twist_from_action(raw)
    # Hand over once the robot is on skill2's tube at or past the chosen merge frame, not
    # only at that exact frame: the executor may settle onto the tube a few frames along, or
    # drive past the merge point. Latch it so a brief contact is never missed.
    on_tube = features.tube_distance(s, tube, scale)[self._merge :]
    self._reached = self._reached or bool(on_tube.min() < CONFIG.merge_tol)
    assert self._history is not None
    self._history = torch.cat([self._history[1:], s[[V, OMEGA]].unsqueeze(0)], dim=0)
    # TEMP(diffdrive): force the handover once the bridge has run for the length of
    # skill2's window. This lets the controller resume skill2 forcibly.
    self._steps += 1
    forced = self._steps >= len(tube)
    done = self._reached or forced
    return np.array([float(v), float(omega)]), done

  def _end_window(self, length: int) -> torch.Tensor:
    """Skill1's approach window from the recent-state buffer, padded to length.

    The buffer holds the last states before the switch, so its last frame is the interrupt.
    If fewer than length states have been seen, the earliest is repeated at the front, the
    same way training pads a short end window.
    """
    buf = [torch.as_tensor(x, dtype=torch.float32) for x in self._buffer]
    if not buf:
      buf = [torch.zeros(5)]
    if len(buf) < length:
      buf = [buf[0]] * (length - len(buf)) + buf
    else:
      buf = buf[-length:]
    return torch.stack(buf)

  def _obs(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    assert self._history is not None and self._window is not None
    return torch.cat(
      [features.observation(state, goal), self._history.reshape(-1), self._window]
    )

  def _infer(self, session, vec: torch.Tensor) -> np.ndarray:
    inputs = {"obs": vec.unsqueeze(0).numpy().astype(np.float32)}
    return np.asarray(session.run(None, inputs)[0])[0]
