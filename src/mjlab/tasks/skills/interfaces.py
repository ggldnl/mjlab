"""
An experiment is always the same five actors wired together:

    World:      the space the robot moves in.
    Robot:      the physical body (MuJoCo model), how to read its reduced state back
                from MjData, and how to turn a high-level command into actuator ctrl.
    Skill:      state -> command. One narrow policy (follow a corridor).
    Bridge:     what to command between skills: reset(from, to) then
                step(state) -> (command, done).
    Controller: owns the skills and one bridge; decides itself when and to what to
                switch, runs the bridge until done, then activates the target.

Shared vocabulary:

    State    = np.ndarray   the robot's reduced state (layout is system-specific, e.g.
                            [x, y, theta, v, omega] for diffdrive).
    Command  = np.ndarray   the high-level action a skill or bridge emits (e.g. a target
                            body twist (v*, omega*) for diffdrive). The Robot then maps a
                            Command to actuator ctrl.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import mujoco
import numpy as np

State = np.ndarray
Command = np.ndarray
Skill = Callable[[State], Command]


class World(ABC):
  """The space the robot moves in."""

  @abstractmethod
  def is_free(self, x, y) -> np.ndarray | bool:
    """Whether world point (x, y) is traversable, or a wall / out of bounds."""
    ...


class Robot(ABC):
  """The physical robot: its MuJoCo model, its state, and how to drive it.

  Lifecycle: attach_to the body into a world spec, compile, bind to cache indices,
  then each tick sense the reduced state and actuate a command into ctrl; reset places
  the body to realize a given reduced state.
  """

  @abstractmethod
  def attach_to(self, world: mujoco.MjSpec) -> None:
    """Attach the robot's bodies into world, before it is compiled."""
    ...

  @abstractmethod
  def bind(self, model: mujoco.MjModel) -> None:
    """Cache indices from the compiled model, after compile and before stepping."""
    ...

  @abstractmethod
  def sense(self, model: mujoco.MjModel, data: mujoco.MjData) -> State:
    """The robot's reduced state read from MjData."""
    ...

  @abstractmethod
  def actuate(
    self, model: mujoco.MjModel, data: mujoco.MjData, command: Command
  ) -> np.ndarray:
    """Map command to actuator ctrl; returns the full (nu,) vector."""
    ...

  @abstractmethod
  def reset(self, model: mujoco.MjModel, data: mujoco.MjData, state: State) -> None:
    """Place the robot (qpos/qvel) so it realizes the reduced state."""
    ...


class Bridge(ABC):
  """What to command between skills."""

  @abstractmethod
  def reset(self, from_skill: Skill, to_skill: Skill) -> None:
    """Begin a transition from from_skill to to_skill."""
    ...

  @abstractmethod
  def step(self, state: State) -> tuple[Command, bool]:
    """Return (command, done); done means the next skill may take over."""
    ...


class Controller(ABC):
  """Decides which skill is active and runs a bridge across switches.

  Owns the skills and one bridge, decides itself when and to what to switch, and uses
  the bridge to transition. The decision is experiment-specific (positional in the
  corridor world; a planner or a net elsewhere); the interface is just a state-driven
  command source.
  """

  @abstractmethod
  def reset(self) -> None:
    """Reset to the initial active skill, no switch in progress."""
    ...

  @abstractmethod
  def step(self, state: State) -> Command:
    """Return the command to apply this tick: active skill, or the bridge mid-switch."""
    ...
