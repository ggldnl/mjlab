"""Entry selection, bridge timing and placement for each Sokoban switch."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).with_name("config.yml")


@dataclass(frozen=True)
class Handoff:
  entry_index: int = 0
  duration: float = 1.0
  start_distance: float = 0.0

  def __post_init__(self) -> None:
    if type(self.entry_index) is not int or self.entry_index < 0:
      raise ValueError("entry_index must be a nonnegative integer")
    for name, value in (
      ("duration", self.duration),
      ("start_distance", self.start_distance),
    ):
      if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (name == "duration" and value == 0)
      ):
        raise ValueError(
          f"{name} must be finite and {'positive' if name == 'duration' else 'nonnegative'}"
        )


@dataclass(frozen=True)
class TurnSettings:
  command_deadzone: float = 0.01
  heading_gain: float = 3.0

  def __post_init__(self) -> None:
    if any(
      type(value) not in (int, float) or not math.isfinite(value) or value <= 0
      for value in (self.command_deadzone, self.heading_gain)
    ):
      raise ValueError("Turn deadzone and heading gain must be finite and positive")
    if self.command_deadzone >= math.pi:
      raise ValueError("Turn command_deadzone must be less than pi")


@dataclass(frozen=True)
class Handoffs:
  turn: TurnSettings = field(default_factory=TurnSettings)
  walk_to_turn: Handoff = field(default_factory=Handoff)
  turn_to_walk: Handoff = field(default_factory=Handoff)
  walk_to_push: Handoff = field(default_factory=Handoff)
  push_to_walk: Handoff = field(default_factory=Handoff)

  @property
  def switches(self) -> dict[str, Handoff]:
    return {
      name: getattr(self, name)
      for name in ("walk_to_turn", "turn_to_walk", "walk_to_push", "push_to_walk")
    }

  @classmethod
  def load(cls, path: Path = CONFIG_PATH) -> Handoffs:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
      raise ValueError("Sokoban config must be a YAML mapping")
    defaults = cls()
    unknown = set(raw) - (defaults.switches.keys() | {"turn"})
    if unknown:
      raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    values = {}
    for name, default in {"turn": defaults.turn, **defaults.switches}.items():
      section = raw.get(name, {})
      if not isinstance(section, dict):
        raise ValueError(f"{name} must be a mapping")
      try:
        values[name] = type(default)(**{**asdict(default), **section})
      except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {name} settings: {error}") from error
    return cls(**values)
