"""Editable scene, policy and handoff settings for the soccer demo."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).with_name("config.yml")


@dataclass(frozen=True)
class Scene:
  robot_position: tuple[float, float] = (0.0, 0.0)
  robot_heading_degrees: float = 0.0
  ball_position: tuple[float, float] = (6.0, 0.0)
  fallen_position: tuple[float, float, float] = (3.0, 0.0, 0.16)
  fallen_rotation_degrees: tuple[float, float, float] = (0.0, 90.0, 90.0)
  fallen_joints: dict[str, float] = field(default_factory=dict)
  fallen_position_jitter: tuple[float, float] = (0.03, 0.02)
  fallen_yaw_jitter_degrees: float = 2.0
  fallen_joint_jitter: dict[str, float] = field(
    default_factory=lambda: {
      "left_shoulder_pitch_joint": 0.04,
      "right_shoulder_pitch_joint": 0.04,
      "left_elbow_joint": 0.04,
      "right_elbow_joint": 0.04,
      "left_hip_roll_joint": 0.025,
      "right_hip_roll_joint": 0.025,
      "left_knee_joint": 0.025,
      "right_knee_joint": 0.025,
    }
  )
  goal_position: tuple[float, float] = (8.0, 0.0)
  goal_heading_degrees: float = 0.0
  goal_width: float = 2.0
  goal_height: float = 1.2
  pitch_length: float = 12.0
  pitch_width: float = 8.0


@dataclass(frozen=True)
class Handoff:
  heading_degrees: float | None = 0.0
  entry_index: int = 0
  start_distance: float = 0.5
  duration: float = 0.8
  tolerance_scale: float = 2.0
  require_endpoint: bool = True


@dataclass(frozen=True)
class Policies:
  locomotion_task: str = "Mjlab-G1-Walk"
  locomotion_checkpoint: str | None = None
  jump_checkpoint: str | None = None
  kick_checkpoint: str | None = None
  selector_path: str = "data/selector/states.npz"
  locomotion_entry_path: str = "data/soccer/g1_locomotion_entry.npz"
  speed_before_jump: float = 0.8
  speed_after_jump: float = 0.8
  record_warmup_steps: int = 150
  record_frames: int = 16


@dataclass(frozen=True)
class BridgeSettings:
  kind: str = "diffusion"
  checkpoint: str | None = None
  tracker_checkpoint: str | None = None
  sample_steps: int | None = None


@dataclass(frozen=True)
class ControllerSettings:
  lateral_tolerance: float = 0.15
  heading_tolerance: float = 0.30
  turn_speed: float = 0.7
  clearance_distance: float = 0.45
  lift_height: float = 0.12
  jump_exit_requires_landing: bool = True
  ball_position_tolerance: float = 0.03
  fall_height: float = 0.40
  phase_timeout: float = 15.0
  kick_timeout: float = 8.0
  launch_speed: float = 0.5
  require_goal: bool = True


@dataclass(frozen=True)
class Settings:
  scene: Scene = field(default_factory=Scene)
  policies: Policies = field(default_factory=Policies)
  bridge: BridgeSettings = field(default_factory=BridgeSettings)
  controller: ControllerSettings = field(default_factory=ControllerSettings)
  walk_to_jump: Handoff = field(
    default_factory=lambda: Handoff(
      entry_index=3,
      start_distance=0.6,
      duration=0.8,
      require_endpoint=False,
    )
  )
  jump_to_walk: Handoff = field(
    default_factory=lambda: Handoff(
      heading_degrees=None, start_distance=0.25, duration=0.5
    )
  )
  walk_to_kick: Handoff = field(
    default_factory=lambda: Handoff(
      heading_degrees=None,
      entry_index=3,
      start_distance=0.2,
      duration=0.4,
      require_endpoint=False,
    )
  )

  @classmethod
  def load(cls, path: Path = CONFIG_PATH) -> Settings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
      raise ValueError("Soccer config must be a YAML mapping")
    defaults = cls()
    groups = {
      "scene": Scene,
      "policies": Policies,
      "bridge": BridgeSettings,
      "controller": ControllerSettings,
      "walk_to_jump": Handoff,
      "jump_to_walk": Handoff,
      "walk_to_kick": Handoff,
    }
    unknown = set(raw) - set(groups)
    if unknown:
      raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    values: dict[str, Any] = {}
    for name, kind in groups.items():
      if name not in raw:
        values[name] = getattr(defaults, name)
        continue
      if not isinstance(raw[name], dict):
        raise ValueError(f"{name} must be a mapping")
      try:
        values[name] = kind(**{**asdict(getattr(defaults, name)), **raw[name]})
      except TypeError as error:
        raise ValueError(f"Invalid {name} settings: {error}") from error
    settings = cls(**values)
    settings.validate()
    return settings

  def validate(self) -> None:
    vectors = {
      "robot_position": (self.scene.robot_position, 2),
      "ball_position": (self.scene.ball_position, 2),
      "fallen_position": (self.scene.fallen_position, 3),
      "fallen_rotation_degrees": (self.scene.fallen_rotation_degrees, 3),
      "fallen_position_jitter": (self.scene.fallen_position_jitter, 2),
      "goal_position": (self.scene.goal_position, 2),
    }
    for name, handoff in self.handoffs.items():
      if type(handoff.entry_index) is not int or handoff.entry_index < 0:
        raise ValueError(f"{name}.entry_index must be a nonnegative integer")
      if handoff.start_distance < 0 or not math.isfinite(handoff.start_distance):
        raise ValueError(f"{name}.start_distance must be finite and nonnegative")
      if (
        handoff.duration <= 0
        or handoff.tolerance_scale <= 0
        or not all(
          math.isfinite(value) for value in (handoff.duration, handoff.tolerance_scale)
        )
      ):
        raise ValueError(
          f"{name} duration and tolerance_scale must be finite and positive"
        )
      if handoff.heading_degrees is not None and not math.isfinite(
        handoff.heading_degrees
      ):
        raise ValueError(f"{name}.heading_degrees must be finite")
    for name, (vector, size) in vectors.items():
      if len(vector) != size or not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain {size} finite numbers")
    for group in (self.scene, self.policies, self.controller):
      for name, value in vars(group).items():
        if (
          type(value) in (float, int)
          and name != "record_warmup_steps"
          and (not math.isfinite(value) or ("degrees" not in name and value <= 0))
        ):
          raise ValueError(f"{name} must be finite and positive")
    if any(not math.isfinite(value) for value in self.scene.fallen_joints.values()):
      raise ValueError("fallen_joints must contain finite angles")
    jitter = (
      *self.scene.fallen_position_jitter,
      self.scene.fallen_yaw_jitter_degrees,
      *self.scene.fallen_joint_jitter.values(),
    )
    if any(not math.isfinite(value) or value < 0 for value in jitter):
      raise ValueError("Fallen robot jitter bounds must be finite and nonnegative")
    if self.policies.record_warmup_steps < 0 or self.policies.record_frames < 1:
      raise ValueError("Recording needs nonnegative warmup and positive frames")
    if self.scene.fallen_position[2] < 0:
      raise ValueError("fallen_position height must be nonnegative")
    if self.bridge.kind not in {"diffusion", "mixed", "no-op"}:
      raise ValueError("bridge.kind must be diffusion, mixed or no-op")
    if self.bridge.sample_steps is not None and self.bridge.sample_steps < 1:
      raise ValueError("bridge.sample_steps must be positive")
    if tuple(self.scene.ball_position) == tuple(self.scene.goal_position):
      raise ValueError("Ball and goal must have distinct positions")

  @property
  def handoffs(self) -> dict[str, Handoff]:
    return {
      "walk_to_jump": self.walk_to_jump,
      "jump_to_walk": self.jump_to_walk,
      "walk_to_kick": self.walk_to_kick,
    }
