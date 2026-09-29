"""Retargeted BABEL motion windows for endpoint conditioned inbetweening."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mjlab.tasks.bridging.bridges.dataset.dataset import Dataset, Segments
from mjlab.tasks.bridging.bridges.diffusion.config import (
  motion_patterns as configured_motion_patterns,
)
from mjlab.tasks.bridging.bridges.diffusion.config import robot_name
from mjlab.tasks.bridging.motion_filter import (
  DEFAULT_MOTION_FILTER,
  MotionFilterCfg,
  motion_bad_frames,
)
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_from_matrix,
  quat_mul,
  yaw_quat,
)

BABEL_ROOT = Path("data") / "babel_retargeted" / "unitree_g1_locomotion_v1"
BABEL_TRAIN_MOTIONS = configured_motion_patterns("g1", "train")
BABEL_EVAL_MOTIONS = configured_motion_patterns("g1", "val")
DEFAULT_MOTIONS = BABEL_TRAIN_MOTIONS

_G1_MIRROR_JOINTS = (
  6,
  7,
  8,
  9,
  10,
  11,
  0,
  1,
  2,
  3,
  4,
  5,
  12,
  13,
  14,
  22,
  23,
  24,
  25,
  26,
  27,
  28,
  15,
  16,
  17,
  18,
  19,
  20,
  21,
)
_G1_MIRROR_SIGNS = (
  1,
  -1,
  -1,
  1,
  1,
  -1,
  1,
  -1,
  -1,
  1,
  1,
  -1,
  -1,
  -1,
  1,
  1,
  -1,
  -1,
  1,
  -1,
  1,
  -1,
  1,
  -1,
  -1,
  1,
  -1,
  1,
  -1,
)


@dataclass(frozen=True)
class Layout:
  joints: int

  @property
  def width(self) -> int:
    return 15 + 2 * self.joints

  @property
  def root_linear_velocity(self) -> slice:
    return slice(9, 12)

  @property
  def root_angular_velocity(self) -> slice:
    return slice(12, 15)

  @property
  def joint_positions(self) -> slice:
    return slice(15, 15 + self.joints)

  @property
  def joint_velocities(self) -> slice:
    return slice(15 + self.joints, self.width)


def rot6d(quat: torch.Tensor) -> torch.Tensor:
  return matrix_from_quat(quat)[..., :, :2].transpose(-1, -2).flatten(-2)


def quat_from_rot6d(features: torch.Tensor) -> torch.Tensor:
  first = F.normalize(features[..., :3], dim=-1)
  first = torch.where(
    first.square().sum(-1, keepdim=True) < 1e-8,
    first.new_tensor([1.0, 0.0, 0.0]),
    first,
  )
  second = features[..., 3:6]
  second = second - (first * second).sum(-1, keepdim=True) * first
  fallback = torch.where(
    first[..., :1].abs() < 0.9,
    first.new_tensor([1.0, 0.0, 0.0]),
    first.new_tensor([0.0, 1.0, 0.0]),
  )
  fallback = fallback - (first * fallback).sum(-1, keepdim=True) * first
  second = F.normalize(
    torch.where(second.square().sum(-1, keepdim=True) < 1e-8, fallback, second),
    dim=-1,
  )
  return quat_from_matrix(
    torch.stack((first, second, torch.cross(first, second, dim=-1)), dim=-1)
  )


def encode(states: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
  """Encode dynamic states in A's heading frame."""
  if states.ndim != 3 or (states.shape[-1] - 13) % 2:
    raise ValueError("states must have shape (batch, time, 13 + 2 * joints)")
  if anchor.shape != states.shape[:1] + states.shape[2:]:
    raise ValueError("anchor must have shape (batch, state)")
  joints = (states.shape[-1] - 13) // 2
  heading = yaw_quat(anchor[:, 3:7])[:, None].expand(-1, states.shape[1], -1)
  origin = anchor[:, None, :3].clone()
  origin[..., 2] = 0.0
  orientation = quat_mul(quat_conjugate(heading), states[..., 3:7])
  return torch.cat(
    (
      quat_apply_inverse(heading, states[..., :3] - origin),
      rot6d(orientation),
      quat_apply_inverse(heading, states[..., 7:10]),
      quat_apply_inverse(heading, states[..., 10:13]),
      states[..., 13 : 13 + joints],
      states[..., 13 + joints :],
    ),
    dim=-1,
  )


def decode(
  features: torch.Tensor, anchor: torch.Tensor, layout: Layout
) -> torch.Tensor:
  """Decode features to world dynamic states."""
  if features.ndim != 3 or features.shape[-1] != layout.width:
    raise ValueError("features have the wrong shape")
  if anchor.shape != (features.shape[0], 13 + 2 * layout.joints):
    raise ValueError("anchor has the wrong shape")
  heading = yaw_quat(anchor[:, 3:7])[:, None].expand(-1, features.shape[1], -1)
  origin = anchor[:, None, :3].clone()
  origin[..., 2] = 0.0
  return torch.cat(
    (
      origin + quat_apply(heading, features[..., :3]),
      quat_mul(heading, quat_from_rot6d(features[..., 3:9])),
      quat_apply(heading, features[..., layout.root_linear_velocity]),
      quat_apply(heading, features[..., layout.root_angular_velocity]),
      features[..., layout.joint_positions],
      features[..., layout.joint_velocities],
    ),
    dim=-1,
  )


def perturb_start_states(
  states: torch.Tensor,
  duration: torch.Tensor,
  history: int,
  fps: float,
  xy_range: float,
  joint_range: float,
  probability: float,
) -> torch.Tensor:
  """Perturb A and blend exactly back to the demonstrated B state."""
  if states.ndim != 3 or (states.shape[-1] - 13) % 2:
    raise ValueError("states must have shape (batch, time, 13 + 2 * joints)")
  if duration.shape != states.shape[:1] or history < 1 or fps <= 0:
    raise ValueError("duration, history, and fps do not match the states")
  if history > states.shape[1] or bool(
    ((duration < 1) | (duration > states.shape[1] - history)).any()
  ):
    raise ValueError("duration must place B inside the state window")
  if min(xy_range, joint_range, probability) < 0 or probability > 1:
    raise ValueError("perturbation ranges and probability must be valid")
  if probability == 0 or (xy_range == 0 and joint_range == 0):
    return states

  batch, columns, width = states.shape
  joints = (width - 13) // 2
  active = (torch.rand(batch, 1, device=states.device) < probability).to(states.dtype)
  delta_xy = torch.empty(batch, 2, device=states.device, dtype=states.dtype).uniform_(
    -xy_range, xy_range
  )
  delta_joint = torch.empty(
    batch, joints, device=states.device, dtype=states.dtype
  ).uniform_(-joint_range, joint_range)
  delta_xy *= active
  delta_joint *= active

  frame = torch.arange(columns, device=states.device) - (history - 1)
  phase = (frame[None] / duration[:, None]).clamp(0.0, 1.0)
  weight = 1 - 3 * phase.square() + 2 * phase.pow(3)
  rate = (6 * phase.square() - 6 * phase) * fps / duration[:, None]

  out = states.clone()
  out[..., :2] += weight[..., None] * delta_xy[:, None]
  out[..., 7:9] += rate[..., None] * delta_xy[:, None]
  out[..., 13 : 13 + joints] += weight[..., None] * delta_joint[:, None]
  out[..., 13 + joints :] += rate[..., None] * delta_joint[:, None]
  return out


def mirror_g1_features(features: torch.Tensor) -> torch.Tensor:
  """Reflect local G1 features across the sagittal plane."""
  layout = Layout(29)
  if features.shape[-1] != layout.width:
    raise ValueError("G1 mirroring requires 29 joints")
  out = features.clone()
  out[..., (1, 4, 6, 8, 10, 12, 14)] *= -1
  indexes = torch.as_tensor(_G1_MIRROR_JOINTS, device=features.device)
  signs = features.new_tensor(_G1_MIRROR_SIGNS)
  out[..., layout.joint_positions] = (
    features[..., layout.joint_positions][..., indexes] * signs
  )
  out[..., layout.joint_velocities] = (
    features[..., layout.joint_velocities][..., indexes] * signs
  )
  return out


def _mirrored_joint(name: str) -> str:
  for left, right in (("left", "right"), ("Left", "Right")):
    if left in name:
      return name.replace(left, right, 1)
    if right in name:
      return name.replace(right, left, 1)
  return name


def mirror_features(
  features: torch.Tensor, joint_names: tuple[str, ...]
) -> torch.Tensor:
  """Reflect local robot features across the sagittal plane."""
  layout = Layout(len(joint_names))
  if features.shape[-1] != layout.width:
    raise ValueError("Feature width and robot joint names differ")
  by_name = {name: index for index, name in enumerate(joint_names)}
  try:
    indexes = [by_name[_mirrored_joint(name)] for name in joint_names]
  except KeyError as error:
    raise ValueError(f"No mirrored joint for {error.args[0]!r}") from None
  signs = [
    -1.0 if any(axis in name.lower() for axis in ("roll", "yaw")) else 1.0
    for name in joint_names
  ]
  out = features.clone()
  out[..., (1, 4, 6, 8, 10, 12, 14)] *= -1
  index = torch.as_tensor(indexes, device=features.device)
  sign = features.new_tensor(signs)
  out[..., layout.joint_positions] = (
    features[..., layout.joint_positions][..., index] * sign
  )
  out[..., layout.joint_velocities] = (
    features[..., layout.joint_velocities][..., index] * sign
  )
  return out


@dataclass
class Normalizer:
  mean: torch.Tensor
  std: torch.Tensor

  @classmethod
  def fit(cls, features: torch.Tensor) -> Normalizer:
    flat = features.flatten(0, 1)
    return cls(flat.mean(0), flat.std(0).clamp_min(1e-3))

  def normalize(self, features: torch.Tensor) -> torch.Tensor:
    return (features - self.mean) / self.std

  def denormalize(self, features: torch.Tensor) -> torch.Tensor:
    return features * self.std + self.mean


@dataclass(frozen=True)
class MotionCorpus:
  """Retargeted kinematic clips kept separate at window boundaries."""

  states: torch.Tensor
  starts: torch.Tensor
  trajectory: torch.Tensor
  frame: torch.Tensor
  names: tuple[str, ...]
  fps: float
  num_joints: int
  joint_names: tuple[str, ...] = ()
  robot: str = ""
  candidate_windows: int = 0
  rejected_windows: tuple[tuple[str, int], ...] = ()

  @property
  def num_windows(self) -> int:
    return int(self.starts.numel())

  def dataset(self) -> Dataset:
    """Expose the same BABEL clips as contiguous endpoint sequences."""
    return Dataset(
      states=self.states,
      skill=self.trajectory,
      trajectory=self.trajectory,
      frame=self.frame,
      names=self.names,
      fps=self.fps,
    )


def motion_files(patterns: tuple[str, ...]) -> tuple[Path, ...]:
  found = sorted(
    {Path(name) for pattern in patterns for name in glob.glob(pattern, recursive=True)}
  )
  if not found:
    raise FileNotFoundError(f"No motion files match {patterns}")
  return tuple(found)


@dataclass(frozen=True)
class _KinematicClips:
  states: torch.Tensor
  trajectory: torch.Tensor
  frame: torch.Tensor
  category: torch.Tensor
  categories: tuple[str, ...]
  names: tuple[str, ...]
  fps: float
  num_joints: int
  joint_names: tuple[str, ...]
  robot: str
  total_frames: int
  candidate_windows: int
  rejected_windows: tuple[tuple[str, int], ...]

  def dataset(self, categories: bool) -> Dataset:
    return Dataset(
      states=self.states,
      skill=self.category if categories else self.trajectory,
      trajectory=self.trajectory,
      frame=self.frame,
      names=self.categories if categories else self.names,
      fps=self.fps,
    )


def _load_kinematic_clips(
  patterns: tuple[str, ...],
  device: str,
  split: str,
  holdout: int,
  filter_cfg: MotionFilterCfg | None,
  columns: int | None = None,
  robot: str | None = None,
) -> _KinematicClips:
  """Load and validate kinematic clips before either consumer cuts windows."""
  if split not in ("train", "eval", "all"):
    raise ValueError("split must be train, eval, or all")
  if holdout < 2:
    raise ValueError("holdout must exceed one")
  if columns is not None and columns < 2:
    raise ValueError("columns must exceed one")
  files = motion_files(patterns)
  selected = (
    list(files)
    if split == "all"
    else [
      path
      for index, path in enumerate(files)
      if (index % holdout == 0) == (split == "eval")
    ]
  )
  if not selected:
    raise ValueError(f"No {split} clips remain after the file split")

  categories = tuple(sorted({path.parent.name for path in selected}))
  states: list[torch.Tensor] = []
  skills: list[torch.Tensor] = []
  trajectories: list[torch.Tensor] = []
  frames: list[torch.Tensor] = []
  names: list[str] = []
  fps: float | None = None
  joints: int | None = None
  clip_robot: str | None = None
  joint_names: tuple[str, ...] | None = None
  expected_robot = robot_name(robot) if robot is not None else None
  total_frames = candidate_windows = 0
  rejected_windows: dict[str, int] = {}
  for trajectory, path in enumerate(selected):
    with np.load(path, allow_pickle=False) as raw:
      clip_fps = float(np.asarray(raw["fps"]).reshape(-1)[0])
      joint_pos = np.asarray(raw["joint_pos"], dtype=np.float32)
      joint_vel = np.asarray(raw["joint_vel"], dtype=np.float32)
      body_pos = np.asarray(raw["body_pos_w"], dtype=np.float32)
      body_quat = np.asarray(raw["body_quat_w"], dtype=np.float32)
      body_lin_vel = np.asarray(raw["body_lin_vel_w"], dtype=np.float32)
      body_ang_vel = np.asarray(raw["body_ang_vel_w"], dtype=np.float32)
      current_robot = str(raw["robot"]) if "robot" in raw else ""
      current_joint_names = (
        tuple(str(name) for name in raw["joint_names"])
        if "joint_names" in raw
        else tuple(f"joint_{index}" for index in range(joint_pos.shape[1]))
      )
    if expected_robot is not None and current_robot != expected_robot:
      raise ValueError(
        f"{path} contains {current_robot or 'no robot metadata'}, expected {expected_robot}"
      )
    if clip_robot is not None and current_robot != clip_robot:
      raise ValueError("All motion clips must use the same robot")
    if joint_names is not None and current_joint_names != joint_names:
      raise ValueError("All motion clips must use the same joint order")
    if fps is not None and abs(clip_fps - fps) > 1e-6:
      raise ValueError("All motion clips must have the same frame rate")
    if joints is not None and joint_pos.shape[1] != joints:
      raise ValueError("All motion clips must use the same robot")
    fps = clip_fps
    joints = joint_pos.shape[1]
    clip_robot = current_robot
    joint_names = current_joint_names
    state = np.concatenate(
      (
        body_pos[:, 0],
        body_quat[:, 0],
        body_lin_vel[:, 0],
        body_ang_vel[:, 0],
        joint_pos,
        joint_vel,
      ),
      axis=-1,
    )
    total_frames += len(state)
    if columns is not None:
      candidate_windows += max(0, len(state) - columns + 1)

    valid = np.ones(len(state), dtype=bool)
    if filter_cfg is not None and len(state):
      bad_frames = motion_bad_frames(
        state,
        body_pos,
        body_quat,
        clip_fps,
        filter_cfg,
        current_robot or expected_robot or "unitree_g1",
      )
      valid &= ~np.logical_or.reduce(tuple(bad_frames.values()))
      if columns is not None and len(state) >= columns:
        for reason, bad in bad_frames.items():
          counts = np.concatenate(([0], np.cumsum(bad, dtype=np.int64)))
          rejected = int(((counts[columns:] - counts[:-columns]) > 0).sum())
          rejected_windows[reason] = rejected_windows.get(reason, 0) + rejected

    kept = np.flatnonzero(valid)
    count = len(kept)
    states.append(torch.from_numpy(state[kept]))
    skills.append(torch.full((count,), categories.index(path.parent.name)))
    trajectories.append(torch.full((count,), trajectory))
    frames.append(torch.from_numpy(kept))
    names.append(path.stem)

  if (
    fps is None
    or joints is None
    or joint_names is None
    or clip_robot is None
    or not states
    or not any(len(item) for item in states)
  ):
    raise ValueError(f"No valid {split} kinematic frames remain after motion filtering")
  return _KinematicClips(
    states=torch.cat(states).to(device),
    trajectory=torch.cat(trajectories).to(device),
    frame=torch.cat(frames).to(device),
    category=torch.cat(skills).to(device),
    categories=categories,
    names=tuple(names),
    fps=fps,
    num_joints=joints,
    joint_names=joint_names,
    robot=clip_robot,
    total_frames=total_frames,
    candidate_windows=candidate_windows,
    rejected_windows=tuple(sorted(rejected_windows.items())),
  )


def kinematic_segments(
  data: Dataset,
  min_steps: int,
  max_steps: int,
  sources: tuple[str, ...] | None = None,
  before: int = 0,
  after: int = 0,
) -> Segments:
  """Index valid contiguous kinematic windows for the planner or tracker."""
  return data.segments(
    min_steps,
    max_steps,
    data.of(sources),
    before=before,
    after=after,
  )


def load_motions(
  patterns: tuple[str, ...],
  columns: int,
  device: str,
  split: str = "train",
  holdout: int = 8,
  filter_cfg: MotionFilterCfg | None = DEFAULT_MOTION_FILTER,
  robot: str | None = None,
) -> MotionCorpus:
  """Load G1 NPZ clips and split by whole motion files."""
  loaded = _load_kinematic_clips(
    patterns, device, split, holdout, filter_cfg, columns, robot
  )
  data = loaded.dataset(categories=False)
  try:
    segments = kinematic_segments(data, columns - 1, columns - 1)
  except ValueError as error:
    raise ValueError(f"No {split} windows remain after motion filtering") from error
  return MotionCorpus(
    states=loaded.states,
    starts=segments.order[segments.starts],
    trajectory=loaded.trajectory,
    frame=loaded.frame,
    names=loaded.names,
    fps=loaded.fps,
    num_joints=loaded.num_joints,
    joint_names=loaded.joint_names,
    robot=loaded.robot,
    candidate_windows=loaded.candidate_windows,
    rejected_windows=loaded.rejected_windows,
  )


def load_kinematic_dataset(
  patterns: tuple[str, ...],
  device: str,
  filter_cfg: MotionFilterCfg | None = DEFAULT_MOTION_FILTER,
  robot: str | None = None,
) -> Dataset:
  """Load retargeted clips as trajectories for the universal tracker."""
  loaded = _load_kinematic_clips(patterns, device, "all", 8, filter_cfg, robot=robot)
  dataset = loaded.dataset(categories=True)
  print(
    f"[dataset] {dataset.states.shape[0]}/{loaded.total_frames} valid kinematic "
    f"states from {len(loaded.names)} clips"
  )
  return dataset


class Windows:
  """Sample A to B windows directly from kinematic clips."""

  def __init__(
    self,
    data: MotionCorpus,
    history: int,
    future: int,
    min_steps: int,
    max_steps: int,
    time_scale_range: tuple[float, float] = (1.0, 1.0),
    start_xy_range: float = 0.0,
    start_joint_range: float = 0.0,
    start_perturb_probability: float = 0.0,
    mirror_probability: float = 0.0,
  ) -> None:
    if history < 1 or future < 1 or min_steps < 2 or max_steps < min_steps:
      raise ValueError("invalid boundary or duration bounds")
    self.data = data
    self.history = history
    self.future = future
    self.min_steps = min_steps
    self.max_steps = max_steps
    low, high = time_scale_range
    if low <= 0 or high < low:
      raise ValueError("time_scale_range must be positive and ordered")
    if min(start_xy_range, start_joint_range, start_perturb_probability) < 0:
      raise ValueError("start perturbation values must be nonnegative")
    if start_perturb_probability > 1 or not 0 <= mirror_probability <= 1:
      raise ValueError("augmentation probabilities must lie in [0, 1]")
    if mirror_probability and data.num_joints != 29:
      if not data.joint_names:
        raise ValueError("mirroring requires robot joint names")
    self.time_scale_range = time_scale_range
    self.start_xy_range = start_xy_range
    self.start_joint_range = start_joint_range
    self.start_perturb_probability = start_perturb_probability
    self.mirror_probability = mirror_probability
    self.columns = history + max_steps + future - 1
    self.layout = Layout(data.num_joints)
    self.offsets = torch.arange(self.columns, device=data.states.device)

  def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local pose windows and A to B durations."""
    if count < 1:
      raise ValueError("count must be positive")
    device = self.data.states.device
    picked = torch.randint(self.data.starts.numel(), (count,), device=device)
    rows = self.data.starts[picked, None] + self.offsets
    states = self.data.states[rows]
    source_duration = torch.randint(
      self.min_steps, self.max_steps + 1, (count,), device=device
    )
    states = perturb_start_states(
      states,
      source_duration,
      self.history,
      self.data.fps,
      self.start_xy_range,
      self.start_joint_range,
      self.start_perturb_probability,
    )
    features = encode(states, states[:, self.history - 1])
    low, high = self.time_scale_range
    scale = torch.empty(count, device=device).uniform_(low, high)
    duration = (
      (source_duration.float() * scale)
      .round()
      .long()
      .clamp(self.min_steps, self.max_steps)
    )
    if bool((duration != source_duration).any()):
      features = rescale_bridge_features(
        features,
        source_duration,
        duration,
        self.layout,
        self.history,
        self.future,
      )
    if self.mirror_probability:
      mirrored = torch.rand(count, 1, 1, device=device) < self.mirror_probability
      reflected = (
        mirror_g1_features(features)
        if self.data.joint_names == () and self.data.num_joints == 29
        else mirror_features(features, self.data.joint_names)
      )
      features = torch.where(mirrored, reflected, features)
    target_last = self.history - 1 + duration + self.future - 1
    time = torch.arange(self.columns, device=device)[None]
    last = features[torch.arange(count, device=device), target_last]
    features = torch.where(
      (time > target_last[:, None])[..., None], last[:, None], features
    )
    return features, duration

  def states(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full state windows and durations for evaluation."""
    if count < 1:
      raise ValueError("count must be positive")
    device = self.data.states.device
    picked = torch.randint(self.data.starts.numel(), (count,), device=device)
    rows = self.data.starts[picked, None] + self.offsets
    states = self.data.states[rows]
    duration = torch.randint(
      self.min_steps, self.max_steps + 1, (count,), device=device
    )
    return states, duration


def rescale_bridge_features(
  features: torch.Tensor,
  source_duration: torch.Tensor,
  duration: torch.Tensor,
  layout: Layout,
  history: int,
  future: int,
) -> torch.Tensor:
  """Time-warp A to B while preserving both endpoint states exactly."""
  if source_duration.shape != duration.shape or features.shape[0] != duration.numel():
    raise ValueError("duration tensors must match the feature batch")
  out = features.clone()
  anchor = history - 1
  for row in range(features.shape[0]):
    old_steps = int(source_duration[row])
    new_steps = int(duration[row])
    old_end = anchor + old_steps
    new_end = anchor + new_steps
    source = features[row, anchor : old_end + 1]
    phase = torch.linspace(0.0, 1.0, new_steps + 1, device=features.device)
    stretch = new_steps / old_steps
    source_phase = stretch * phase + (1 - stretch) * (
      3 * phase.square() - 2 * phase.pow(3)
    )
    source_at = (source_phase * old_steps).clamp(0, old_steps)
    lower = source_at.floor().long()
    upper = source_at.ceil().long()
    blend = source_at.frac()[:, None]
    bridge = source[lower] * (1 - blend) + source[upper] * blend
    speed = (old_steps / new_steps) * (
      stretch + (1 - stretch) * (6 * phase - 6 * phase.square())
    )
    bridge[:, layout.root_linear_velocity] *= speed[:, None]
    bridge[:, layout.root_angular_velocity] *= speed[:, None]
    bridge[:, layout.joint_velocities] *= speed[:, None]
    out[row, anchor : new_end + 1] = bridge
    if future > 1:
      out[row, new_end + 1 : new_end + future] = features[
        row, old_end + 1 : old_end + future
      ]
    final = out[row, new_end + future - 1]
    out[row, new_end + future :] = final
  return out


def bridge_mask(
  batch: int,
  columns: int,
  layout: Layout,
  history: int,
  future: int,
  duration: torch.Tensor,
) -> torch.Tensor:
  """Condition on pre A history and B's short continuation."""
  if duration.shape != (batch,) or bool(
    ((duration < 2) | (duration > columns - history - future + 1)).any()
  ):
    raise ValueError("duration must fit the prediction horizon")
  mask = torch.zeros(
    batch, columns, layout.width, dtype=torch.bool, device=duration.device
  )
  mask[:, :history] = True
  rows = history - 1 + duration
  offsets = torch.arange(future, device=duration.device)
  indexes = torch.arange(batch, device=duration.device)[:, None]
  mask[indexes, rows[:, None] + offsets] = True
  return mask


def training_mask(
  layout: Layout,
  columns: int,
  history: int,
  future: int,
  duration: torch.Tensor,
  bridge_probability: float = 1.0,
) -> torch.Tensor:
  """Use the deployment mask, with optional generic inpainting examples."""
  if not 0.0 <= bridge_probability <= 1.0:
    raise ValueError("bridge_probability must lie in [0, 1]")
  batch = duration.shape[0]
  device = duration.device
  mask = torch.rand(batch, columns, layout.width, device=device) < 0.25
  temporal = torch.rand(batch, device=device) < 0.5
  if bool(temporal.any()):
    frames = torch.rand(int(temporal.sum()), columns, 1, device=device) < 0.25
    mask[temporal] = frames.expand(-1, -1, layout.width)
  mask[:, 0] = True
  mask[:, -1] = True
  deployment = torch.rand(batch, device=device) < bridge_probability
  if bool(deployment.any()):
    fixed = bridge_mask(batch, columns, layout, history, future, duration)
    mask[deployment] = fixed[deployment]
  return mask
