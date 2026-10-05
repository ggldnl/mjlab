"""Footstep channels for the footstep diffusion planner.

A footstep is one stance run of one foot: the frames where its sole is planted, the
position it is planted at and its yaw. Every frame of a window carries six channels
per foot, appended after the diffusion planner's condition channels:

    contact      1 on stance frames, 0 on swing frames
    position     planted sole position in A's heading frame, zero on swing frames
    yaw          planted sole yaw as cos and sin, zero on swing frames

Training reads footsteps from the clip, inference gets them from the heuristic
planner in this module. Swing trajectories are never given: the diffusion model
generates them. The known mask decides what the model sees:

    contact known, position known     planted here, at this spot
    contact known, position unknown   planted here, anywhere
    contact unknown                   no opinion

Contact is a sole below a height threshold moving slower than a speed threshold.
Stance runs shorter than min_stance frames and swing gaps shorter than min_swing
frames are treated as noise and removed.
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from mjlab.tasks.bridging.bridges.diffusion.dataset.motions import (
  Layout,
  bridge_mask,
  encode_pose,
  quat_from_rot6d,
)
from mjlab.tasks.bridging.bridges.diffusion.planner.process import (
  RobotFootKinematics,
  pose_states,
)
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_from_angle_axis,
  quat_mul,
  yaw_quat,
)

FOOT_CHANNELS = 6
"""Contact, position xyz, cos yaw, sin yaw."""


@dataclass(frozen=True)
class FootstepLayout(Layout):
  """Diffusion planner layout plus FOOT_CHANNELS condition channels per foot."""

  @property
  def width(self) -> int:
    return 12 + 2 * self.joints + 2 * FOOT_CHANNELS

  @property
  def joint_positions(self) -> slice:
    return slice(12 + self.joints, 12 + 2 * self.joints)

  @property
  def feet(self) -> slice:
    start = 12 + 2 * self.joints
    return slice(start, start + 2 * FOOT_CHANNELS)

  @property
  def contact_channels(self) -> list[int]:
    start = self.feet.start
    return [start, start + FOOT_CHANNELS]

  @property
  def place_channels(self) -> list[int]:
    start = self.feet.start
    return [
      start + foot * FOOT_CHANNELS + offset
      for foot in range(2)
      for offset in range(1, FOOT_CHANNELS)
    ]


@dataclass(frozen=True)
class ContactCfg:
  height: float = 0.05
  speed: float = 0.25
  min_stance: int = 3
  min_swing: int = 2


Table = tuple[tuple[float, ...], tuple[float, ...]]


@dataclass(frozen=True)
class Gait:
  """Corpus statistics the heuristic planner uses for spacing and timing.

  The tables hold one median per root speed in speeds, first row for steady or
  speeding up motion, second row for braking harder than brake m/s^2:

      stride     distance between two consecutive plants of the same foot
      interval   ticks between two consecutive landings, one per foot
      swing      ticks a foot spends in the air
      lead       how far ahead of the root, along its velocity, a foot lands

  step_length and swing_ticks are the overall medians, kept for reporting and
  for the minimum swing length. max_turn is the largest turn a planted foot makes
  by pivoting: in the corpus a foot that turns without moving steps only above
  about 1.5 rad.
  """

  step_length: float = 0.6
  swing_ticks: float = 18.0
  stance_height: float = 0.03
  max_turn: float = 1.5
  speeds: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.5)
  stride: Table = ((0.3, 0.5, 0.75, 1.0, 1.4), (0.3, 0.45, 0.65, 0.85, 1.2))
  interval: Table = ((30, 28, 25, 22, 17), (26, 24, 21, 18, 15))
  swing: Table = ((20, 20, 19, 17, 15), (18, 18, 17, 15, 13))
  lead: Table = ((0.0, 0.1, 0.18, 0.25, 0.3), (0.05, 0.15, 0.25, 0.32, 0.38))
  brake: float = 0.5

  def to_dict(self) -> dict:
    return asdict(self)

  @classmethod
  def from_dict(cls, values: dict[str, Any]) -> Gait:
    """A Gait saved with to_dict. Checkpoints may hold its tuples as lists."""
    fields: dict[str, Any] = {}
    for name, value in values.items():
      if name == "speeds":
        value = tuple(float(item) for item in value)
      elif name in ("stride", "interval", "swing", "lead"):
        value = tuple(tuple(float(item) for item in row) for row in value)
      fields[name] = value
    return cls(**fields)

  def lookup(self, table: Table, speed: float, braking: bool) -> float:
    return float(np.interp(speed, self.speeds, table[int(braking)]))


def yaw_of(quat: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def sole_frames(
  feet: RobotFootKinematics, pose: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Sole positions (..., 2, 3) and yaws (..., 2) of poses in A's heading frame."""
  position, orientation = feet.frames(pose_states(pose))
  return position, yaw_of(orientation)


def rewind(states: torch.Tensor, fps: float) -> torch.Tensor:
  """Dynamic states one tick earlier, stepped back along their own velocities."""
  joints = (states.shape[-1] - 13) // 2
  dt = 1.0 / fps
  out = states.clone()
  out[..., :3] -= dt * states[..., 7:10]
  spin = states[..., 10:13]
  angle = spin.norm(dim=-1)
  axis = spin / angle.clamp_min(1e-9)[..., None]
  back = quat_from_angle_axis(-angle * dt, axis)
  out[..., 3:7] = quat_mul(back, states[..., 3:7])
  out[..., 13 : 13 + joints] -= dt * states[..., 13 + joints :]
  return out


def run_ids(flag: torch.Tensor) -> torch.Tensor:
  """Index of the constant run each frame belongs to, along dim 1."""
  change = flag[:, 1:] != flag[:, :-1]
  first = torch.zeros_like(flag[:, :1], dtype=torch.long)
  return torch.cat((first, change.long().cumsum(1)), dim=1)


def _run_sum(values: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
  """Sum values (B, T, F, C) over each run of ids (B, T, F), gathered back per frame."""
  sums = torch.zeros_like(values).scatter_add_(
    1, ids[..., None].expand_as(values), values
  )
  return sums.gather(1, ids[..., None].expand_as(values))


def _drop_short(flag: torch.Tensor, value: bool, minimum: int) -> torch.Tensor:
  """Flip runs equal to value shorter than minimum, unless they touch a window edge."""
  if minimum <= 1:
    return flag
  ids = run_ids(flag)
  length = _run_sum(torch.ones_like(flag, dtype=torch.float32)[..., None], ids)[..., 0]
  inner = (ids != ids[:, :1]) & (ids != ids[:, -1:])
  short = (flag == value) & (length < minimum) & inner
  return torch.where(short, ~flag, flag)


def detect_contacts(
  position: torch.Tensor, fps: float, cfg: ContactCfg, speed: bool = True
) -> torch.Tensor:
  """Planted flags (B, T, 2) from sole positions (B, T, 2, 3).

  Without speed, or with a single frame, contact is decided by height alone.
  """
  low = position[..., 2] <= cfg.height
  if speed and position.shape[1] > 1:
    ahead = torch.cat((position[:, 1:], position[:, -1:]), dim=1)
    behind = torch.cat((position[:, :1], position[:, :-1]), dim=1)
    span = torch.full((position.shape[1],), 2.0, device=position.device)
    span[0] = span[-1] = 1.0
    rate = (ahead - behind)[..., :2].norm(dim=-1) * fps / span[None, :, None]
    low = low & (rate <= cfg.speed)
  contact = _drop_short(low, False, cfg.min_swing)
  return _drop_short(contact, True, cfg.min_stance)


def footstep_channels(
  position: torch.Tensor, yaw: torch.Tensor, contact: torch.Tensor
) -> torch.Tensor:
  """Per frame channels (B, T, 2, 6) holding the mean plant of each stance run."""
  weight = contact.float()[..., None]
  ids = run_ids(contact)
  values = torch.cat((position, yaw.cos()[..., None], yaw.sin()[..., None]), dim=-1)
  total = _run_sum(values * weight, ids)
  count = _run_sum(weight, ids).clamp_min(1.0)
  mean = total / count
  heading = mean[..., 3:5]
  heading = heading / heading.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  place = torch.cat((mean[..., :3], heading), dim=-1) * weight
  return torch.cat((weight, place), dim=-1)


def extract(
  feet: RobotFootKinematics,
  pose: torch.Tensor,
  fps: float,
  cfg: ContactCfg,
  speed: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Contacts (B, T, 2), channels (B, T, 2, 6) and soles (B, T, 2, 3) of a pose window."""
  position, yaw = sole_frames(feet, pose)
  contact = detect_contacts(position, fps, cfg, speed)
  return contact, footstep_channels(position, yaw, contact), position


def foot_values(layout: FootstepLayout, values: torch.Tensor) -> torch.Tensor:
  """The (..., 2, 6) foot channels of planner features."""
  return values[..., layout.feet].unflatten(-1, (2, FOOT_CHANNELS))


@dataclass(frozen=True)
class MaskCfg:
  """How often each condition group is shown during training.

  Each window picks one footstep mode: all footsteps, a random subset of runs, or
  none. Independently, positions can be hidden while timing stays, B can keep only
  its root, and a few root keyframes can be revealed.
  """

  full_probability: float = 0.4
  none_probability: float = 0.2
  keep_run_probability: float = 0.5
  timing_only_probability: float = 0.1
  target_root_only_probability: float = 0.1
  keyframe_probability: float = 0.1
  keyframes: int = 2
  position_noise: float = 0.015
  yaw_noise: float = 0.05
  timing_jitter_probability: float = 0.3

  def __post_init__(self) -> None:
    probabilities = (
      self.full_probability,
      self.none_probability,
      self.keep_run_probability,
      self.timing_only_probability,
      self.target_root_only_probability,
      self.keyframe_probability,
      self.timing_jitter_probability,
    )
    if not all(0.0 <= value <= 1.0 for value in probabilities):
      raise ValueError("mask probabilities must lie in [0, 1]")
    if self.full_probability + self.none_probability > 1.0:
      raise ValueError("full and none footstep modes cannot exceed probability 1")
    if min(self.position_noise, self.yaw_noise) < 0 or self.keyframes < 0:
      raise ValueError("noise and keyframe counts cannot be negative")


def jitter_timing(
  contact: torch.Tensor, anchor: int, probability: float
) -> torch.Tensor:
  """Grow or shrink stance runs by one frame after A, per window and foot."""
  if probability == 0:
    return contact
  batch, columns, feet = contact.shape
  device = contact.device
  picked = torch.rand(batch, 1, feet, device=device) < probability
  grow = torch.rand(batch, 1, feet, device=device) < 0.5
  ahead = torch.cat((contact[:, 1:], contact[:, -1:]), dim=1)
  behind = torch.cat((contact[:, :1], contact[:, :-1]), dim=1)
  grown = contact | ahead | behind
  shrunk = contact & ahead & behind
  jittered = torch.where(grow, grown, shrunk)
  after = (torch.arange(columns, device=device) > anchor)[None, :, None]
  return torch.where(picked & after, jittered, contact)


def perturb_places(
  channels: torch.Tensor,
  contact: torch.Tensor,
  position_noise: float,
  yaw_noise: float,
) -> torch.Tensor:
  """Shift every footstep by one random offset in xy and yaw."""
  if position_noise == 0 and yaw_noise == 0:
    return channels
  ids = run_ids(contact)
  batch, columns, feet = contact.shape
  noise = torch.randn(batch, columns, feet, 3, device=channels.device)
  noise = noise.gather(1, ids[..., None].expand(-1, -1, -1, 3))
  out = channels.clone()
  out[..., 1:3] += position_noise * noise[..., :2]
  turn = yaw_noise * noise[..., 2]
  cos, sin = channels[..., 4], channels[..., 5]
  out[..., 4] = cos * turn.cos() - sin * turn.sin()
  out[..., 5] = sin * turn.cos() + cos * turn.sin()
  return out * contact.float()[..., None]


def deployment_mask(
  layout: FootstepLayout,
  columns: int,
  history: int,
  future: int,
  duration: torch.Tensor,
  contact: torch.Tensor,
  feet_known: torch.Tensor,
) -> torch.Tensor:
  """The inference mask: A history, B with its continuation, and given footsteps.

  feet_known (B, T, 2) marks frames whose footstep channels are given. Places are
  only known on frames that are also planted.
  """
  mask = bridge_mask(duration.shape[0], columns, layout, history, future, duration)
  last = history - 1 + duration + future - 1
  inside = torch.arange(columns, device=duration.device)[None] <= last[:, None]
  timing = feet_known & inside[..., None]
  place = timing & contact
  starts = layout.contact_channels
  for foot in range(2):
    mask[..., starts[foot]] = timing[..., foot]
    begin = starts[foot] + 1
    mask[..., begin : begin + FOOT_CHANNELS - 1] = place[..., foot, None]
  return mask


def training_mask(
  layout: FootstepLayout,
  columns: int,
  history: int,
  future: int,
  duration: torch.Tensor,
  contact: torch.Tensor,
  cfg: MaskCfg,
) -> torch.Tensor:
  """Sample one MaskedMimic style mask per window, see MaskCfg."""
  batch = duration.shape[0]
  device = duration.device
  mode = torch.rand(batch, device=device)
  full = mode < cfg.full_probability
  none = mode >= 1.0 - cfg.none_probability
  ids = run_ids(contact)
  keep_run = torch.rand(batch, columns, 2, device=device) < cfg.keep_run_probability
  keep = keep_run.gather(1, ids)
  feet_known = torch.where(full[:, None, None], True, keep) & ~none[:, None, None]
  mask = deployment_mask(
    layout, columns, history, future, duration, contact, feet_known
  )

  timing_only = torch.rand(batch, device=device) < cfg.timing_only_probability
  places = torch.tensor(layout.place_channels, device=device)
  mask[..., places] &= ~timing_only[:, None, None]

  root_only = torch.rand(batch, device=device) < cfg.target_root_only_probability
  rows = history - 1 + duration
  time = torch.arange(columns, device=device)[None]
  after_b = (time >= rows[:, None]) & root_only[:, None]
  mask[..., layout.joint_positions] &= ~after_b[..., None]
  mask[..., layout.joint_steps] &= ~after_b[..., None]

  revealed = torch.rand(batch, device=device) < cfg.keyframe_probability
  if cfg.keyframes and bool(revealed.any()):
    picked = (
      history
      + (
        torch.rand(batch, cfg.keyframes, device=device)
        * (duration[:, None] - 1).float()
      ).long()
    )
    hit = torch.zeros(batch, columns, dtype=torch.bool, device=device)
    hit.scatter_(1, picked, True)
    hit &= revealed[:, None]
    mask[..., layout.root_position] |= hit[..., None]
  return mask


def _medians(
  speed: np.ndarray,
  braking: np.ndarray,
  values: np.ndarray,
  edges: np.ndarray,
  fallback: tuple[float, ...],
  minimum: int,
) -> Table:
  """Median of values per speed bin and regime, empty bins interpolated."""
  centers = 0.5 * (edges[1:] + edges[:-1])
  rows = []
  for regime in (False, True):
    picked = braking == regime
    medians = np.full(centers.size, np.nan)
    for index in range(centers.size):
      inside = picked & (speed >= edges[index]) & (speed < edges[index + 1])
      if inside.sum() >= minimum:
        medians[index] = np.median(values[inside])
    rows.append(medians)
  for index, row in enumerate(rows):
    valid = ~np.isnan(row)
    if not valid.any():
      other = rows[1 - index]
      if (~np.isnan(other)).any():
        rows[index] = other.copy()
      else:
        rows[index] = np.interp(centers, np.linspace(0, 2.5, len(fallback)), fallback)
  for row in rows:
    valid = ~np.isnan(row)
    row[~valid] = np.interp(centers[~valid], centers[valid], row[valid])
  return tuple(float(value) for value in rows[0]), tuple(
    float(value) for value in rows[1]
  )


def fit_gait(
  contact: torch.Tensor,
  channels: torch.Tensor,
  root: torch.Tensor,
  fps: float,
  max_turn: float = 1.5,
  bin_width: float = 0.25,
  max_speed: float = 3.0,
  minimum: int = 30,
) -> Gait:
  """Fit the gait tables from sampled windows.

  Every landing inside a window is one sample, labelled with the root speed and the
  braking flag at that tick. Only swings and intervals that start and end inside
  the window count, and only steps moving the foot more than 5 cm count toward
  step_length. Lead is only sampled above 0.1 m/s, where the root has a direction.
  """
  defaults = Gait()
  contact_np = contact.cpu().numpy()
  plants = channels[..., 1:4].cpu().numpy()
  position = root.cpu().double().numpy()
  velocity = np.gradient(position, axis=1) * fps
  acceleration = np.gradient(velocity, axis=1) * fps
  speeds, brakes, intervals, swings, leads, lead_speeds, lead_brakes = (
    [],
    [],
    [],
    [],
    [],
    [],
    [],
  )
  strides: list[float] = []
  stride_speeds: list[float] = []
  stride_brakes: list[bool] = []
  for row in range(contact_np.shape[0]):
    landings = []
    for foot in range(2):
      flag = contact_np[row, :, foot]
      edges = np.flatnonzero(np.diff(flag.astype(np.int8))) + 1
      previous = plants[row, 0, foot] if flag[0] else None  # the stance at the start
      lift = None
      for tick in edges:
        if not flag[tick]:
          lift = tick
          continue
        landings.append(tick)
        here = plants[row, tick, foot]
        v = velocity[row, tick]
        speed = float(np.linalg.norm(v))
        along = float(acceleration[row, tick] @ v) / max(speed, 1e-6)
        braking = along < -defaults.brake
        if previous is not None:
          distance = float(np.linalg.norm(here[:2] - previous[:2]))
          if distance > 0.05:
            strides.append(distance)
            stride_speeds.append(speed)
            stride_brakes.append(braking)
        previous = here
        if lift is not None:
          swings.append(tick - lift)
          speeds.append(speed)
          brakes.append(braking)
        if speed > 0.1:
          leads.append(float((here[:2] - position[row, tick]) @ v) / speed)
          lead_speeds.append(speed)
          lead_brakes.append(braking)
      lift = None
    landings.sort()
    for before, after in zip(landings[:-1], landings[1:], strict=True):
      if after > before:
        v = velocity[row, after]
        speed = float(np.linalg.norm(v))
        along = float(acceleration[row, after] @ v) / max(speed, 1e-6)
        intervals.append((after - before, speed, along < -defaults.brake))
  heights = plants[..., 2][contact_np]
  if not strides or not swings or not intervals or heights.size == 0:
    raise ValueError("The sampled windows hold no complete steps to fit a gait")
  edges = np.arange(0.0, max_speed + bin_width, bin_width)
  centers = 0.5 * (edges[1:] + edges[:-1])
  interval_values = np.array(intervals)
  return Gait(
    step_length=float(np.median(strides)),
    swing_ticks=float(np.median(swings)),
    stance_height=float(np.median(heights)),
    max_turn=max_turn,
    speeds=tuple(float(value) for value in centers),
    stride=_medians(
      np.array(stride_speeds),
      np.array(stride_brakes),
      np.array(strides),
      edges,
      defaults.stride[0],
      minimum,
    ),
    interval=_medians(
      interval_values[:, 1],
      interval_values[:, 2].astype(bool),
      interval_values[:, 0],
      edges,
      defaults.interval[0],
      minimum,
    ),
    swing=_medians(
      np.array(speeds),
      np.array(brakes),
      np.array(swings, dtype=float),
      edges,
      defaults.swing[0],
      minimum,
    ),
    lead=_medians(
      np.array(lead_speeds),
      np.array(lead_brakes),
      np.array(leads),
      edges,
      defaults.lead[0],
      minimum,
    ),
  )


def _wrap(angle: float) -> float:
  return (angle + math.pi) % (2 * math.pi) - math.pi


def _min_swing(gait: Gait) -> int:
  return max(2, round(0.5 * gait.swing_ticks))


@dataclass(frozen=True)
class Boundary:
  """Root and feet at A or B, in A's heading frame."""

  root: np.ndarray
  """xy position."""

  velocity: np.ndarray
  """xy velocity in m/s."""

  yaw: float
  rate: float
  """Yaw rate in rad/s."""

  place: np.ndarray
  """(2, 3) sole positions."""

  foot_yaw: np.ndarray
  """(2,) sole yaws."""

  contact: np.ndarray
  """(2,) planted flags."""


@dataclass(frozen=True)
class RootPath:
  """Root xy and yaw from A to B per tick, a cubic Hermite through both boundaries."""

  position: np.ndarray
  velocity: np.ndarray
  yaw: np.ndarray
  braking: np.ndarray

  @property
  def speed(self) -> np.ndarray:
    return np.linalg.norm(self.velocity, axis=-1)


def root_path(
  start: Boundary, end: Boundary, duration: int, fps: float, brake: float
) -> RootPath:
  """Hermite curve matching position and velocity at A and B.

  A fast A and a still B gives a decelerating root, a still A and a fast B an
  accelerating one. Braking marks ticks decelerating harder than brake m/s^2.
  """
  span = duration / fps
  s = np.arange(duration + 1)[:, None] / duration
  basis = np.concatenate(
    (2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2),
    axis=1,
  )
  first = np.concatenate(
    (6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1, -6 * s**2 + 6 * s, 3 * s**2 - 2 * s),
    axis=1,
  )
  second = np.concatenate((12 * s - 6, 6 * s - 4, -12 * s + 6, 6 * s - 2), axis=1)

  def curve(p0, v0, p1, v1):
    knots = np.stack((p0, span * v0, p1, span * v1))
    return basis @ knots, first @ knots / span, second @ knots / span**2

  position, velocity, acceleration = curve(
    start.root, start.velocity, end.root, end.velocity
  )
  end_yaw = start.yaw + _wrap(end.yaw - start.yaw)
  yaw, _, _ = curve(
    np.array([start.yaw]),
    np.array([start.rate]),
    np.array([end_yaw]),
    np.array([end.rate]),
  )
  speed = np.linalg.norm(velocity, axis=-1)
  along = (acceleration * velocity).sum(-1) / np.maximum(speed, 1e-6)
  return RootPath(position, velocity, yaw[:, 0], along < -brake)


def _lateral(boundary: Boundary, foot: int) -> tuple[float, float]:
  """Sideways offset and relative yaw of a planted foot in the root's frame."""
  cos, sin = math.cos(boundary.yaw), math.sin(boundary.yaw)
  delta = boundary.place[foot, :2] - boundary.root
  side = -sin * delta[0] + cos * delta[1]
  return side, _wrap(float(boundary.foot_yaw[foot]) - boundary.yaw)


def _offsets(start: Boundary, end: Boundary) -> list[tuple[tuple, tuple]]:
  """Offsets at A and B per foot, borrowed from the other end when lifted there."""
  out = []
  for foot in range(2):
    default = (0.1 if foot == 0 else -0.1, 0.0)
    at_start = _lateral(start, foot) if start.contact[foot] else None
    at_end = _lateral(end, foot) if end.contact[foot] else None
    at_start = at_start or at_end or default
    at_end = at_end or at_start
    out.append((at_start, at_end))
  return out


def _landing(
  path: RootPath,
  offsets: list[tuple[tuple, tuple]],
  foot: int,
  land: int,
  duration: int,
  gait: Gait,
) -> tuple[np.ndarray, float]:
  """Plant and yaw of a landing: root path, stance offset and corpus lead."""
  fraction = land / duration
  (side_a, turn_a), (side_b, turn_b) = offsets[foot]
  side = (1 - fraction) * side_a + fraction * side_b
  yaw = float(path.yaw[land])
  velocity = path.velocity[land]
  speed = float(np.linalg.norm(velocity))
  lead = (
    gait.lookup(gait.lead, speed, bool(path.braking[land])) * velocity / speed
    if speed > 0.1
    else np.zeros(2)
  )
  xy = path.position[land] + side * np.array((-math.sin(yaw), math.cos(yaw))) + lead
  turn = (1 - fraction) * turn_a + fraction * turn_b
  return np.array((xy[0], xy[1], gait.stance_height)), yaw + turn


def schedule(
  path: RootPath, start: Boundary, end: Boundary, duration: int, gait: Gait
) -> list[tuple[int, int, int]]:
  """Swings as (foot, lift tick, landing tick), landing after B for a lifted foot.

  How many steps: a foot that moves more than 4 cm or turns more than max_turn
  covers its distance in corpus strides at the peak root speed, at least one. A
  foot that starts and ends in place does not step, so a root swaying on the spot
  plans no steps. In the corpus 97% of planted feet moving less than 4 cm never
  step, and those moving more step once per stride.

  When: landings alternate between the feet while both have steps left, spaced by
  the corpus interval at the root speed of the moment, each swing lasting the
  corpus swing time at that speed. So a braking root gets the corpus braking
  rhythm, a fast root quick steps. Steps that do not fit before B are compressed
  in time. A foot lifted at A lands first, a foot lifted at B swings through B,
  lifting once the other foot has landed its last step.
  """
  speed = path.speed
  needs = [
    not start.contact[foot]
    or not end.contact[foot]
    or np.linalg.norm(end.place[foot, :2] - start.place[foot, :2]) > 0.04
    or abs(_wrap(float(end.foot_yaw[foot] - start.foot_yaw[foot]))) > gait.max_turn
    for foot in range(2)
  ]
  if not any(needs):
    return []

  def at(table: Table, tick: int) -> int:
    tick = min(max(tick, 0), duration)
    return round(gait.lookup(table, float(speed[tick]), bool(path.braking[tick])))

  shortest = _min_swing(gait)
  lifted = [not start.contact[foot] for foot in range(2)]
  hover = [
    lifted[foot] and not end.contact[foot] and speed.max() < 0.5 for foot in range(2)
  ]
  if any(hover):
    # a balance or kick: the lifted foot stays up, the other steps once if it must
    swings = [(foot, 0, duration + 1) for foot in range(2) if hover[foot]]
    for foot in range(2):
      if not hover[foot] and needs[foot]:
        lift = max(1, (duration - shortest) // 2)
        swings.append((foot, lift, min(duration, lift + shortest)))
    return swings

  stride = gait.lookup(gait.stride, float(speed.max()), bool(path.braking.mean() > 0.5))
  counts = []
  for foot in range(2):
    if not needs[foot]:
      counts.append(0)
      continue
    distance = float(np.linalg.norm(end.place[foot, :2] - start.place[foot, :2]))
    planted = bool(start.contact[foot] and end.contact[foot])
    count = round(distance / stride)
    if not end.contact[foot]:
      count -= 1  # the last stretch is the swing through B
    least = int(planted or lifted[foot])  # a lifted foot that is not hovering lands
    counts.append(max(count, least))

  if lifted[0] != lifted[1]:
    foot = 0 if lifted[0] else 1
  elif speed[0] > 0.2:
    heading = start.velocity / max(float(np.linalg.norm(start.velocity)), 1e-6)
    foot = int(np.argmin(start.place[:, :2] @ heading))  # the trailing foot
  else:
    foot = int(np.argmax(counts))
  sequence = []
  left = list(counts)
  while sum(left):
    if not left[foot]:
      foot = 1 - foot
    sequence.append(foot)
    left[foot] -= 1
    foot = 1 - foot

  swings: list[tuple[int, int, int]] = []
  last_land = [0, 0]
  for foot in sequence:
    if not swings:
      if lifted[foot]:
        lift, land = 0, max(2, at(gait.swing, 0) // 2)
      else:
        lift = 1 if speed[0] > 0.2 else max(1, at(gait.interval, 0) // 2)
        land = lift + max(shortest, at(gait.swing, lift))
    else:
      previous = swings[-1][2]
      gap = max(3, at(gait.interval, previous))
      if foot == swings[-1][0]:
        gap = max(gap, shortest + 2)
      land = previous + gap
      if lifted[foot] and not last_land[foot]:
        lift = 0
      else:
        lift = land - max(shortest, at(gait.swing, previous))
        lift = max(lift, last_land[foot] + 2, 1)
      land = max(land, lift + 2)
    swings.append((foot, lift, land))
    last_land[foot] = land
  if swings and swings[-1][2] > duration:
    origin = min(1, swings[0][1])
    scale = (duration - origin) / (swings[-1][2] - origin)
    swings = [
      (
        foot,
        round(origin + (lift - origin) * scale),
        round(origin + (land - origin) * scale),
      )
      for foot, lift, land in swings
    ]
    swings = [(foot, lift, max(land, lift + 1)) for foot, lift, land in swings]

  for foot in range(2):
    if end.contact[foot]:
      continue
    own = [entry for entry in swings if entry[0] == foot]
    if lifted[foot] and not own:
      swings.append((foot, 0, duration + 1))
      continue
    other = [entry[2] for entry in swings if entry[0] != foot and entry[2] < duration]
    begin = max(own[-1][2] + 2 if own else 1, max(other, default=0))
    lift = max(begin, duration - max(shortest, at(gait.swing, duration)))
    swings.append((foot, min(lift, duration), duration + 1))
  return sorted(swings, key=lambda entry: entry[2])


def plan_window(
  start: Boundary, end: Boundary, duration: int, fps: float, gait: Gait
) -> tuple[np.ndarray, np.ndarray]:
  """Contacts (duration + 1, 2) and channels (duration + 1, 2, 6) from A to B.

  Timing comes from schedule. A foot lands where the root path will be at
  touchdown, moved sideways by the foot's stance offset and ahead along the root
  velocity by the corpus lead at that speed. The last landing of a foot planted at
  B is B's own plant.
  """
  path = root_path(start, end, duration, fps, gait.brake)
  swings = schedule(path, start, end, duration, gait)
  offsets = _offsets(start, end)
  contact = np.ones((duration + 1, 2), dtype=bool)
  channels = np.zeros((duration + 1, 2, FOOT_CHANNELS))
  stances: list[list[tuple[int, np.ndarray, float]]] = []
  for foot in range(2):
    place = start.place[foot].copy()
    if not start.contact[foot]:
      place[2] = gait.stance_height
    stances.append([(0, place, float(start.foot_yaw[foot]))])
  final = {
    foot: max(e[2] for e in swings if e[0] == foot) for foot in {e[0] for e in swings}
  }
  for foot, lift, land in swings:
    contact[lift:land, foot] = False
    if land > duration:
      continue
    if land == final[foot] and end.contact[foot]:
      stances[foot].append((land, end.place[foot].copy(), float(end.foot_yaw[foot])))
      continue
    place, yaw = _landing(path, offsets, foot, land, duration, gait)
    stances[foot].append((land, place, yaw))
  for foot in range(2):
    if not start.contact[foot]:
      contact[0, foot] = False
    if not end.contact[foot]:
      contact[duration, foot] = False
    ordered = sorted(stances[foot], key=lambda entry: entry[0])
    for (begin, place, yaw), following in zip(
      ordered, [*ordered[1:], (duration + 1, None, 0.0)], strict=True
    ):
      channels[begin : following[0], foot, 1:4] = place
      channels[begin : following[0], foot, 4] = math.cos(yaw)
      channels[begin : following[0], foot, 5] = math.sin(yaw)
  channels[..., 0] = contact
  channels *= contact[..., None]
  return contact, channels


class FootstepPlanner:
  """Heuristic footsteps between a known history and a known B continuation.

  History and continuation frames get the footsteps read from their own poses. The
  frames between A and B come from plan_window.
  """

  def __init__(
    self,
    feet: RobotFootKinematics,
    gait: Gait,
    contact: ContactCfg,
    fps: float,
  ) -> None:
    self.feet = feet
    self.gait = gait
    self.contact = contact
    self.fps = fps

  def boundaries(
    self,
    states: torch.Tensor,
    pose: torch.Tensor,
    heading: torch.Tensor,
    planted: torch.Tensor,
  ) -> list[Boundary]:
    """One Boundary per row of dynamic states and their poses in A's heading frame."""
    velocity = quat_apply_inverse(heading, states[:, 7:10])[:, :2]
    yaw = yaw_of(quat_from_rot6d(pose[:, 3:9]))
    place, foot_yaw = sole_frames(self.feet, pose)
    values = [
      tensor.cpu().double().numpy()
      for tensor in (pose[:, :2], velocity, yaw, states[:, 12], place, foot_yaw)
    ]
    flags = planted.cpu().numpy()
    return [
      Boundary(
        values[0][row],
        values[1][row],
        float(values[2][row]),
        float(values[3][row]),
        values[4][row],
        values[5][row],
        flags[row],
      )
      for row in range(states.shape[0])
    ]

  @torch.no_grad()
  def __call__(
    self,
    history: torch.Tensor,
    target: torch.Tensor,
    duration: torch.Tensor,
    columns: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Contacts (B, columns, 2) and channels (B, columns, 2, 6) for dynamic states."""
    steps = history.shape[1]
    future = target.shape[1]
    anchor_state = history[:, -1]
    history_pose = encode_pose(history, anchor_state)
    target_pose = encode_pose(target, anchor_state)
    heading = yaw_quat(anchor_state[:, 3:7])
    before, before_channels, _ = extract(
      self.feet, history_pose, self.fps, self.contact
    )
    # a single B state has no sole speed, so step it back one tick by its
    # velocities: a foot gliding low just before touchdown is not planted yet
    previous = encode_pose(rewind(target[:, :1], self.fps), anchor_state)
    after, after_channels, _ = extract(
      self.feet,
      torch.cat((previous, target_pose), dim=1),
      self.fps,
      self.contact,
    )
    after, after_channels = after[:, 1:], after_channels[:, 1:]
    starts = self.boundaries(
      history[:, -1], history_pose[:, -1], heading, before[:, -1]
    )
    ends = self.boundaries(target[:, 0], target_pose[:, 0], heading, after[:, 0])
    batch = duration.shape[0]
    device = duration.device
    contact = torch.zeros(batch, columns, 2, dtype=torch.bool, device=device)
    channels = torch.zeros(batch, columns, 2, FOOT_CHANNELS, device=device)
    contact[:, :steps] = before
    channels[:, :steps] = before_channels
    anchor = steps - 1
    for row in range(batch):
      ticks = int(duration[row])
      window_contact, window_channels = plan_window(
        starts[row], ends[row], ticks, self.fps, self.gait
      )
      inner = slice(anchor + 1, anchor + ticks)
      contact[row, inner] = torch.from_numpy(window_contact[1:ticks]).to(device)
      channels[row, inner] = torch.from_numpy(window_channels[1:ticks]).to(
        device, torch.float32
      )
      tail = slice(anchor + ticks, anchor + ticks + future)
      contact[row, tail] = after[row]
      channels[row, tail] = after_channels[row]
    return contact, channels
