"""The corpus arch_4 is trained on, and the profiles it aims at.

Two things live here, and they are the two halves of the design.

The corpus is a set of recorded motions -- a long, varied performance of a real body
doing many different things in sequence. A training example is one clip with a hole cut
in it: the frames before the hole, the frames after it, and the frames inside it, which
the bridge never sees and is rewarded for reproducing. Nothing about a skill, a policy
or a hand-over appears anywhere in it. That is deliberate: what is being learned is how
one stretch of motion connects to another, and a corpus made of policy rollouts could
only teach the connections those policies already make.

Where the hole is cut matters more than anything else here. A motion capture corpus is
mostly uneventful -- standing, idling, walking in a straight line -- and a hole cut in
the middle of a steady walk is filled correctly by continuing to walk, which teaches a
bridge nothing. So windows are drawn toward the busy parts: where the body accelerates,
turns, leaves the ground or moves its joints hard. Those are the places where the two
sides of the hole are genuinely different motions, and the in-between is a real answer
rather than an extrapolation. `eventfulness` in the config is how hard that bias is
applied, and setting it to zero recovers uniform sampling if the corpus is already
curated.

The profiles are the other half. At inference there is no clip to read the after-context
out of, so it comes from a bank recorded by letting each skill run in the arena: the
"rollout of the next policy" the bridge is asked to stitch to. Which part of that rollout
is a genuine choice, and `profile_offset` is where it is made -- the opening of the skill,
or a later window of it once it has settled.

The mismatch between the two is the design's central bet and is worth stating outright:
the bridge is fitted on motion capture and deployed against policies. Nothing here checks
that it transfers. What makes the bet reasonable is that both sides are expressed in the
same body-frame vector (see frames.py), so what the bridge learned is a relationship
between motions rather than between a particular dataset's coordinates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.skills.architectures.arch_4.config import MaskedTraining
from mjlab.tasks.skills.architectures.arch_4.frames import (
  Groups,
  clip_frames,
  frame_dim,
  robot_frame,
)
from mjlab.tasks.skills.experiment import Experiment
from mjlab.tasks.skills.skill import SkillPool

# How a skill is set up before it is recorded for its profile: the experiment writes
# whatever commands that skill reads (the corridor's speed, the jump's distance). None
# means the arena's own defaults are already right.
PrepareSkill = Callable[[ManagerBasedRlEnv, int], None]


@dataclass
class MaskedBatch:
  """One masked window per environment, ready to roll."""

  past: torch.Tensor
  """Frames before the gap, (num_envs, past_steps, frame_dim)."""

  future: torch.Tensor
  """Frames after the gap, (num_envs, future_steps, frame_dim)."""

  reference: torch.Tensor
  """What actually happened, (num_envs, horizon, frame_dim).

  Covers the gap *and* the visible window past it, because they are one continuous
  stretch of the same clip: a rollout tracks this whole thing, and the only difference
  between the two halves is that the bridge has been shown the second one. Never given
  to the policy; the reward reads it and nothing else does.
  """

  gap: torch.Tensor
  """How long each environment's gap is, (num_envs,) int64."""

  states: torch.Tensor
  """What it takes to put the robot on any frame of the rollout, (num_envs, horizon, S).

  Root position, quaternion, linear and angular velocity (13, the layout
  `write_root_state_to_sim` wants), then joint positions and velocities. The whole
  stream rather than only the first frame, because a robot that falls part way through
  is put back on the reference where it should have been rather than left in the arena's
  own reset state, which is the reference-state initialization every tracking task in
  this repository relies on.
  """

  num_joints: int

  def at(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(root state, joint positions, joint velocities) at `step` of the rollout."""
    state = self.states[:, min(step, self.states.shape[1] - 1)]
    return (
      state[:, :13],
      state[:, 13 : 13 + self.num_joints],
      state[:, 13 + self.num_joints :],
    )


class MotionCorpus:
  """Converted motion clips, as frames plus the states needed to start inside them.

  Clips have different lengths, so everything is padded to the longest and the true
  lengths are kept alongside. Nothing ever reads past a clip's end: the sampler only
  offers windows that fit.
  """

  def __init__(
    self,
    motion_files: tuple[str, ...] | list[str],
    device: str,
    horizon: int,
    past_steps: int,
    stride: int,
    eventfulness: float,
  ) -> None:
    if not motion_files:
      raise FileNotFoundError(
        "No motion files. arch_4 is trained on a corpus of recorded motions, not on "
        "the skill pool; point `motion_dir` at converted clips and see the experiment's "
        "dataset.py for how to produce them."
      )

    paths = [Path(f) for f in motion_files]
    raw = [np.load(p) for p in paths]
    self.device = device
    self.fps = float(np.asarray(raw[0]["fps"]).reshape(-1)[0])
    self.dt = 1.0 / self.fps
    self.num_joints = int(raw[0]["joint_pos"].shape[-1])
    self.groups = Groups(self.num_joints)
    self.frame_dim = frame_dim(self.num_joints)

    # A window needs a full past context before it and the whole rollout after it, so a
    # clip shorter than that offers nothing and is dropped rather than padded into
    # existence. The context reaches back `stride` frames per step it holds.
    self.stride = max(stride, 1)
    self.back_span = self.stride * (past_steps - 1) + 1
    span = self.back_span + horizon
    usable = [(p, d) for p, d in zip(paths, raw, strict=True) if _length(d) > span]
    dropped = len(paths) - len(usable)
    if not usable:
      raise ValueError(
        f"Every clip is shorter than the {span} frames one training example spans "
        f"({self.back_span} of context plus a {horizon}-step rollout). Either the clips "
        f"are too short or the window is too long."
      )

    self.names = [p.stem for p, _ in usable]
    lengths = [_length(d) for _, d in usable]
    self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    self.num_clips = len(usable)
    padded = int(max(lengths))

    frames: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    for _, data in usable:
      frame, state = _read_clip(data, device)
      frames.append(_pad(frame, padded))
      states.append(_pad(state, padded))
    self.frames = torch.stack(frames)
    self.states = torch.stack(states)

    self._build_sampler(horizon, eventfulness)

    print(
      f"[corpus] {self.num_clips} clip(s), "
      f"{int(self.lengths.sum())} frames at {self.fps:g} Hz "
      f"({float(self.lengths.sum()) * self.dt:.0f} s)"
      + (f", {dropped} too short to use" if dropped else "")
    )
    print(
      f"[corpus] {int(self._weights.shape[0])} maskable windows, "
      f"eventfulness bias {eventfulness:g}"
    )

  ##
  # Where to cut.
  ##

  def _build_sampler(self, horizon: int, eventfulness: float) -> None:
    """Enumerate every window that fits, and score it by how much happens inside it."""
    score = self._eventfulness()
    # Window score is the mean of the per-frame score over the rollout it covers,
    # computed for every start at once out of a prefix sum.
    cumulative = torch.cat([torch.zeros_like(score[:, :1]), score.cumsum(dim=1)], dim=1)

    clip_ids: list[torch.Tensor] = []
    starts: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for clip in range(self.num_clips):
      last = int(self.lengths[clip]) - horizon
      if last <= self.back_span:
        continue
      start = torch.arange(self.back_span, last, device=self.device)
      window = (cumulative[clip, start + horizon] - cumulative[clip, start]) / horizon
      clip_ids.append(torch.full_like(start, clip))
      starts.append(start)
      weights.append((1.0 + window).clamp_min(1e-6) ** eventfulness)

    self._clip_ids = torch.cat(clip_ids)
    self._starts = torch.cat(starts)
    self._weights = torch.cat(weights)

  def _eventfulness(self) -> torch.Tensor:
    """Per frame, how much is happening: (num_clips, padded).

    Four things, each normalized by its own median over the corpus so that a metre per
    second and a radian per second can be added together at all. The median rather than
    the mean because a corpus with long idle stretches has a mean dominated by them.
    """
    groups = self.groups
    lin_vel = self.frames[..., groups.lin_vel]
    ang_vel = self.frames[..., groups.ang_vel]
    joint_vel = self.frames[..., groups.joint_vel]

    # Acceleration: the clearest sign that the motion is changing rather than repeating.
    acceleration = torch.zeros_like(lin_vel)
    acceleration[:, 1:] = (lin_vel[:, 1:] - lin_vel[:, :-1]) / self.dt

    parts = (
      acceleration.norm(dim=-1),
      ang_vel[..., 2].abs(),
      lin_vel[..., 2].abs(),
      joint_vel.norm(dim=-1),
    )
    total = torch.zeros_like(parts[0])
    for part in parts:
      scale = part.median().clamp_min(1e-6)
      total = total + part / scale
    return total / len(parts)

  ##
  # Drawing a batch.
  ##

  def sample(
    self,
    num_envs: int,
    gap_range: tuple[int, int],
    past_steps: int,
    future_steps: int,
    horizon: int,
  ) -> MaskedBatch:
    """One masked window per environment, drawn toward the eventful parts."""
    picked = torch.multinomial(self._weights, num_envs, replacement=True)
    clip_ids = self._clip_ids[picked]
    starts = self._starts[picked]
    low, high = gap_range
    gap = torch.randint(low, high + 1, (num_envs,), device=self.device)

    offsets = torch.arange(horizon, device=self.device)
    reference = self.frames[clip_ids[:, None], starts[:, None] + offsets[None, :]]

    # The context reaches back one frame in `stride`, ending on the frame immediately
    # before the hole, so the most recent thing the policy sees is the last thing that
    # happened. Same convention as the rolling history at inference (see __init__.py).
    back = -1 - self.stride * torch.arange(past_steps - 1, -1, -1, device=self.device)
    past = self.frames[clip_ids[:, None], starts[:, None] + back[None, :]]

    # The visible window begins where this environment's gap ends, so it moves with the
    # gap length rather than sitting at a fixed offset. It fits inside the rollout by
    # construction: the horizon is the longest gap plus the span this window covers.
    forward = self.stride * torch.arange(future_steps, device=self.device)
    future_index = starts[:, None] + gap[:, None] + forward[None, :]
    future = self.frames[clip_ids[:, None], future_index]

    states = self.states[clip_ids[:, None], starts[:, None] + offsets[None, :]]
    return MaskedBatch(
      past=past,
      future=future,
      reference=reference,
      gap=gap,
      states=states,
      num_joints=self.num_joints,
    )


def _length(data) -> int:
  return int(data["joint_pos"].shape[0])


def _pad(values: torch.Tensor, length: int) -> torch.Tensor:
  """Pad a (T, D) clip out to `length` by repeating its final frame."""
  if values.shape[0] >= length:
    return values[:length]
  tail = values[-1:].expand(length - values.shape[0], -1)
  return torch.cat([values, tail], dim=0)


def _read_clip(data, device: str) -> tuple[torch.Tensor, torch.Tensor]:
  """One npz to (frames, states).

  `frames` is the body-frame description everything downstream works in; `states` is
  what it takes to put the robot exactly there, which is the world-frame root pose and
  the joint state. Body 0 is the root link, the same convention the tracking pipeline's
  converter writes and `motion_lib` reads.
  """

  def get(key: str) -> torch.Tensor:
    return torch.tensor(np.asarray(data[key]), dtype=torch.float32, device=device)

  root_pos = get("body_pos_w")[:, 0]
  root_quat = get("body_quat_w")[:, 0]
  root_lin_vel = get("body_lin_vel_w")[:, 0]
  root_ang_vel = get("body_ang_vel_w")[:, 0]
  joint_pos = get("joint_pos")
  joint_vel = get("joint_vel")

  frames = clip_frames(
    root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel
  )
  states = torch.cat(
    [root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel], dim=-1
  )
  return frames, states


def discover_motions(motion_dir: str, pattern: str) -> tuple[str, ...]:
  """Converted clips in `motion_dir`, sorted by name."""
  directory = Path(motion_dir)
  if not directory.is_dir():
    raise FileNotFoundError(
      f"No motion directory at {directory.resolve()}. arch_4 trains on a corpus of "
      f"recorded motions; the experiment's dataset.py downloads and converts one."
    )
  return tuple(str(p) for p in sorted(directory.glob(pattern)))


##
# Profiles: what a transition aims at, at inference.
##


@torch.no_grad()
def collect_profiles(
  env: ManagerBasedRlEnv,
  exp: Experiment,
  cfg: MaskedTraining,
  prepare: PrepareSkill | None = None,
) -> dict[int, torch.Tensor]:
  """Record a bank of frame windows per skill, by letting each one run.

  This is the "first M frames of a rollout of the next policy" the design is written
  around, and `profile_offset` is the "where" it leaves open: the window is taken that
  many steps into the skill's own rollout rather than always at its very first frame.

  Environments that break before the window is complete are dropped rather than
  recorded: a skill that fell over is not a thing to aim at.
  """
  device = env.device
  num_envs = env.num_envs
  pool = exp.pool
  entity: Entity = env.scene[exp.entity_name]
  low, high = cfg.profile_offset
  stride = max(cfg.context_stride, 1)
  # Recorded at the same stride the corpus is read at, or the window would cover a
  # different amount of time than the one the policy was trained against.
  span = high + stride * (cfg.future_steps - 1) + 1
  everything = torch.ones(num_envs, dtype=torch.bool, device=device)
  profiles: dict[int, torch.Tensor] = {}

  print(f"\n=== profiles: {span} steps of each skill, {cfg.profile_rows} rows each ===")
  for skill_id in range(len(pool)):
    skill = pool[skill_id]
    obs, _ = env.reset()
    if prepare is not None:
      prepare(env, skill_id)
    pool.reset(everything)

    recorded: list[torch.Tensor] = []
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    involved = pool.driven_by(skill_id, num_envs, device)
    assignment = torch.full((num_envs,), skill_id, device=device)
    for _ in range(span):
      actions = SkillPool.select(pool.act_each(obs, involved), assignment)
      recorded.append(robot_frame(entity))
      obs, _, terminated, time_out, _ = env.step(actions)
      alive = alive & ~(terminated | time_out)

    frames = torch.stack(recorded, dim=1)
    offsets = torch.randint(low, high + 1, (num_envs,), device=device)
    window = stride * torch.arange(cfg.future_steps, device=device)
    picked = frames[
      torch.arange(num_envs, device=device)[:, None], offsets[:, None] + window[None, :]
    ]
    kept = picked[alive & torch.isfinite(picked).all(dim=-1).all(dim=-1)]
    if kept.shape[0] == 0:
      raise RuntimeError(
        f"Every environment broke while recording '{skill.name}', so there is no "
        f"profile to bridge toward it with. That skill cannot run in this arena."
      )
    profiles[skill_id] = kept[: cfg.profile_rows].clone()
    print(
      f"[profile] '{skill.name}': {int(profiles[skill_id].shape[0])} rows "
      f"of {cfg.future_steps} frames, offset {low}-{high}, "
      f"{float(alive.float().mean()):.0%} survived"
    )

  return profiles
