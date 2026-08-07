"""The window the bridge is asked to cross, and the reference it is scored against.

One episode is one hole. On reset the robot is teleported onto the hand-off -- the last
context frame of a window drawn from the corpus -- carrying that frame's joint angles and,
crucially, its velocities. The momentum the whole architecture exists to exploit is in
those velocities, and a reset that dropped them would be posing the robot rather than
handing it over.

The clip is then rebased onto the environment. The hand-off frame's horizontal position is
translated to the environment origin and nothing else is touched, so the reference and the
robot live in the same coordinates and every comparison downstream is a plain subtraction.
A pure translation is enough because the robot is spawned at the clip's own heading; there
is no rotation to undo.

##
# What the policy is told, and what it is not
##

It is told where it has to arrive: the future context, as poses relative to wherever the
robot is right now, refreshed every step. It is told how far through the hole it is and
how long the hole is.

It is not told what the body actually did inside the hole. That absence is the
architecture. A policy shown the reference it is being scored against learns to follow a
reference, and at inference there is no reference to follow -- only a target handed over by
the descriptor. A policy shown two ends and paid for what went between has to learn what
usually goes between, which is the part that transfers.

The reference is still read every step, by the reward. Training signal and observation are
different channels and only one of them is allowed to see the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.skills.architectures.arch_4.bridge import frames
from mjlab.tasks.skills.architectures.arch_4.bridge.dataset import (
  ROOT_STATE_DIM,
  BridgeDataset,
  Corpus,
  CorpusCfg,
  WindowCfg,
  build_corpus,
)
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)


class BridgeCommand(CommandTerm):
  """Draws a window per environment, teleports onto its hand-off, holds its reference."""

  cfg: BridgeCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: BridgeCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    corpus = Corpus(build_corpus(cfg.corpus))
    dataset = BridgeDataset(corpus, cfg.windows, split=cfg.split)
    if not len(dataset):
      raise ValueError(f"No windows in the '{cfg.split}' split of this corpus.")

    self.num_joints = corpus.num_joints
    self.pose_dim = frames.pose_dim(self.num_joints)
    self.past = cfg.windows.past
    self.future = cfg.windows.future
    self.max_gap = cfg.windows.gap_range[1]
    self.tail = cfg.tail_steps

    # The corpus as one tensor, with a window reduced to an offset into it. Sampling is
    # then index arithmetic and the whole corpus lives on the device once.
    lengths = [corpus.length(i) for i in range(len(corpus))]
    offsets = torch.tensor(
      [sum(lengths[:i]) for i in range(len(lengths))], device=self.device
    )
    self.flat = torch.cat(corpus.states, dim=0).to(self.device)
    self.window_base = (
      torch.tensor([w.start for w in dataset.windows], device=self.device)
      + offsets[torch.tensor([w.clip for w in dataset.windows], device=self.device)]
    )
    self.window_gap = torch.tensor([w.gap for w in dataset.windows], device=self.device)
    self.num_windows = len(dataset.windows)
    print(f"[bridge] {self.num_windows} windows in the '{cfg.split}' split")

    # Per environment: the reference from the hand-off onward, already rebased, plus how
    # long this hole is and how far into it we are.
    span = self.max_gap + self.future
    self.reference = torch.zeros(
      self.num_envs, span, self.flat.shape[1], device=self.device
    )
    self.gap = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
    self.step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # Which frames of the future context are shown as the target. A handful spread across
    # it rather than all of them: consecutive mocap frames are nearly identical and the
    # observation would be mostly repetition.
    self.target_index = torch.linspace(
      0, self.future - 1, cfg.target_samples, device=self.device
    ).long()

    # Which window each environment drew, and an override for a viewer that wants to
    # study one hole rather than take what it is given. Training leaves this alone.
    self.chosen = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.force_window: int | None = None

    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_root_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["progress"] = torch.zeros(self.num_envs, device=self.device)

  def context_before(self, env_id: int) -> torch.Tensor:
    """The frames leading up to the hand-off, rebased like the reference. (past, 13 + 2J).

    Not needed by training, which starts at the hand-off. An evaluator needs it to show
    where the robot came from, since a bridge is only meaningful against the motion that
    fed into it.
    """
    base = int(self.window_base[self.chosen[env_id]])
    context = self.flat[base : base + self.past].clone()
    hand_off = self.flat[base + self.past - 1]
    origin = self._env.scene.env_origins[env_id]
    context[:, 0:2] += (origin[:2] - hand_off[0:2]).unsqueeze(0)
    return context

  @property
  def command(self) -> torch.Tensor:
    """Where the robot has to get to, and how much of the hole is left.

    The target poses are relative to the robot's current base, so this stays a live
    instruction rather than a fact about the past: as the robot moves, what remains to be
    covered shrinks, and that is what the policy has to act on.
    """
    return torch.cat([self.target_in_base(), self.phase()], dim=-1)

  def phase(self) -> torch.Tensor:
    """How far through the hole, and how long it is. (num_envs, 2)."""
    progress = (self.step.float() / self.gap.float()).clamp(max=1.0)
    return torch.stack([progress, self.gap.float() / self.max_gap], dim=-1)

  def target_in_base(self) -> torch.Tensor:
    """The future context, seen from where the robot is now. (num_envs, samples * pose)."""
    index = (self.gap.unsqueeze(1) + self.target_index.unsqueeze(0)).clamp(
      max=self.reference.shape[1] - 1
    )
    target = torch.gather(
      self.reference,
      1,
      index.unsqueeze(-1).expand(-1, -1, self.reference.shape[2]),
    )
    base_pos = self.robot.data.root_link_pos_w
    base_yaw = yaw_quat(self.robot.data.root_link_quat_w)
    return frames.encode(target, base_pos, base_yaw).flatten(start_dim=1)

  def reference_now(self) -> torch.Tensor:
    """The reference state at this step, for the reward. (num_envs, 13 + 2J).

    Clamped past the end of the hole so the tail keeps reading the future context, which
    is what asks whether the robot arrived somewhere the recording carries on from.
    """
    index = self.step.clamp(max=self.reference.shape[1] - 1)
    return self.reference[torch.arange(self.num_envs, device=self.device), index]

  def arrival_weight(self) -> torch.Tensor:
    """A ramp from 1 at the hand-off to `cfg.arrival_weight` at the far end.

    Filling a hole plausibly and arriving somewhere else is the failure this architecture
    exists to prevent, so the two must not score the same. The ramp is what says so.
    """
    progress = (self.step.float() / self.gap.float()).clamp(max=1.0)
    return 1.0 + (self.cfg.arrival_weight - 1.0) * progress

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Draw a window per environment and put the robot on its hand-off."""
    if env_ids.numel() == 0:
      return
    count = env_ids.numel()
    if self.force_window is None:
      choice = torch.randint(0, self.num_windows, (count,), device=self.device)
    else:
      # An evaluator wants to look at one particular hole rather than whatever came up.
      choice = torch.full(
        (count,), self.force_window % self.num_windows, device=self.device
      )
    base = self.window_base[choice]
    gap = self.window_gap[choice]
    self.chosen[env_ids] = choice
    self.gap[env_ids] = gap
    self.step[env_ids] = 0

    # The window from the hand-off onward, padded to the widest hole this run allows.
    span = self.reference.shape[1]
    index = base.unsqueeze(1) + self.past + torch.arange(span, device=self.device)
    index = index.clamp(max=self.flat.shape[0] - 1)
    window = self.flat[index].clone()

    # Rebase: slide the clip so its hand-off sits over this environment's origin. Only
    # the horizontal position moves, so heights, headings and every velocity survive.
    hand_off = self.flat[base + self.past - 1]
    origin = self._env.scene.env_origins[env_ids]
    shift = origin[:, :2] - hand_off[:, :2]
    window[:, :, 0:2] += shift.unsqueeze(1)
    self.reference[env_ids] = window

    root_pos = hand_off[:, 0:3].clone()
    root_pos[:, 0:2] = origin[:, :2]
    root_quat = hand_off[:, 3:7]
    root_lin_vel = hand_off[:, 7:10]
    root_ang_vel = hand_off[:, 10:ROOT_STATE_DIM]
    joint_pos = hand_off[:, ROOT_STATE_DIM : ROOT_STATE_DIM + self.num_joints]
    joint_vel = hand_off[:, ROOT_STATE_DIM + self.num_joints :]

    # A share of episodes start off the reference. At inference the robot arrives from
    # whatever the outgoing skill left behind, never from a corpus frame, and a policy
    # that has only ever started exactly on the reference has never had to steer.
    if self.cfg.start_noise > 0.0:
      scale = self.cfg.start_noise
      root_pos = root_pos + torch.randn_like(root_pos) * scale * 0.05
      root_lin_vel = root_lin_vel + torch.randn_like(root_lin_vel) * scale * 0.2
      root_ang_vel = root_ang_vel + torch.randn_like(root_ang_vel) * scale * 0.2
      joint_pos = joint_pos + torch.randn_like(joint_pos) * scale * 0.05
      joint_vel = joint_vel + torch.randn_like(joint_vel) * scale * 0.5

    limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(limits[:, :, 0], limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    self.robot.write_root_state_to_sim(
      torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1),
      env_ids=env_ids,
    )
    self.robot.reset(env_ids=env_ids)

  def _update_command(self) -> None:
    self.step += 1

  def _update_metrics(self) -> None:
    reference = self.reference_now()
    joints = reference[:, ROOT_STATE_DIM : ROOT_STATE_DIM + self.num_joints]
    self.metrics["error_joint_pos"] = (
      (self.robot.data.joint_pos - joints).abs().mean(dim=-1)
    )
    self.metrics["error_root_pos"] = (
      self.robot.data.root_link_pos_w - reference[:, 0:3]
    ).norm(dim=-1)
    self.metrics["progress"] = (self.step.float() / self.gap.float()).clamp(max=1.0)


def relative_pose(
  base_pos: torch.Tensor,
  base_quat: torch.Tensor,
  pos: torch.Tensor,
  quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """A pose seen from a base frame, using heading only so tilt does not leak in."""
  yaw = yaw_quat(base_quat)
  return quat_apply_inverse(yaw, pos - base_pos), quat_mul(quat_conjugate(yaw), quat)


@dataclass(kw_only=True)
class BridgeCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  corpus: CorpusCfg = field(default_factory=CorpusCfg)
  windows: WindowCfg = field(default_factory=WindowCfg)

  split: str = "train"
  """Which half of the corpus this env draws from. The evaluation env uses 'eval', so a
  run is scored on subjects it was never trained on."""

  target_samples: int = 5
  """Frames of the future context shown to the policy."""

  tail_steps: int = 10
  """Steps the policy keeps driving after the hole closes, scored against the future
  context. This is the only part of training that asks the question inference asks: not
  'did you fill the hole' but 'did you leave the robot somewhere the next thing can pick
  up from'."""

  arrival_weight: float = 3.0
  """How much more the far end of a hole counts than the near end."""

  start_noise: float = 1.0
  """Scale on the perturbation applied to the hand-off state. Zero starts every episode
  exactly on the reference."""

  def build(self, env: ManagerBasedRlEnv) -> BridgeCommand:
    return BridgeCommand(self, env)
