"""Per frame feasibility critic, a differentiable stand in for the frozen tracker.

The simulator and the tracker cannot be differentiated, so the planner cannot learn
from a rollout directly. The critic learns to predict, for every frame of a planned
window, how well the frozen tracker will follow it. Once trained, its prediction is
a loss the planner can be pushed through.

It reads the planned pose window in A's heading frame, the same poses the planner
rebuilds from its steps, plus three flags that mark history, bridge and B's
continuation. Frames after the continuation are zeroed and ignored.

Labels come from executed plans, successes and failures alike. A failed plan is the
most useful example, since its label shows where tracking broke down:

    label     tracking score of the frame, 1 is perfect, see imitation.command.score
    fallen    every frame from the first fallen frame on is labelled 0
    rejected  plans the kinematic gate stops are never executed, all frames are 0
    scored    frames after A up to the last continuation frame

Rejected plans matter. In a short test without them, one fine tuning cycle against
a young critic dropped the gate pass rate from 44% to 1%: the critic had never seen
a broken plan, so it did not stop the planner from producing them.

The pose normalizer is copied from the planner once and saved with the critic, so a
resumed run reads poses the way the critic was trained.
"""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from mjlab.tasks.bridging.bridges.diffusion.dataset.motions import Layout, Normalizer
from mjlab.tasks.bridging.bridges.imitation.command import score

SEGMENTS = 3
"""Flags per frame: history up to A, bridge, B's continuation"""


@dataclass(frozen=True)
class CriticCfg:
  width: int = 128
  layers: int = 3
  heads: int = 4
  learning_rate: float = 3e-4
  batch: int = 256
  updates_per_cycle: int = 500
  capacity: int = 32_768
  min_samples: int = 2_048
  """Executed plans needed before the critic's prediction is used as a loss"""
  max_grad_norm: float = 1.0

  def __post_init__(self) -> None:
    if self.width % self.heads or min(self.width, self.layers, self.heads) < 1:
      raise ValueError("Critic width must be a positive multiple of heads")
    if min(self.batch, self.capacity, self.updates_per_cycle) < 1:
      raise ValueError("Critic batch, capacity and updates must be positive")
    if self.min_samples < 1 or self.min_samples > self.capacity:
      raise ValueError("min_samples must be in [1, capacity]")


def segment_flags(
  columns: int, history: int, future: int, duration: torch.Tensor
) -> torch.Tensor:
  """History, bridge and continuation flags, (batch, columns, 3)."""
  anchor = history - 1
  time = torch.arange(columns, device=duration.device)[None]
  target = anchor + duration[:, None]
  before = (time <= anchor).expand(duration.shape[0], -1)
  bridge = (time > anchor) & (time < target)
  after = (time >= target) & (time < target + future)
  return torch.stack((before, bridge, after), dim=-1).float()


def scored_frames(
  columns: int, history: int, future: int, duration: torch.Tensor
) -> torch.Tensor:
  """Frames after A up to the last continuation frame, (batch, columns)."""
  flags = segment_flags(columns, history, future, duration)
  return flags[..., 1:].any(dim=-1)


def frame_labels(
  errors: torch.Tensor,
  fallen: torch.Tensor,
  tolerances: torch.Tensor,
  history: int,
  columns: int,
) -> torch.Tensor:
  """Per frame tracking score of executed routes, placed on planner columns.

  Route step 0 is A, which sits on column history - 1.
  """
  tracking = score(torch.nan_to_num(errors, nan=1e3), tolerances)
  down = fallen.int().cummax(dim=1).values.bool()
  tracking = tracking * ~down
  labels = tracking.new_zeros(errors.shape[0], columns)
  steps = min(tracking.shape[1], columns - history + 1)
  labels[:, history - 1 : history - 1 + steps] = tracking[:, :steps]
  return labels


class FeasibilityCritic(nn.Module):
  """Bidirectional transformer, one feasibility logit per frame."""

  mean: torch.Tensor
  std: torch.Tensor

  def __init__(
    self,
    layout: Layout,
    normalizer: Normalizer,
    history: int,
    future: int,
    columns: int,
    cfg: CriticCfg,
  ) -> None:
    super().__init__()
    self.history = history
    self.future = future
    self.columns = columns
    pose_width = 9 + layout.joints
    # Pose order from encode_pose and integrate: root position, 6D rotation, joints
    for name, source in (("mean", normalizer.mean), ("std", normalizer.std)):
      stats = torch.cat(
        (
          source[layout.root_position],
          source[layout.rotation],
          source[layout.joint_positions],
        )
      )
      self.register_buffer(name, stats.detach().clone())
    self.input = nn.Linear(pose_width + SEGMENTS, cfg.width)
    self.position = nn.Parameter(torch.zeros(1, columns, cfg.width))
    nn.init.normal_(self.position, std=0.02)
    layer = nn.TransformerEncoderLayer(
      d_model=cfg.width,
      nhead=cfg.heads,
      dim_feedforward=4 * cfg.width,
      dropout=0.0,
      activation="gelu",
      batch_first=True,
      norm_first=True,
    )
    self.blocks = nn.TransformerEncoder(
      layer, num_layers=cfg.layers, enable_nested_tensor=False
    )
    self.output = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, 1))

  def forward(self, pose: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
    """Logits (batch, columns) for raw poses (batch, columns, 9 + joints)."""
    if pose.shape[1:] != (self.columns, self.mean.numel()):
      raise ValueError("pose window has the wrong shape")
    flags = segment_flags(self.columns, self.history, self.future, duration)
    inside = flags.any(dim=-1, keepdim=True)
    normalized = (pose - self.mean) / self.std * inside
    embedded = self.input(torch.cat((normalized, flags), dim=-1)) + self.position
    return self.output(self.blocks(embedded))[..., 0]


class CriticReplay:
  """Bounded replay of executed pose windows and their frame labels."""

  def __init__(self, capacity: int) -> None:
    self.capacity = capacity
    self.pose: torch.Tensor | None = None
    self.duration: torch.Tensor | None = None
    self.labels: torch.Tensor | None = None

  def __len__(self) -> int:
    return 0 if self.duration is None else self.duration.numel()

  def add(
    self, pose: torch.Tensor, duration: torch.Tensor, labels: torch.Tensor
  ) -> None:
    if pose.shape[0] == 0:
      return
    # A frame poses stay within a few meters, half precision is enough
    pose = pose.detach().cpu().to(torch.float16)
    duration = duration.detach().cpu()
    labels = labels.detach().cpu().to(torch.float16)
    if self.pose is None or self.duration is None or self.labels is None:
      self.pose, self.duration, self.labels = pose, duration, labels
    else:
      self.pose = torch.cat((self.pose, pose))[-self.capacity :]
      self.duration = torch.cat((self.duration, duration))[-self.capacity :]
      self.labels = torch.cat((self.labels, labels))[-self.capacity :]

  def sample(
    self, count: int, device: str
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if self.pose is None or self.duration is None or self.labels is None:
      raise RuntimeError("Critic replay is empty")
    picked = torch.randint(len(self), (count,))
    return (
      self.pose[picked].to(device=device, dtype=torch.float32),
      self.duration[picked].to(device),
      self.labels[picked].to(device=device, dtype=torch.float32),
    )


class CriticTrainer:
  """Critic, its optimizer and replay."""

  def __init__(self, critic: FeasibilityCritic, cfg: CriticCfg, device: str) -> None:
    self.critic = critic.to(device)
    self.cfg = cfg
    self.device = device
    self.replay = CriticReplay(cfg.capacity)
    self.optimizer = torch.optim.AdamW(critic.parameters(), lr=cfg.learning_rate)
    self.updates = 0
    self.freeze()

  @property
  def ready(self) -> bool:
    return len(self.replay) >= self.cfg.min_samples

  def freeze(self) -> None:
    """Eval mode with frozen weights, gradients still reach the input poses."""
    self.critic.eval().requires_grad_(False)

  def scored(self, duration: torch.Tensor) -> torch.Tensor:
    critic = self.critic
    return scored_frames(critic.columns, critic.history, critic.future, duration)

  @torch.no_grad()
  def check(
    self, pose: torch.Tensor, duration: torch.Tensor, labels: torch.Tensor
  ) -> dict[str, float]:
    """Error on plans the critic has not trained on yet."""
    frames = self.scored(duration)
    predicted = torch.sigmoid(self.critic(pose, duration))
    count = frames.sum(dim=1).clamp_min(1)
    plan_predicted = (predicted * frames).sum(dim=1) / count
    plan_label = (labels * frames).sum(dim=1) / count
    return {
      "critic/fresh_frame_error": float((predicted - labels).abs()[frames].mean()),
      "critic/fresh_plan_error": float((plan_predicted - plan_label).abs().mean()),
      "critic/fresh_label": float(plan_label.mean()),
    }

  def train(self, updates: int) -> dict[str, float]:
    if len(self.replay) == 0 or updates < 1:
      return {"critic/replay": float(len(self.replay))}
    self.critic.train().requires_grad_(True)
    total = 0.0
    for _ in range(updates):
      pose, duration, labels = self.replay.sample(self.cfg.batch, self.device)
      frames = self.scored(duration)
      logits = self.critic(pose, duration)
      loss = F.binary_cross_entropy_with_logits(logits[frames], labels[frames])
      self.optimizer.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
      self.optimizer.step()
      total += loss.item()
    self.updates += updates
    self.freeze()
    return {
      "critic/loss": total / updates,
      "critic/replay": float(len(self.replay)),
      "critic/updates": float(self.updates),
    }

  def state_dict(self) -> dict:
    return {
      "model": {k: v.cpu() for k, v in self.critic.state_dict().items()},
      "optimizer": self.optimizer.state_dict(),
      "updates": self.updates,
    }

  def load_state_dict(self, saved: dict) -> None:
    self.critic.load_state_dict(saved["model"])
    self.optimizer.load_state_dict(saved["optimizer"])
    self.updates = int(saved["updates"])
    self.freeze()
