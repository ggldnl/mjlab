"""The style half of a skill: what its reference clips look like, learned.

This is the machinery behind the `amp_style_reward` term in `mdp.py`, kept apart
from it because none of it is an MDP term: it is a feature definition and a small
GAN.

Two pieces:

- `amp_features` is the pose/velocity summary both sides are compared through. One
  function serves the reference clips and the live robot, because if the two ever
  disagreed the discriminator would learn to spot *that* rather than the gait.
  Everything is expressed in the root's yaw frame, so a feature vector carries no
  absolute position and no heading: the same stride looks identical wherever on the
  plane it happens and whichever way it faces.
- `Amp` owns the reference transitions, the discriminator and the policy replay
  buffer, and turns "does this look like the reference" into a number in [0, 1].

The discriminator is deliberately *not* conditioned on the velocity command. Each
skill has its own reference folder, so its discriminator already answers one
question -- "does this look like running" -- and conditioning it inside a single
behavior buys nothing while adding a way for it to cheat: if policy commands and
reference commands are drawn from even slightly different distributions, the
command channel alone separates real from fake and the motion stops being looked
at. Velocity does the goal-conditioning instead, through the command box and the
tracking rewards. If a single skill ever has to span visibly different gaits, that
is when to bring conditioning back.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

# Bodies whose position (in the root yaw frame) enters the features. Feet carry most
# of what separates a walk from a run from a jump; elbows capture arm swing, torso
# captures the lean.
G1_KEY_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_elbow_link",
  "right_elbow_link",
  "torso_link",
)


def amp_features(
  root_pos_w: torch.Tensor,
  root_quat_w: torch.Tensor,
  root_lin_vel_w: torch.Tensor,
  root_ang_vel_w: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  key_body_pos_w: torch.Tensor,
) -> torch.Tensor:
  """Pack one frame per batch element into the discriminator's input space."""
  heading = yaw_quat(root_quat_w)
  num_key = key_body_pos_w.shape[1]

  gravity_w = torch.zeros_like(root_pos_w)
  gravity_w[:, 2] = -1.0

  key_offsets_w = key_body_pos_w - root_pos_w.unsqueeze(1)
  heading_per_key = heading.unsqueeze(1).expand(-1, num_key, -1)

  return torch.cat(
    [
      root_pos_w[:, 2:3],
      quat_apply_inverse(root_quat_w, gravity_w),
      quat_apply_inverse(heading, root_lin_vel_w),
      quat_apply_inverse(heading, root_ang_vel_w),
      joint_pos,
      joint_vel,
      quat_apply_inverse(heading_per_key, key_offsets_w).flatten(1),
    ],
    dim=-1,
  )


def features_from_entity(
  entity: Entity, key_body_indexes: torch.Tensor
) -> torch.Tensor:
  """The features for every env, read off the live robot."""
  data = entity.data
  return amp_features(
    data.root_link_pos_w,
    data.root_link_quat_w,
    data.root_link_lin_vel_w,
    data.root_link_ang_vel_w,
    data.joint_pos,
    data.joint_vel,
    data.body_link_pos_w[:, key_body_indexes],
  )


def features_from_clip(
  clip_path: Path, key_body_indexes: torch.Tensor, device: str
) -> torch.Tensor:
  """The features for every frame of one clip npz written by `dataset.py`.

  Features are derived here rather than baked into the clip on purpose: changing the
  feature set is then a code change, not a dataset rebuild.
  """
  data = np.load(clip_path)
  as_tensor = lambda key: torch.tensor(data[key], dtype=torch.float32, device=device)  # noqa: E731
  # Body 0 is the root link, the same convention the replay recorded them in.
  body_pos_w = as_tensor("body_pos_w")
  return amp_features(
    root_pos_w=body_pos_w[:, 0],
    root_quat_w=as_tensor("body_quat_w")[:, 0],
    root_lin_vel_w=as_tensor("body_lin_vel_w")[:, 0],
    root_ang_vel_w=as_tensor("body_ang_vel_w")[:, 0],
    joint_pos=as_tensor("joint_pos"),
    joint_vel=as_tensor("joint_vel"),
    key_body_pos_w=body_pos_w[:, key_body_indexes],
  )


class Discriminator(nn.Module):
  def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
    super().__init__()
    layers: list[nn.Module] = []
    prev = input_dim
    for dim in hidden_dims:
      layers += [nn.Linear(prev, dim), nn.ReLU()]
      prev = dim
    self.trunk = nn.Sequential(*layers)
    self.head = nn.Linear(prev, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.head(self.trunk(x)).squeeze(-1)


class Amp:
  """One skill's discriminator, its reference transitions and its replay buffer."""

  def __init__(
    self,
    clip_dir: str | Path,
    key_body_indexes: torch.Tensor,
    device: str,
    *,
    hidden_dims: tuple[int, ...] = (1024, 512),
    learning_rate: float = 1e-4,
    buffer_size: int = 100_000,
    grad_penalty_coef: float = 5.0,
    logit_weight_decay: float = 0.05,
  ) -> None:
    paths = sorted(Path(clip_dir).glob("*.npz"))
    if not paths:
      raise FileNotFoundError(
        f"No reference clips in {clip_dir}. Build them first with "
        "`uv run python -m mjlab.tasks.skills.experiments.parkour.dataset`."
      )

    self.device = device
    self.num_clips = len(paths)
    # Pairs stay inside one clip, so no transition ever spans two cuts: there is no
    # seam anywhere for the policy to be asked to imitate.
    features = [features_from_clip(p, key_body_indexes, device) for p in paths]
    self._expert = torch.cat([f[:-1] for f in features])
    self._expert_next = torch.cat([f[1:] for f in features])

    # Fixed normalization from the reference set. Features mix metres, radians and
    # rad/s, so an unnormalized discriminator would key on whichever channel happens
    # to be largest.
    self._mean = self._expert.mean(0, keepdim=True)
    self._std = self._expert.std(0, keepdim=True).clamp_min(1e-4)

    self.feature_dim = self._expert.shape[1]
    input_dim = 2 * self.feature_dim

    self.net = Discriminator(input_dim, hidden_dims).to(device)
    self._optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
    self._grad_penalty_coef = grad_penalty_coef
    self._logit_weight_decay = logit_weight_decay

    self._buffer = torch.zeros(buffer_size, input_dim, device=device)
    self._buffer_size = buffer_size
    self._buffer_head = 0
    self._buffer_filled = 0

  @property
  def num_transitions(self) -> int:
    return int(self._expert.shape[0])

  def _disc_input(
    self, features: torch.Tensor, next_features: torch.Tensor
  ) -> torch.Tensor:
    return torch.cat(
      [
        (features - self._mean) / self._std,
        (next_features - self._mean) / self._std,
      ],
      dim=-1,
    )

  def push(self, features: torch.Tensor, next_features: torch.Tensor) -> None:
    """Record policy transitions as the discriminator's negative examples."""
    batch = self._disc_input(features, next_features).detach()
    num = batch.shape[0]
    if num >= self._buffer_size:
      self._buffer.copy_(batch[-self._buffer_size :])
      self._buffer_head = 0
      self._buffer_filled = self._buffer_size
      return
    end = self._buffer_head + num
    if end <= self._buffer_size:
      self._buffer[self._buffer_head : end] = batch
    else:
      split = self._buffer_size - self._buffer_head
      self._buffer[self._buffer_head :] = batch[:split]
      self._buffer[: end - self._buffer_size] = batch[split:]
    self._buffer_head = end % self._buffer_size
    self._buffer_filled = min(self._buffer_filled + num, self._buffer_size)

  @torch.no_grad()
  def style_reward(
    self, features: torch.Tensor, next_features: torch.Tensor
  ) -> torch.Tensor:
    """Reward in [0, 1]: 1 when the transition is indistinguishable from reference."""
    score = self.net(self._disc_input(features, next_features))
    return torch.clamp(1.0 - 0.25 * (score - 1.0) ** 2, min=0.0)

  def update(self, num_updates: int, batch_size: int) -> dict[str, float]:
    """Run discriminator gradient steps against the current replay buffer."""
    if self._buffer_filled < batch_size:
      return {}

    logs = {"loss": 0.0, "expert_score": 0.0, "policy_score": 0.0}
    with torch.enable_grad():
      for _ in range(num_updates):
        expert_idx = torch.randint(
          0, self.num_transitions, (batch_size,), device=self.device
        )
        expert = self._disc_input(
          self._expert[expert_idx], self._expert_next[expert_idx]
        ).requires_grad_(True)
        policy_idx = torch.randint(
          0, self._buffer_filled, (batch_size,), device=self.device
        )
        policy = self._buffer[policy_idx]

        expert_score = self.net(expert)
        policy_score = self.net(policy)

        # Least-squares GAN: reference maps to +1, policy to -1. Bounded targets keep
        # the score in the range `style_reward` above expects.
        loss = 0.5 * (
          ((expert_score - 1.0) ** 2).mean() + ((policy_score + 1.0) ** 2).mean()
        )

        # Penalizing the gradient on reference samples keeps the discriminator from
        # sharpening into a step function the policy gets no usable signal from.
        grad = torch.autograd.grad(
          expert_score.sum(), expert, create_graph=True, retain_graph=True
        )[0]
        loss = loss + self._grad_penalty_coef * 0.5 * grad.pow(2).sum(-1).mean()
        loss = loss + self._logit_weight_decay * self.net.head.weight.pow(2).sum()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        logs["loss"] += loss.item() / num_updates
        logs["expert_score"] += expert_score.mean().item() / num_updates
        logs["policy_score"] += policy_score.mean().item() / num_updates
    return logs
