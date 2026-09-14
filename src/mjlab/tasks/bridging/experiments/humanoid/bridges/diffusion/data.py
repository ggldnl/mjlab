"""What the model sees: one window of the corpus, canonicalized into a feature tensor.

A window is history + horizon control ticks of one rollout, each tick carrying the state
the robot was in and the action that put it there. The model diffuses the whole block at
once, states and actions together, which is the one structural decision BeyondMimic makes
and both alternatives get wrong: a states-only model is a kinematic planner and needs a
tracker underneath it that may not be able to follow what it plans, and an actions-only
model has nothing to compare a state-space goal against and so cannot be steered.

Per tick, in this order:

    root_xy       2   horizontal position, in the anchor frame
    root_h        1   height above the floor, untouched by the anchor
    root_rot6d    6   orientation relative to the anchor heading
    root_lin_vel  3   in the anchor frame
    root_ang_vel  3   in the anchor frame
    joint_pos     J
    joint_vel     J
    action        J   the joint targets that produced this tick's state
    body_pos      3B  every body in the root frame, when the corpus recorded them

The anchor is the last history tick, the one the robot is actually standing in when a plan
is drawn. Everything horizontal and every heading is expressed relative to it, so a walk
across the arena and the same walk ten metres away are one training sample rather than two,
and height, pitch and roll stay absolute because gravity does not move with the robot. This
is the whole reason a corpus of 140000 states is enough to train on at all.

Body positions are forward kinematics of the joint angles, so they add no information and
are predicted anyway: BeyondMimic's representation ablation found Cartesian body positions
beat joint angles outright, because a small angle error at the hip is centimetres at the
foot and the angle loss cannot see that. They are auxiliary here, never guided, since a
target dynamic state names joint angles and not body positions.

The action at tick k is the one that drove the robot from tick k-1 to tick k, which is the
convention dataset.entry_context records and not the one a policy uses. So the first action
a plan executes is the one at column history, not history - 1. See policy.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  ROOT_STATE_DIM,
  Dataset,
)
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)


def rot6d(quat: torch.Tensor) -> torch.Tensor:
  """First two columns of the rotation matrix, flattened. (..., 4) -> (..., 6).

  Same form the imitation bridge puts in its observation. Six numbers rather than four
  because a quaternion has two representations of every rotation and a network trained to
  regress one of them is being asked to pick a sign.
  """
  matrix = matrix_from_quat(quat)
  return matrix[..., :, :2].transpose(-1, -2).reshape(*quat.shape[:-1], 6)


@dataclass(frozen=True)
class Layout:
  """Where each quantity sits in a feature vector. Built once from the corpus."""

  num_joints: int
  num_bodies: int
  """Zero for a corpus with no body_pos_b column, which drops those features entirely."""

  @property
  def width(self) -> int:
    return 15 + 3 * self.num_joints + 3 * self.num_bodies

  @property
  def root_pos(self) -> slice:
    """xy and height together, which is what an arrival is measured over."""
    return slice(0, 3)

  @property
  def root_ori(self) -> slice:
    return slice(3, 9)

  @property
  def root_lin_vel(self) -> slice:
    return slice(9, 12)

  @property
  def root_ang_vel(self) -> slice:
    return slice(12, 15)

  @property
  def joint_pos(self) -> slice:
    return slice(15, 15 + self.num_joints)

  @property
  def joint_vel(self) -> slice:
    return slice(15 + self.num_joints, 15 + 2 * self.num_joints)

  @property
  def action(self) -> slice:
    return slice(15 + 2 * self.num_joints, 15 + 3 * self.num_joints)

  @property
  def bodies(self) -> slice:
    return slice(15 + 3 * self.num_joints, self.width)

  @staticmethod
  def of(data: Dataset) -> Layout:
    bodies = 0 if data.body_pos_b is None else int(data.body_pos_b.shape[1])
    return Layout(num_joints=data.num_joints, num_bodies=bodies)


def encode(
  layout: Layout,
  states: torch.Tensor,
  anchor: torch.Tensor,
  actions: torch.Tensor | None = None,
  bodies: torch.Tensor | None = None,
) -> torch.Tensor:
  """Turn recorded ticks into features, relative to one anchor state. (N, T, F).

  Args:
    states: (N, T, 13 + 2J) dataset rows, in one shared frame.
    anchor: (N, 13 + 2J) the tick everything is expressed relative to.
    actions: (N, T, J), or None to leave the action features zero.
    bodies: (N, T, B, 3), or None to leave the body features zero.

  Only a horizontal slide and a yaw, so nothing is stretched or bent. The inverse is not
  needed and is not written: a target state is compared to a prediction by encoding it
  against the same anchor, never by decoding the prediction back to the world.
  """
  count, span = states.shape[0], states.shape[1]
  flat = states.reshape(count * span, -1)

  heading = yaw_quat(anchor[:, 3:7])
  spread = heading.unsqueeze(1).expand(-1, span, -1).reshape(count * span, 4)

  origin = torch.zeros_like(anchor[:, 0:3])
  origin[:, 0:2] = anchor[:, 0:2]
  offset = flat[:, 0:3] - origin.unsqueeze(1).expand(-1, span, -1).reshape(-1, 3)

  # The anchor's height is not subtracted and a yaw rotation leaves z alone, so the third
  # component comes out as metres above the floor rather than relative to anything
  out = torch.zeros(count * span, layout.width, device=states.device, dtype=flat.dtype)
  out[:, layout.root_pos] = quat_apply_inverse(spread, offset)
  out[:, layout.root_ori] = rot6d(quat_mul(quat_conjugate(spread), flat[:, 3:7]))
  out[:, layout.root_lin_vel] = quat_apply_inverse(spread, flat[:, 7:10])
  out[:, layout.root_ang_vel] = quat_apply_inverse(spread, flat[:, 10:ROOT_STATE_DIM])
  joints = slice(ROOT_STATE_DIM, ROOT_STATE_DIM + layout.num_joints)
  out[:, layout.joint_pos] = flat[:, joints]
  out[:, layout.joint_vel] = flat[:, ROOT_STATE_DIM + layout.num_joints :]
  if actions is not None:
    out[:, layout.action] = actions.reshape(count * span, -1)
  if bodies is not None and layout.num_bodies:
    out[:, layout.bodies] = bodies.reshape(count * span, -1)
  return out.reshape(count, span, layout.width)


@dataclass
class Normalizer:
  """Per feature mean and standard deviation. Carried in the checkpoint.

  Diffusion wants every channel at roughly unit scale, and these are not: a joint velocity
  runs to 20 rad/s and a body position to 0.3 m. Fitted on the training split only, and
  reused unchanged at inference, because a guidance cost quoted in metres has to mean the
  same thing as the numbers the model was trained on.
  """

  mean: torch.Tensor
  std: torch.Tensor

  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self.mean) / self.std

  def invert(self, x: torch.Tensor) -> torch.Tensor:
    return x * self.std + self.mean

  def to(self, device: str | torch.device) -> Normalizer:
    return Normalizer(mean=self.mean.to(device), std=self.std.to(device))

  @staticmethod
  def fit(features: torch.Tensor, floor: float = 1e-3) -> Normalizer:
    """Mean and deviation over a sample of windows. (N, T, F) -> one Normalizer.

    The floor is what keeps a channel that never moves from being amplified into noise.
    A wrist that stays at its default through every clip has a deviation near zero, and
    dividing by it turns a millimetre of numerical jitter into a feature the model spends
    capacity predicting.
    """
    flat = features.reshape(-1, features.shape[-1])
    return Normalizer(mean=flat.mean(dim=0), std=flat.std(dim=0).clamp(min=floor))


class Windows:
  """Every stretch of the corpus long enough to be one training window.

  An index, not a table of materialised windows. The corpus holds about 140000 states and
  nearly every one of them opens a window, so drawing the rows on demand is the difference
  between a few megabytes and a table the same size as the corpus itself.
  """

  def __init__(
    self,
    data: Dataset,
    history: int,
    horizon: int,
    sources: tuple[str, ...] | None = None,
  ) -> None:
    if history < 1 or horizon < 1:
      raise ValueError("history and horizon are both at least one tick")
    self.data = data
    self.layout = Layout.of(data)
    self.history = history
    self.horizon = horizon
    self.columns = history + horizon
    # A window is columns ticks long, so its two ends are columns - 1 apart. Asking
    # segments for exactly that span leaves starts holding the positions a full window
    # fits at, and available is then constant and unused.
    #
    # Restricting the start rows to a set of clips restricts the whole window to them,
    # since a trajectory never spans two clips. So there is one filter here and not two
    self.segments = data.segments(self.columns - 1, self.columns - 1, data.of(sources))
    self.offsets = torch.arange(self.columns, device=data.states.device)

  def __len__(self) -> int:
    return int(self.segments.starts.numel())

  def rows(self, count: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Dataset rows of `count` random windows. (count, columns)."""
    device = self.segments.starts.device
    picked = torch.randint(0, len(self), (count,), device=device, generator=generator)
    position = self.segments.starts[picked]
    return self.segments.order[position.unsqueeze(-1) + self.offsets]

  def features(self, rows: torch.Tensor) -> torch.Tensor:
    """Encode those windows against their own anchor tick. (count, columns, F).

    The anchor is column history - 1, the last tick of history, which at inference is the
    state the robot is standing in when the plan is drawn. Anchoring anywhere else would
    train the model on a frame it cannot construct at deployment.
    """
    data = self.data
    states = data.states[rows]
    anchor = states[:, self.history - 1]
    actions = None if data.previous_action is None else data.previous_action[rows]
    bodies = None if data.body_pos_b is None else data.body_pos_b[rows]
    return encode(self.layout, states, anchor, actions, bodies)

  def sample(self, count: int) -> torch.Tensor:
    """One batch of encoded windows. (count, columns, F)."""
    return self.features(self.rows(count))
