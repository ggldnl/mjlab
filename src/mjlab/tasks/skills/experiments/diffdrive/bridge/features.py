"""Goal-relative features and the goal-closeness metric.

The reduced state is [x, y, theta, v, omega]. Everything here is expressed
relative to a goal state, so it does not depend on where in the world the switch
happens: the policy sees the same picture at every junction. The functions
broadcast, so they work on a single state, a batch of envs, or a whole window of
candidate goals at once, which is all that training and deployment need.
"""

from __future__ import annotations

import torch

from mjlab.tasks.skills.experiments.diffdrive.bridge import config
from mjlab.tasks.skills.experiments.diffdrive.robot import OMEGA, THETA, V, X, Y


def _wrap(angle: torch.Tensor) -> torch.Tensor:
  """Wrap an angle to (-pi, pi]."""
  return torch.atan2(torch.sin(angle), torch.cos(angle))


def heading_error(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
  """Signed heading from the robot's heading to the goal's."""
  return _wrap(goal[..., THETA] - state[..., THETA])


def position_error(
  state: torch.Tensor, goal: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Forward and lateral offset to the goal, in the robot's heading frame."""
  dx = goal[..., X] - state[..., X]
  dy = goal[..., Y] - state[..., Y]
  cos_t, sin_t = torch.cos(state[..., THETA]), torch.sin(state[..., THETA])
  forward = cos_t * dx + sin_t * dy
  lateral = -sin_t * dx + cos_t * dy
  return forward, lateral


def speed_error(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
  """Goal speed minus current speed."""
  return goal[..., V] - state[..., V]


def observation(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
  """The bridge's observation: where the goal is and how the robot is moving.

  Layout: forward offset, lateral offset, cos and sin of the heading error,
  current speed, current yaw rate, goal speed. Shape: [..., OBS_DIM].
  """
  forward, lateral = position_error(state, goal)
  head = heading_error(state, goal)
  return torch.stack(
    [
      forward,
      lateral,
      torch.cos(head),
      torch.sin(head),
      state[..., V],
      state[..., OMEGA],
      goal[..., V],
    ],
    dim=-1,
  )


def goal_distance(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
  """Velocity-aware closeness of a state to a goal (smaller is closer).

  This is the metric the bridge uses to pick which window state to join: the one
  closest to the interrupt state at the moment of the switch.
  """
  forward, lateral = position_error(state, goal)
  pos = torch.sqrt(forward * forward + lateral * lateral)
  head = torch.abs(heading_error(state, goal))
  spd = torch.abs(speed_error(state, goal))
  return config.W_POS * pos + config.W_HEAD * head + config.W_SPEED * spd


def success(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
  """Whether the robot is within tolerance of the goal in pose and speed."""
  forward, lateral = position_error(state, goal)
  pos = torch.sqrt(forward * forward + lateral * lateral)
  head = torch.abs(heading_error(state, goal))
  spd = torch.abs(speed_error(state, goal))
  return (pos < config.POS_TOL) & (head < config.HEAD_TOL) & (spd < config.SPEED_TOL)


def twist_from_action(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Map a raw policy action to a clamped body twist (v, omega)."""
  v = torch.clamp(
    raw[..., 0] * config.V_SCALE + config.V_OFFSET, config.V_MIN, config.V_MAX
  )
  omega = torch.clamp(
    raw[..., 1] * config.OMEGA_SCALE, -config.OMEGA_MAX, config.OMEGA_MAX
  )
  return v, omega
