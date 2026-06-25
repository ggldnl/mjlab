"""Goal-relative features and a tube-distance metric, shared by training and deployment.

The reduced state is [x, y, theta, v, omega]. The observation is expressed relative to a
goal state, so it does not depend on where in the world the switch happens: the policy
sees the same picture at every junction. The tube distance measures how close a state is
to skill2's tube, normalized by the tube's own spread so it needs no hand-tuned weights.
The functions broadcast, so they work on a single state, a batch of envs, or a whole set
of candidate goals at once.
"""

from __future__ import annotations

import torch

from mjlab.tasks.skills.experiments.diffdrive.experiment import CONFIG
from mjlab.tasks.skills.experiments.diffdrive.robot import OMEGA, THETA, V, X, Y

# State dimensions that define "on the tube": pose and speed, not the fast-changing yaw
# rate (the robot may still be turning as it merges).
MERGE_DIMS = (X, Y, THETA, V)


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


def observation(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
  """The bridge's observation: where the goal is and how the robot is moving.

  Layout: forward offset, lateral offset, cos and sin of the heading error,
  current speed, current yaw rate, goal speed. Shape: [..., 7].
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


def to_reference(states: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
  """Express each state in a reference pose's frame, position-independent.

  states is [..., L, 5] and ref is [..., 5]. The output is [..., L, 6]: forward and lateral
  offset in the reference heading frame, cos and sin of the heading error, then the state's
  own speed and yaw rate. Encoding a window this way makes it look the same wherever in the
  world the switch happens, so the selector and executor see the junction, not the place.
  """
  ref = ref.unsqueeze(-2)  # broadcast the reference over the L frames
  forward, lateral = position_error(ref, states)
  head = heading_error(ref, states)
  return torch.stack(
    [
      forward,
      lateral,
      torch.cos(head),
      torch.sin(head),
      states[..., V],
      states[..., OMEGA],
    ],
    dim=-1,
  )


def window_features(
  end_window: torch.Tensor, start_window: torch.Tensor
) -> torch.Tensor:
  """Encode a couple of windows, both relative to the interrupt, flattened to one vector.

  The interrupt is the last frame of skill1's end window; it is the reference both windows
  are expressed in. The result is [..., (Le + Ls) * 6], the input the selector reads to
  pick a merge frame and the context the executor carries through the bridge.
  """
  ref = end_window[..., -1, :]
  end = to_reference(end_window, ref)
  start = to_reference(start_window, ref)
  return torch.cat([end.flatten(-2), start.flatten(-2)], dim=-1)


def tube_scale(tube: torch.Tensor) -> torch.Tensor:
  """Per-dimension scale of a set of tube states, so the distance is dimensionless.

  The spread (std) of each merge dimension across the tube, floored away from zero. A
  narrow direction (the tube laterally) gets a small scale, so deviations there count for
  more; a long direction (along travel) gets a large scale, so merging anywhere along the
  tube is cheap. This falls out of the data, with no hand-tuned weights.
  """
  flat = tube.reshape(-1, tube.shape[-1])[:, MERGE_DIMS]
  return flat.std(dim=0).clamp_min(1e-3)


def tube_distance(
  state: torch.Tensor, tube: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
  """Normalized distance from state [..., 5] to each tube state [..., K, 5] -> [..., K].

  Per-dimension differences over the merge dimensions (heading wrapped), divided by scale,
  then a 2-norm. Smaller means closer to the tube.
  """
  diff = state.unsqueeze(-2) - tube
  parts = torch.stack(
    [diff[..., X], diff[..., Y], _wrap(diff[..., THETA]), diff[..., V]], dim=-1
  )
  return torch.linalg.vector_norm(parts / scale, dim=-1)


def twist_from_action(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Map a raw policy action to a clamped body twist (v, omega)."""
  v = torch.clamp(
    raw[..., 0] * CONFIG.v_scale + CONFIG.v_offset, CONFIG.v_min, CONFIG.v_max
  )
  omega = torch.clamp(
    raw[..., 1] * CONFIG.omega_scale, -CONFIG.omega_max, CONFIG.omega_max
  )
  return v, omega
