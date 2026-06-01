"""Ballistic jump abstraction.

The template model for a jump is the simplest possible one: a point mass (the
trunk) on a parabolic free-flight arc. Given the takeoff point, a sampled
landing target in front of the robot, and a sampled apex height, the required
takeoff velocity and flight time follow in closed form from projectile motion.

This abstraction does **not** try to track the whole arc tightly during stance
(where the legs do work and the trunk is *not* ballistic). Instead, it produces
three signals, each meaningful in a different phase of the motion:

- ``takeoff``: at the liftoff instant, how well the trunk velocity matches the
  required takeoff velocity. This single, physically exact signal implicitly
  determines the entire flight - physics does the rest.
- ``tracking``: while airborne, how closely the trunk follows the reference
  parabola. Dense scaffolding to break symmetry early in training.
- ``landing``: at touchdown, how close the trunk lands to the target. The true
  objective.

Phases are derived from the foot contact sensor:

    STANCE (0) --takeoff--> FLIGHT (1) --touchdown--> LANDED (2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.abstraction.abstraction import Abstraction, AbstractionCfg
from mjlab.entity import Entity
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

# Phase encoding.
STANCE = 0
FLIGHT = 1
LANDED = 2
_NUM_PHASES = 3


class JumpAbstraction(Abstraction):
  cfg: JumpAbstractionCfg

  def __init__(self, cfg: JumpAbstractionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.base_body_id = self.robot.body_names.index(cfg.base_body_name)
    self.contact_sensor: ContactSensor = env.scene[cfg.contact_sensor_name]

    device = self.device
    n = self.num_envs

    # Per-episode reference, all in world frame.
    self.start_pos_w = torch.zeros(n, 3, device=device)
    self.target_pos_w = torch.zeros(n, 3, device=device)
    self.takeoff_vel_w = torch.zeros(n, 3, device=device)
    self.flight_time = torch.zeros(n, device=device)
    self.apex_height = torch.zeros(n, device=device)

    # Phase tracking.
    self.phase = torch.zeros(n, dtype=torch.long, device=device)
    self.has_taken_off = torch.zeros(n, dtype=torch.bool, device=device)
    self.time_since_takeoff = torch.zeros(n, device=device)
    self.takeoff_vel_actual_w = torch.zeros(n, 3, device=device)

    self.metrics["landing_error"] = torch.zeros(n, device=device)

    # Pre-populate signals and observations with zeros so their shapes are
    # available before the first ``compute`` (e.g. when the observation manager
    # probes term dimensions at construction).
    self._signals = {
      "takeoff": torch.zeros(n, device=device),
      "tracking": torch.zeros(n, device=device),
      "landing": torch.zeros(n, device=device),
    }
    self._obs = {
      "target": torch.zeros(n, 3, device=device),
      "takeoff_velocity": torch.zeros(n, 3, device=device),
      "phase": torch.zeros(n, _NUM_PHASES, device=device),
    }

  # Convenience masks.

  @property
  def is_airborne(self) -> torch.Tensor:
    """Float mask, 1 while in FLIGHT, else 0. Shape ``(num_envs,)``."""
    return (self.phase == FLIGHT).float()

  @property
  def is_grounded(self) -> torch.Tensor:
    """Float mask, 1 in STANCE or LANDED, else 0. Shape ``(num_envs,)``."""
    return 1.0 - self.is_airborne

  @property
  def is_landed(self) -> torch.Tensor:
    """Float mask, 1 once the robot has completed the jump (LANDED), else 0.

    Distinct from ``is_grounded``, which is also 1 during the initial STANCE.
    Use this to gate terminal rewards so they cannot be farmed by standing
    still at the start without jumping.
    """
    return (self.phase == LANDED).float()

  def target_proximity(self, std: float) -> torch.Tensor:
    """Closeness of the base to the landing target in the xy-plane, in [0, 1].

    ``exp(-||xy - target_xy||^2 / std^2)``. Combined with ``is_landed`` this
    gates terminal rewards to *successful* jumps: a hop in place reaches LANDED
    but lands far from the (across-the-gap) target, so this is ~0 there.
    """
    base_pos_w = self.robot.data.body_link_pos_w[:, self.base_body_id]
    dist_sq = torch.sum(
      torch.square(base_pos_w[:, :2] - self.target_pos_w[:, :2]), dim=-1
    )
    return torch.exp(-dist_sq / std**2)

  # Reference.

  def _resample(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
      return
    device = self.device

    origin = self._env.scene.env_origins[env_ids]  # (n, 3)
    start = origin.clone()
    start[:, 2] = start[:, 2] + self.cfg.nominal_base_height

    target = self._sample_target(env_ids, start)
    apex = torch.empty(n, device=device).uniform_(*self.cfg.apex_height_range)

    v0, flight_time = _solve_ballistic(start, target, apex, gravity=self.cfg.gravity)

    self.start_pos_w[env_ids] = start
    self.target_pos_w[env_ids] = target
    self.takeoff_vel_w[env_ids] = v0
    self.flight_time[env_ids] = flight_time
    self.apex_height[env_ids] = apex

    self.phase[env_ids] = STANCE
    self.has_taken_off[env_ids] = False
    self.time_since_takeoff[env_ids] = 0.0
    self.takeoff_vel_actual_w[env_ids] = 0.0

  def _sample_target(self, env_ids: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    """Sample a landing target for ``env_ids``.

    Prefers the terrain's per-cell landing patches (which sit just past the gap
    on the far platform, so distance scales with the curriculum gap). Falls back
    to a forward/lateral offset from the start when the terrain publishes no
    such patches (e.g. plane terrain).
    """
    n = len(env_ids)
    device = self.device
    terrain = getattr(self._env.scene, "terrain", None)
    patch_name = self.cfg.landing_patch_name
    patches = getattr(terrain, "flat_patches", {}) if terrain is not None else {}

    if terrain is not None and patch_name in patches:
      patch_pos = patches[patch_name]  # (rows, cols, num_patches, 3)
      levels = terrain.terrain_levels[env_ids]
      types = terrain.terrain_types[env_ids]
      num_patches = patch_pos.shape[2]
      p_idx = torch.randint(0, num_patches, (n,), device=device)
      target = patch_pos[levels, types, p_idx].clone()
      target[:, 2] = target[:, 2] + self.cfg.nominal_base_height
      return target

    # Fallback: forward (+x) and lateral offset from the start.
    r = torch.empty(n, device=device)
    forward = r.uniform_(*self.cfg.forward_range).clone()
    lateral = torch.empty(n, device=device).uniform_(*self.cfg.lateral_range)
    target = start.clone()
    target[:, 0] = target[:, 0] + forward
    target[:, 1] = target[:, 1] + lateral
    return target

  # Per-step update.

  def _update(self, dt: float) -> None:
    base_pos_w = self.robot.data.body_link_pos_w[:, self.base_body_id]  # (n, 3)
    base_vel_w = self.robot.data.body_link_lin_vel_w[:, self.base_body_id]  # (n, 3)
    base_quat_w = self.robot.data.body_link_quat_w[:, self.base_body_id]  # (n, 4)

    # Contact: any foot touching ground.
    assert self.contact_sensor.data.found is not None
    in_contact = (self.contact_sensor.data.found > 0).any(dim=-1)  # (n,)
    airborne = ~in_contact

    # Transition: STANCE -> FLIGHT (first time both feet leave the ground).
    just_took_off = (self.phase == STANCE) & airborne
    self.takeoff_vel_actual_w = torch.where(
      just_took_off.unsqueeze(-1), base_vel_w, self.takeoff_vel_actual_w
    )
    self.has_taken_off = self.has_taken_off | just_took_off
    self.phase = torch.where(
      just_took_off, torch.full_like(self.phase, FLIGHT), self.phase
    )

    # Advance flight timer.
    self.time_since_takeoff = torch.where(
      self.phase == FLIGHT,
      self.time_since_takeoff + dt,
      self.time_since_takeoff,
    )

    # Transition: FLIGHT -> LANDED (regained contact after taking off).
    just_landed = (self.phase == FLIGHT) & in_contact
    self.phase = torch.where(
      just_landed, torch.full_like(self.phase, LANDED), self.phase
    )

    self._compute_signals(base_pos_w, just_took_off, just_landed)
    self._compute_obs(base_pos_w, base_quat_w)

    # Metric: landing error at touchdown (logged at episode reset).
    landing_err = torch.norm(base_pos_w[:, :2] - self.target_pos_w[:, :2], dim=-1)
    self.metrics["landing_error"] = torch.where(
      just_landed, landing_err, self.metrics["landing_error"]
    )

  def _compute_signals(
    self,
    base_pos_w: torch.Tensor,
    just_took_off: torch.Tensor,
    just_landed: torch.Tensor,
  ) -> None:
    # Takeoff: match required takeoff velocity at the liftoff instant.
    takeoff_err = torch.sum(
      torch.square(self.takeoff_vel_actual_w - self.takeoff_vel_w), dim=-1
    )
    takeoff = torch.exp(-takeoff_err / self.cfg.takeoff_std**2) * just_took_off.float()

    # Tracking: follow the reference parabola while airborne, back-loaded along
    # the arc. The reference is anchored at the spawn, so at takeoff (tau~0) the
    # robot is trivially "on" the parabola; without back-loading a brief tip
    # that lifts the feet would harvest near-maximal tracking. Weighting by a
    # progress factor that rises from 0 (takeoff) to 1 (landing) makes the
    # reward accrue only as the robot actually travels along the arc, so the
    # more of the trajectory it follows, the more it earns.
    tau = torch.clamp(self.time_since_takeoff, max=self.flight_time)
    ref = _ballistic_point(
      self.start_pos_w, self.takeoff_vel_w, tau, gravity=self.cfg.gravity
    )
    track_err = torch.sum(torch.square(base_pos_w - ref), dim=-1)
    progress = torch.clamp(tau / self.flight_time.clamp(min=1e-3), 0.0, 1.0)
    progress_weight = _progress_weight(progress, self.cfg.tracking_progress_rate)
    tracking = (
      torch.exp(-track_err / self.cfg.tracking_std**2)
      * self.is_airborne
      * progress_weight
    )

    # Landing: land close to the target (xy) at touchdown.
    landing_err = torch.sum(
      torch.square(base_pos_w[:, :2] - self.target_pos_w[:, :2]), dim=-1
    )
    landing = torch.exp(-landing_err / self.cfg.landing_std**2) * just_landed.float()

    self._signals = {
      "takeoff": takeoff,
      "tracking": tracking,
      "landing": landing,
    }

  def _compute_obs(self, base_pos_w: torch.Tensor, base_quat_w: torch.Tensor) -> None:
    target_rel_b = quat_apply_inverse(base_quat_w, self.target_pos_w - base_pos_w)
    takeoff_vel_b = quat_apply_inverse(base_quat_w, self.takeoff_vel_w)
    phase_onehot = torch.nn.functional.one_hot(self.phase, _NUM_PHASES).float()
    self._obs = {
      "target": target_rel_b,
      "takeoff_velocity": takeoff_vel_b,
      "phase": phase_onehot,
    }

  # Visualization.

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = list(visualizer.get_env_indices(self.num_envs))
    if not env_indices:
      return

    start = self.start_pos_w.cpu().numpy()
    v0 = self.takeoff_vel_w.cpu().numpy()
    flight_time = self.flight_time.cpu().numpy()
    g = self.cfg.gravity

    for i in env_indices:
      # Reference parabola (sampled polyline approximated with short arrows).
      ts = np.linspace(0.0, float(flight_time[i]), 12)
      pts = (
        start[i][None, :]
        + v0[i][None, :] * ts[:, None]
        + 0.5 * np.array([0.0, 0.0, -g]) * (ts[:, None] ** 2)
      )
      for a, b in zip(pts[:-1], pts[1:], strict=True):
        visualizer.add_arrow(a, b, color=(1.0, 0.8, 0.1, 0.8), width=0.01)


def _progress_weight(progress: torch.Tensor, rate: float) -> torch.Tensor:
  """Monotonic weight rising from 0 at ``progress=0`` to 1 at ``progress=1``.

  ``rate == 0`` gives a linear ramp. ``rate > 0`` gives a convex (back-loaded)
  exponential ramp ``(exp(rate * p) - 1) / (exp(rate) - 1)`` - the larger the
  rate, the more reward is concentrated near the end of the trajectory.
  """
  if rate == 0.0:
    return progress
  return (torch.exp(rate * progress) - 1.0) / (math.expm1(rate))


def _solve_ballistic(
  start: torch.Tensor,
  target: torch.Tensor,
  apex_height: torch.Tensor,
  gravity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Solve the projectile boundary-value problem.

  Given a start and target position and the desired apex height *above the
  start*, return the takeoff velocity vector and the total flight time.

  Args:
    start: Takeoff position, shape ``(n, 3)``.
    target: Landing position, shape ``(n, 3)``.
    apex_height: Peak height above the start, shape ``(n,)``.
    gravity: Gravitational acceleration (positive scalar).

  Returns:
    ``(v0, flight_time)`` where ``v0`` has shape ``(n, 3)`` and ``flight_time``
    has shape ``(n,)``.
  """
  g = gravity
  dz = target[:, 2] - start[:, 2]
  # Apex must clear both endpoints.
  h = torch.clamp(apex_height, min=torch.clamp(dz, min=0.0) + 1e-3)

  vz0 = torch.sqrt(2.0 * g * h)
  t_up = vz0 / g
  t_down = torch.sqrt(2.0 * (h - dz) / g)
  flight_time = t_up + t_down

  vx0 = (target[:, 0] - start[:, 0]) / flight_time
  vy0 = (target[:, 1] - start[:, 1]) / flight_time
  v0 = torch.stack([vx0, vy0, vz0], dim=-1)
  return v0, flight_time


def _ballistic_point(
  start: torch.Tensor,
  v0: torch.Tensor,
  tau: torch.Tensor,
  gravity: float,
) -> torch.Tensor:
  """Position on the reference parabola at flight time ``tau``. Shape ``(n, 3)``."""
  accel = torch.zeros_like(v0)
  accel[:, 2] = -gravity
  tau = tau.unsqueeze(-1)
  return start + v0 * tau + 0.5 * accel * tau**2


@dataclass(kw_only=True)
class JumpAbstractionCfg(AbstractionCfg):
  entity_name: str
  base_body_name: str
  """Name of the trunk/base body whose trajectory follows the parabola."""
  contact_sensor_name: str
  """Foot-ground contact sensor used to detect takeoff and landing."""

  forward_range: tuple[float, float]
  """Fallback forward (+x) distance to the target, used only when the terrain
  publishes no landing patches (e.g. plane terrain)."""
  lateral_range: tuple[float, float] = (-0.5, 0.5)
  """Fallback lateral (y) offset of the target, used only without landing
  patches. Kept small so the target stays "more or less in front"."""
  landing_patch_name: str = "landing"
  """Name of the terrain flat-patch set to sample landing targets from. When the
  terrain provides it, the target comes from the far platform (distance scales
  with the curriculum gap) instead of ``forward_range``."""
  apex_height_range: tuple[float, float] = (0.2, 0.5)
  """Peak height of the trunk above the takeoff point, in meters."""

  nominal_base_height: float = 0.665
  """Standing height of the base body above the spawn origin, in meters. Used
  as the takeoff and landing reference height."""
  gravity: float = 9.81

  takeoff_std: float = 0.7
  tracking_std: float = 0.3
  landing_std: float = 0.4

  tracking_progress_rate: float = 3.0
  """Back-loading of the tracking signal along the arc. ``0`` is a linear ramp
  from takeoff to landing; larger values concentrate the reward near the end of
  the trajectory, so a brief tip near the start earns almost nothing."""

  def build(self, env: ManagerBasedRlEnv) -> JumpAbstraction:
    return JumpAbstraction(self, env)
