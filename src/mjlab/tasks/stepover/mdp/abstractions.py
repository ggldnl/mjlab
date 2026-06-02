"""Swing-foot step-over abstraction.

The robot starts standing still right in front of a barrier and must step over
it in place - one leg at a time - and end standing still on the far side. There
is no walking approach; the whole episode is the crossing maneuver.

The template model is a *via-point* for each foot: to clear a barrier of known
position and height, the swing foot must pass over a waypoint sitting above the
barrier top. The foot is actuated, so this reference is **imposed kinematic
scaffolding** rather than exact physics (unlike a free-flight arc, where the
body is genuinely underactuated). It still gives RL a cheap, dense gradient
toward the deliberate one-leg-at-a-time high step that nothing else rewards.

The abstraction produces two dense signals, each meaningful continuously rather
than as a sparse one-shot spike:

- ``clearance``: while a foot is horizontally over the barrier, how far it has
  lifted toward its via-point height. This is the core "step over" signal.
- ``cross``: fraction of feet currently on the far side. Rewards actually being
  across (it drops if a foot comes back), so it ramps up through the maneuver
  and saturates once the robot stands beyond the barrier.

A per-foot ``crossed`` latch (did this foot ever reach the far side) drives the
phase encoding and the curriculum:

    APPROACH (0) --one foot over--> CROSSING (1) --both feet over--> CROSSED (2)

The "settle standing on the far side" objective is supplied by reward terms
gated on ``is_beyond`` (the trunk has passed the barrier); see ``mdp.rewards``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.abstraction.abstraction import Abstraction, AbstractionCfg
from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

# Phase encoding.
APPROACH = 0
CROSSING = 1
CROSSED = 2
_NUM_PHASES = 3


class StepOverAbstraction(Abstraction):
  cfg: StepOverAbstractionCfg

  def __init__(self, cfg: StepOverAbstractionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.base_body_id = self.robot.body_names.index(cfg.base_body_name)
    self.foot_site_ids = [
      self.robot.site_names.index(name) for name in cfg.foot_site_names
    ]
    self.num_feet = len(self.foot_site_ids)

    device = self.device
    n = self.num_envs

    # Per-episode reference (world frame): the barrier top-center and height.
    self.barrier_pos_w = torch.zeros(n, 3, device=device)

    # Phase / state tracking.
    self.crossed = torch.zeros(n, self.num_feet, dtype=torch.bool, device=device)
    self.phase = torch.zeros(n, dtype=torch.long, device=device)
    self.beyond = torch.zeros(n, dtype=torch.bool, device=device)

    self.metrics["crossed_feet"] = torch.zeros(n, device=device)

    # Pre-populate so shapes are available before the first ``compute`` (the
    # observation manager probes term dimensions at construction).
    self._signals = {
      "clearance": torch.zeros(n, device=device),
      "cross": torch.zeros(n, device=device),
    }
    self._obs = {
      "barrier": torch.zeros(n, 3, device=device),
      "via_points": torch.zeros(n, 3 * self.num_feet, device=device),
      "phase": torch.zeros(n, _NUM_PHASES, device=device),
      "feet_crossed": torch.zeros(n, self.num_feet, device=device),
    }

  # Convenience masks.

  @property
  def has_crossed(self) -> torch.Tensor:
    """Float mask, 1 once both feet have reached the far side. ``(num_envs,)``."""
    return (self.phase == CROSSED).float()

  @property
  def is_beyond(self) -> torch.Tensor:
    """Float mask, 1 while the trunk is past the barrier. ``(num_envs,)``.

    Gates the "settle standing on the far side" rewards so they cannot be farmed
    by standing on the near side.
    """
    return self.beyond.float()

  # Reference.

  def _resample(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    self.barrier_pos_w[env_ids] = self._sample_barrier(env_ids)
    self.crossed[env_ids] = False
    self.phase[env_ids] = APPROACH
    self.beyond[env_ids] = False

  def _sample_barrier(self, env_ids: torch.Tensor) -> torch.Tensor:
    """Barrier top-center (world frame) for ``env_ids``.

    Prefers the terrain's per-cell ``barrier`` patch, whose height scales with
    the curriculum level. Falls back to a fixed forward offset and height from
    the spawn origin when the terrain publishes no such patch (e.g. plane).
    """
    terrain = getattr(self._env.scene, "terrain", None)
    patch_name = self.cfg.barrier_patch_name
    patches = getattr(terrain, "flat_patches", {}) if terrain is not None else {}

    if terrain is not None and patch_name in patches:
      patch_pos = patches[patch_name]  # (rows, cols, num_patches, 3)
      levels = terrain.terrain_levels[env_ids]
      types = terrain.terrain_types[env_ids]
      return patch_pos[levels, types, 0].clone()

    origin = self._env.scene.env_origins[env_ids].clone()
    origin[:, 0] = origin[:, 0] + self.cfg.barrier_offset
    origin[:, 2] = origin[:, 2] + self.cfg.barrier_height_fallback
    return origin

  # Per-step update.

  def _update(self, dt: float) -> None:
    del dt
    base_pos_w = self.robot.data.body_link_pos_w[:, self.base_body_id]  # (n, 3)
    base_quat_w = self.robot.data.body_link_quat_w[:, self.base_body_id]  # (n, 4)
    foot_pos_w = self.robot.data.site_pos_w[:, self.foot_site_ids]  # (n, F, 3)

    barrier_x = self.barrier_pos_w[:, 0:1]  # (n, 1)

    foot_x = foot_pos_w[..., 0]  # (n, F)
    foot_z = foot_pos_w[..., 2]  # (n, F)

    # Latch per-foot crossing once the foot is clear past the barrier.
    self.crossed = self.crossed | (foot_x > (barrier_x + self.cfg.cross_margin))
    num_crossed = self.crossed.sum(dim=1)  # (n,)
    self.phase = torch.clamp(num_crossed, max=CROSSED).long()
    self.beyond = base_pos_w[:, 0] > barrier_x[:, 0]

    self._compute_signals(foot_x, foot_z)
    self._compute_obs(base_pos_w, base_quat_w, foot_pos_w)

    self.metrics["crossed_feet"] = self.crossed.float().mean(dim=1)

  def _compute_signals(self, foot_x: torch.Tensor, foot_z: torch.Tensor) -> None:
    barrier_x = self.barrier_pos_w[:, 0:1]  # (n, 1)
    target_z = self.barrier_pos_w[:, 2:3] + self.cfg.clearance  # (n, 1)

    clearance = _clearance_signal(
      foot_x,
      foot_z,
      barrier_x,
      target_z,
      crossing_band=self.cfg.crossing_band,
      clearance_std=self.cfg.clearance_std,
      air_threshold=self.cfg.air_threshold,
    )

    # Cross: fraction of feet *currently* on the far side. Drops if a foot comes
    # back, so it cannot be farmed by lifting one foot over and retracting it.
    cross = (foot_x > barrier_x).float().mean(dim=1)

    self._signals = {"clearance": clearance, "cross": cross}

  def _compute_obs(
    self,
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
    foot_pos_w: torch.Tensor,
  ) -> None:
    # Vector from base to the barrier (at the base's own lateral position, since
    # the barrier spans the full width), expressed in the base frame.
    barrier_point = torch.stack(
      [self.barrier_pos_w[:, 0], base_pos_w[:, 1], self.barrier_pos_w[:, 2]], dim=-1
    )
    barrier_rel_b = quat_apply_inverse(base_quat_w, barrier_point - base_pos_w)

    # Per-foot via-point (over the barrier at the foot's lateral position),
    # expressed relative to that foot in the base frame.
    via_w = foot_pos_w.clone()
    via_w[..., 0] = self.barrier_pos_w[:, 0:1]
    via_w[..., 2] = self.barrier_pos_w[:, 2:3] + self.cfg.clearance
    via_rel = via_w - foot_pos_w  # (n, F, 3)
    quat_expand = base_quat_w.unsqueeze(1).expand(-1, self.num_feet, -1)
    via_rel_b = quat_apply_inverse(quat_expand, via_rel).reshape(self.num_envs, -1)

    phase_onehot = torch.nn.functional.one_hot(self.phase, _NUM_PHASES).float()

    self._obs = {
      "barrier": barrier_rel_b,
      "via_points": via_rel_b,
      "phase": phase_onehot,
      "feet_crossed": self.crossed.float(),
    }

  # Visualization.

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = list(visualizer.get_env_indices(self.num_envs))
    if not env_indices:
      return

    foot_pos_w = self.robot.data.site_pos_w[:, self.foot_site_ids]  # (n, F, 3)
    via_w = foot_pos_w.clone()
    via_w[..., 0] = self.barrier_pos_w[:, 0:1]
    via_w[..., 2] = self.barrier_pos_w[:, 2:3] + self.cfg.clearance
    via_np = via_w.cpu().numpy()

    radius = 0.04
    green = (0.1, 0.9, 0.2, 1.0)
    for i in env_indices:
      for f in range(self.num_feet):
        visualizer.add_sphere(via_np[i, f], radius, green)


def _clearance_signal(
  foot_x: torch.Tensor,
  foot_z: torch.Tensor,
  barrier_x: torch.Tensor,
  target_z: torch.Tensor,
  crossing_band: float,
  clearance_std: float,
  air_threshold: float,
) -> torch.Tensor:
  """Swing-foot clearance reward over the barrier, in ``[0, 1]``.

  For each foot horizontally within ``crossing_band`` of the barrier and lifted
  above ``air_threshold``, score ``exp(-deficit^2 / std^2)`` where ``deficit`` is
  how far the foot sits *below* its via-point height ``target_z`` (clearing
  higher is free). The per-env reward averages over the feet currently crossing,
  and is ``0`` when no foot is over the barrier.

  Args:
    foot_x: Foot x-positions, shape ``(n, F)``.
    foot_z: Foot heights, shape ``(n, F)``.
    barrier_x: Barrier x-position, shape ``(n, 1)``.
    target_z: Via-point height (barrier top + clearance), shape ``(n, 1)``.

  Returns:
    Per-env clearance reward, shape ``(n,)``.
  """
  in_band = (torch.abs(foot_x - barrier_x) < crossing_band).float()
  swinging = (foot_z > air_threshold).float()
  weight = in_band * swinging  # (n, F)
  deficit = torch.clamp(target_z - foot_z, min=0.0)  # (n, F)
  per_foot = torch.exp(-torch.square(deficit) / clearance_std**2)
  return (per_foot * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)


@dataclass(kw_only=True)
class StepOverAbstractionCfg(AbstractionCfg):
  entity_name: str
  base_body_name: str
  """Trunk/base body, used for the barrier observation and the far-side gate."""
  foot_site_names: tuple[str, ...]
  """Sites tracked as the feet (their world positions drive the via-points)."""

  barrier_patch_name: str = "barrier"
  """Terrain flat-patch set holding the per-cell barrier top-center and height.
  When present, the barrier height scales with the curriculum level."""
  barrier_offset: float = 0.4
  """Fallback forward (+x) distance from the spawn to the barrier, used only when
  the terrain publishes no barrier patch (e.g. plane terrain)."""
  barrier_height_fallback: float = 0.15
  """Fallback barrier top height, used only without a barrier patch."""

  clearance: float = 0.12
  """Desired swing-foot height above the barrier top at the via-point, meters."""
  clearance_std: float = 0.1
  """Width of the clearance reward kernel, meters."""
  crossing_band: float = 0.25
  """Half-width in x around the barrier within which a foot is "over" it and its
  clearance is scored, meters."""
  air_threshold: float = 0.03
  """Foot height above ground past which it counts as a (swinging) foot whose
  clearance should be scored, meters."""
  cross_margin: float = 0.1
  """Distance past the barrier a foot must reach to latch as crossed, meters."""

  def build(self, env: ManagerBasedRlEnv) -> StepOverAbstraction:
    return StepOverAbstraction(self, env)
