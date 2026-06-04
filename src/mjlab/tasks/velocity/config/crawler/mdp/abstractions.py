"""Trot gait-clock + velocity-path abstraction for the crawler.

The crawler is a ~60 g quadruped that tips over before it can stumble onto a
gait through random exploration, so plain velocity-tracking RL collapses to
standing/vibrating exploits. This abstraction injects the *only* domain
knowledge the robot is missing - a periodic diagonal trot and a sense of where
"forward at the commanded speed" is - as a reduced-order template model, and
turns it into dense reward signals and policy observations.

It encodes two coupled reference models:

1. A **gait clock**: a single phase that advances at a fixed cadence. Each leg
   reads the clock through its own offset (the robot's diagonal-trot
   ``LEG_PHASE_OFFSETS``); ``cos(phase + offset) > 0`` means that leg should be
   in stance, otherwise swing. This is the rhythm the policy cannot discover on
   its own.

2. A **velocity path**: a reference point that integrates the *commanded* twist
   in the world frame, leashed to stay a bounded distance ahead of the base.
   Because the carrot keeps moving in the commanded direction, standing still
   steadily loses reward, which is what breaks the standing local optimum. The
   leash stops the reference from running away when the robot briefly can't keep
   up, so the gradient never vanishes.

Signals (each in ``[0, 1]``, surfaced as ordinary positive reward terms):

- ``gait``: agreement between the actual foot-contact pattern and the scheduled
  stance/swing pattern. When the command is ~zero the schedule collapses to
  "all feet planted", so it does not force marching in place while standing.
- ``clearance``: how well each foot's height matches its scheduled vertical
  target (lifted to ``swing_height`` in swing, planted in stance). Gated on a
  nonzero command so a standing robot cannot farm it.
- ``path``: ``exp(-d^2 / std^2)`` for the horizontal distance ``d`` from the base
  to the leashed reference point. The main driver of locomotion.

Observations fed to the policy:

- ``gait_clock``: ``[cos, sin]`` of every leg's phase, so the policy knows the
  rhythm it is being scored against (shape ``2 * num_legs``).
- ``path_error``: the reference point relative to the base, expressed in the
  base frame, plus the heading error (shape ``3``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.abstraction.abstraction import Abstraction, AbstractionCfg
from mjlab.entity import Entity
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class TrotGaitAbstraction(Abstraction):
  cfg: TrotGaitAbstractionCfg

  def __init__(self, cfg: TrotGaitAbstractionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.base_body_id = self.robot.body_names.index(cfg.base_body_name)
    self.foot_site_ids = [
      self.robot.site_names.index(name) for name in cfg.foot_site_names
    ]
    self.num_feet = len(self.foot_site_ids)
    assert len(cfg.leg_phase_offsets) == self.num_feet, (
      "leg_phase_offsets must have one entry per foot site"
    )

    device = self.device
    n = self.num_envs

    self.offsets = torch.tensor(cfg.leg_phase_offsets, device=device)  # (F,)

    # Gait clock and leashed reference path (world frame).
    self.phase = torch.zeros(n, device=device)
    self.ref_pos_w = torch.zeros(n, 2, device=device)
    self.ref_yaw = torch.zeros(n, device=device)

    self.metrics["path_error"] = torch.zeros(n, device=device)

    # Pre-populate so shapes are known before the first ``compute`` (the
    # observation manager probes term dimensions at construction time).
    self._signals = {
      "gait": torch.zeros(n, device=device),
      "clearance": torch.zeros(n, device=device),
      "path": torch.zeros(n, device=device),
    }
    self._obs = {
      "gait_clock": torch.zeros(n, 2 * self.num_feet, device=device),
      "path_error": torch.zeros(n, 3, device=device),
    }

  # Reference.

  def _resample(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    n = len(env_ids)
    # Randomize the phase so envs are spread across the gait cycle.
    self.phase[env_ids] = torch.rand(n, device=self.device) * (2.0 * math.pi)
    # Anchor the reference path to the robot's spawn pose.
    base_pos_w = self.robot.data.body_link_pos_w[env_ids, self.base_body_id]
    self.ref_pos_w[env_ids] = base_pos_w[:, :2]
    self.ref_yaw[env_ids] = self.robot.data.heading_w[env_ids]

  # Per-step update.

  def _update(self, dt: float) -> None:
    base_pos_w = self.robot.data.body_link_pos_w[:, self.base_body_id]  # (n, 3)
    base_quat_w = self.robot.data.body_link_quat_w[:, self.base_body_id]  # (n, 4)
    foot_pos_w = self.robot.data.site_pos_w[:, self.foot_site_ids]  # (n, F, 3)
    heading_w = self.robot.data.heading_w  # (n,)

    command = self._env.command_manager.get_command(self.cfg.command_name)  # (n, 3)
    assert command is not None, f"Command '{self.cfg.command_name}' not found."
    vx, vy, wz = command[:, 0], command[:, 1], command[:, 2]
    speed_cmd = torch.norm(command[:, :2], dim=1) + torch.abs(wz)
    moving = (speed_cmd > self.cfg.command_threshold).float()  # (n,)

    self._advance_clock(dt)
    self._advance_path(dt, base_pos_w, vx, vy, wz)

    desired_stance = self._desired_stance(moving)  # (n, F) bool
    self._compute_signals(foot_pos_w, base_pos_w, desired_stance, moving)
    self._compute_obs(base_pos_w, base_quat_w, heading_w)

    d = torch.norm(self.ref_pos_w - base_pos_w[:, :2], dim=1)
    self.metrics["path_error"] += d

  def _advance_clock(self, dt: float) -> None:
    self.phase = torch.remainder(
      self.phase + 2.0 * math.pi * self.cfg.gait_frequency * dt, 2.0 * math.pi
    )

  def _advance_path(
    self,
    dt: float,
    base_pos_w: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    wz: torch.Tensor,
  ) -> None:
    # Integrate the commanded body-frame twist in the world frame, using the
    # reference heading so the path curves with the angular command.
    self.ref_yaw = wrap_to_pi(self.ref_yaw + wz * dt)
    cos_y, sin_y = torch.cos(self.ref_yaw), torch.sin(self.ref_yaw)
    world_vx = cos_y * vx - sin_y * vy
    world_vy = sin_y * vx + cos_y * vy
    self.ref_pos_w[:, 0] += world_vx * dt
    self.ref_pos_w[:, 1] += world_vy * dt

    # Leash: keep the carrot a bounded distance ahead of the base so the path
    # signal never saturates to zero when the robot lags.
    offset = self.ref_pos_w - base_pos_w[:, :2]  # (n, 2)
    dist = torch.norm(offset, dim=1, keepdim=True).clamp(min=1e-6)
    over = (dist > self.cfg.max_lead).float()
    leashed = base_pos_w[:, :2] + offset / dist * self.cfg.max_lead
    self.ref_pos_w = over * leashed + (1.0 - over) * self.ref_pos_w

  def _desired_stance(self, moving: torch.Tensor) -> torch.Tensor:
    """Scheduled stance mask per leg, ``(n, F)``.

    While moving, follow the trot schedule; while standing, plant every foot.
    """
    phases = self.phase.unsqueeze(1) + self.offsets.unsqueeze(0)  # (n, F)
    trot_stance = torch.cos(phases) > 0.0
    standing = moving.unsqueeze(1) < 0.5
    return trot_stance | standing

  def _compute_signals(
    self,
    foot_pos_w: torch.Tensor,
    base_pos_w: torch.Tensor,
    desired_stance: torch.Tensor,
    moving: torch.Tensor,
  ) -> None:
    sensor = self._env.scene.sensors[self.cfg.contact_sensor_name]
    actual_contact = sensor.data.found.reshape(self.num_envs, -1).float()  # (n, F)

    # Gait: contact-pattern agreement.
    gait = 1.0 - torch.abs(desired_stance.float() - actual_contact).mean(dim=1)

    # Clearance: foot height vs. its scheduled vertical target. Heights are taken
    # relative to the foot's own spawn-stance height so the signal is about the
    # *lift*, not the absolute z.
    foot_z = foot_pos_w[..., 2]  # (n, F)
    target_z = torch.where(
      desired_stance,
      torch.zeros_like(foot_z),
      torch.full_like(foot_z, self.cfg.swing_height),
    )
    z_rel = foot_z - self.cfg.ground_height
    clearance = torch.exp(
      -torch.square(z_rel - target_z) / self.cfg.clearance_std**2
    ).mean(dim=1)
    clearance = clearance * moving

    # Path: how close the base is to the leashed reference carrot.
    d = torch.norm(self.ref_pos_w - base_pos_w[:, :2], dim=1)
    path = torch.exp(-torch.square(d) / self.cfg.path_std**2)

    self._signals = {"gait": gait, "clearance": clearance, "path": path}

  def _compute_obs(
    self,
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
    heading_w: torch.Tensor,
  ) -> None:
    phases = self.phase.unsqueeze(1) + self.offsets.unsqueeze(0)  # (n, F)
    gait_clock = torch.stack([torch.cos(phases), torch.sin(phases)], dim=-1)
    gait_clock = gait_clock.reshape(self.num_envs, -1)  # (n, 2F)

    # Reference offset expressed in the base frame (rotate by -heading).
    offset_w = self.ref_pos_w - base_pos_w[:, :2]  # (n, 2)
    cos_h, sin_h = torch.cos(heading_w), torch.sin(heading_w)
    err_x = cos_h * offset_w[:, 0] + sin_h * offset_w[:, 1]
    err_y = -sin_h * offset_w[:, 0] + cos_h * offset_w[:, 1]
    yaw_err = wrap_to_pi(self.ref_yaw - heading_w)
    path_error = torch.stack([err_x, err_y, yaw_err], dim=-1)  # (n, 3)

    del base_quat_w  # Heading already captures the planar orientation we need.
    self._obs = {"gait_clock": gait_clock, "path_error": path_error}

  # Visualization.

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = list(visualizer.get_env_indices(self.num_envs))
    if not env_indices:
      return
    base_z = self.robot.data.body_link_pos_w[:, self.base_body_id, 2]
    ref = torch.stack([self.ref_pos_w[:, 0], self.ref_pos_w[:, 1], base_z], dim=-1)
    ref_np = ref.cpu().numpy()
    for i in env_indices:
      visualizer.add_sphere(ref_np[i], 0.02, (0.1, 0.9, 0.2, 1.0))


@dataclass(kw_only=True)
class TrotGaitAbstractionCfg(AbstractionCfg):
  entity_name: str = "robot"
  base_body_name: str
  """Floating base body; drives the path reference and the gait-clock frame."""
  foot_site_names: tuple[str, ...]
  """Sites tracked as the feet (world z drives the clearance signal)."""
  leg_phase_offsets: tuple[float, ...]
  """Per-leg phase offset (rad) defining the gait; diagonal trot for the crawler."""

  contact_sensor_name: str = "feet_ground_contact"
  command_name: str = "twist"

  gait_frequency: float = 2.5
  """Trot cadence (Hz). ~20 control steps per cycle at 50 Hz control."""
  swing_height: float = 0.015
  """Target foot lift above the stance height during swing (m)."""
  ground_height: float = 0.0
  """Foot site z at rest on the ground (m); subtracted before scoring clearance."""
  clearance_std: float = 0.01
  """Width of the clearance kernel (m)."""
  path_std: float = 0.05
  """Width of the path-tracking kernel (m). Small, matched to the robot's scale."""
  max_lead: float = 0.08
  """How far ahead of the base the reference carrot is leashed (m)."""
  command_threshold: float = 0.03
  """Command magnitude below which the env counts as standing."""

  def build(self, env: ManagerBasedRlEnv) -> TrotGaitAbstraction:
    return TrotGaitAbstraction(self, env)
