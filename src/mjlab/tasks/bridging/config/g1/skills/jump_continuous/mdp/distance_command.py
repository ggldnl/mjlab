"""The jump's deployment command: a distance, and nothing about any clip.

At reset, or whenever a new distance is asked for, a target point is fixed that distance
ahead of the robot, along its heading. Every step the command reports how much of it is
left, along the heading. That number is the whole interface of the deployed jump policy.

JumpCommand reports the same number during training, off the clip's landing point, so the
policy distilled there runs here unchanged. Nothing is tracked, nothing is reset onto a
reference, and no clip is loaded.

Run

1. Play the distilled policy with this command, which is what the play configs install.

    uv run play Mjlab-G1-Jump-Continuous

2. Pick a distance with the viewer's "Jump distance" slider and press "Jump".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  sample_uniform,
  yaw_quat,
)

if TYPE_CHECKING:
  from collections.abc import Callable

  import viser

  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


class JumpDistanceCommand(CommandTerm):
  """A target point some distance ahead of the robot, reported as distance left."""

  cfg: JumpDistanceCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: JumpDistanceCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.distance = torch.zeros(self.num_envs, device=self.device)
    self.target_xy = torch.zeros(self.num_envs, 2, device=self.device)
    # Targets are placed after the next forward pass. At reset the robot's pose is written
    # but its world frame is not computed yet, and placing from it would use the old episode
    self._pending = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    self._requested: float | None = None
    self.metrics["remaining_distance"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def remaining_distance(self) -> torch.Tensor:
    """Distance still to cover along the robot's heading, [B, 1]."""
    root = self.robot.data.root_link_pos_w
    offset = torch.zeros(self.num_envs, 3, device=self.device)
    offset[:, :2] = self.target_xy - root[:, :2]
    heading = yaw_quat(self.robot.data.root_link_quat_w)
    return quat_apply_inverse(heading, offset)[:, 0:1]

  @property
  def command(self) -> torch.Tensor:
    return self.remaining_distance

  def jump(self, env_ids: torch.Tensor, distance: float) -> None:
    """Jump this far from where the robot stands now. Placed on the next step."""
    self.distance[env_ids] = distance
    self._pending[env_ids] = True

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self._requested is not None:
      self.distance[env_ids] = self._requested
    else:
      lo, hi = self.cfg.distance_range
      self.distance[env_ids] = sample_uniform(lo, hi, (len(env_ids),), self.device)
    self._pending[env_ids] = True

  def _update_command(self) -> None:
    if not bool(self._pending.any()):
      return
    ids = self._pending.nonzero(as_tuple=True)[0]
    heading = yaw_quat(self.robot.data.root_link_quat_w[ids])
    forward = torch.zeros(len(ids), 3, device=self.device)
    forward[:, 0] = 1.0
    direction = quat_apply(heading, forward)[:, :2]
    self.target_xy[ids] = (
      self.robot.data.root_link_pos_w[ids, :2]
      + self.distance[ids].unsqueeze(-1) * direction
    )
    self._pending[ids] = False

  def _update_metrics(self) -> None:
    self.metrics["remaining_distance"] = self.remaining_distance[:, 0]

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """A distance dial and a button. The whole user facing interface."""
    lo, hi = self.cfg.distance_range
    with server.gui.add_folder(name.capitalize()):
      slider = server.gui.add_slider(
        "Jump distance (m)",
        min=round(lo, 2),
        max=round(hi, 2),
        step=0.05,
        initial_value=round(0.5 * (lo + hi), 2),
      )
      all_envs = server.gui.add_checkbox("All envs", initial_value=True)
      button = server.gui.add_button("Jump")

      @button.on_click
      def _(_) -> None:
        self._requested = float(slider.value)
        if all_envs.value:
          ids = torch.arange(self.num_envs, device=self.device)
        else:
          ids = torch.tensor([get_env_idx()], device=self.device)
        self.jump(ids, self._requested)


@dataclass(kw_only=True)
class JumpDistanceCommandCfg(CommandTermCfg):
  entity_name: str
  distance_range: tuple[float, float]
  """Distances drawn at reset, in metres, until one is picked in the viewer."""

  def build(self, env: ManagerBasedRlEnv) -> JumpDistanceCommand:
    return JumpDistanceCommand(self, env)
