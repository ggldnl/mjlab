"""The MDP terms `parkour_env_cfg.py` assembles a skill out of.

A skill's reward is goal + style, and the two are decoupled exactly as in the AMP
paper:

- The goal is a velocity, [v_fwd, v_lat, v_up, yaw_rate] in the robot's own yaw
  frame. `TwistCommand` samples it from a per-skill box and exposes it as sliders in
  the viewer; the policy observes it, and the three `track_*` rewards pay for
  realizing it. That -- not the reference data -- is what makes the skill
  conditionable: the controller drives a trained skill by writing this command.
- The style is `amp_style_reward`: a discriminator score saying the motion could
  have come from this skill's reference clips. It replaces the hand-written gait
  reward stack (foot clearance, air time, swing height, slip, posture, soft
  landing), each of which encoded somebody's idea of what walking should look like.

Nothing here names a gait. "Run" is the reference folder the discriminator was
pointed at plus the speed range the command box covers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.skills.experiments.parkour.style import (
  G1_KEY_BODY_NAMES,
  Amp,
  features_from_entity,
)
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  import viser

  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# The command channels, in order. Every skill carries all four even though only jump
# uses v_up, because the skill pool evaluates each frozen policy on one shared
# observation: the skills have to agree on their observation space.
COMMAND_CHANNELS = ("v_fwd", "v_lat", "v_up", "yaw_rate")


def heading_frame_velocity(asset: Entity) -> tuple[torch.Tensor, torch.Tensor]:
  """Root linear and angular velocity in the yaw frame the features use.

  The command, the tracking rewards and the AMP features are all expressed in this
  one frame, so "the commanded velocity" and "the realized velocity" are the same
  quantity measured twice rather than two nearly-equal derivations.
  """
  heading = yaw_quat(asset.data.root_link_quat_w)
  return (
    quat_apply_inverse(heading, asset.data.root_link_lin_vel_w),
    quat_apply_inverse(heading, asset.data.root_link_ang_vel_w),
  )


class TwistCommand(CommandTerm):
  """A [v_fwd, v_lat, v_up, yaw_rate] goal, drawn from a uniform per-skill box."""

  cfg: TwistCommandCfg

  def __init__(self, cfg: TwistCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.vel_command_b = torch.zeros(self.num_envs, 4, device=self.device)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_z"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    steps = self.cfg.resampling_time_range[1] / self._env.step_dt
    lin_vel, ang_vel = heading_frame_velocity(self.robot)
    self.metrics["error_vel_xy"] += (
      torch.norm(self.vel_command_b[:, :2] - lin_vel[:, :2], dim=-1) / steps
    )
    self.metrics["error_vel_z"] += (
      torch.abs(self.vel_command_b[:, 2] - lin_vel[:, 2]) / steps
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 3] - ang_vel[:, 2]) / steps
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    for i, channel_range in enumerate(self.cfg.ranges.as_tuple()):
      self.vel_command_b[env_ids, i] = r.uniform_(*channel_range)
    standing = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
    self.vel_command_b[env_ids[standing]] = 0.0

  def _update_command(self) -> None:
    pass

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Sliders to drive the skill by hand: this is the inference interface."""
    from viser import Icon

    sliders: list = []
    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      for label, (low, high) in zip(
        COMMAND_CHANNELS, self.cfg.ranges.as_tuple(), strict=True
      ):
        # Widened past the trained box so it is possible to see where the skill
        # stops extrapolating, and so a pinned channel still has a movable slider.
        sliders.append(
          server.gui.add_slider(
            label,
            min=round(min(low, -0.5) - 0.5, 2),
            max=round(max(high, 0.5) + 0.5, 2),
            step=0.05,
            initial_value=0.0,
          )
        )
      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for slider in sliders:
          slider.value = 0.0

    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, slider in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = slider.value


@dataclass(kw_only=True)
class TwistCommandCfg(CommandTermCfg):
  entity_name: str

  @dataclass
  class Ranges:
    """The box a skill's goal is drawn from -- its main tuning knob.

    Set a channel to (0.0, 0.0) to pin it, which is what walk, run and sprint do
    with v_up: they are not asked to leave the ground. Seed these from the
    statistics `dataset.py` prints for the skill, then widen or narrow by hand --
    asking a skill for a velocity its reference clips never contain puts the goal
    and the style rewards in direct conflict.
    """

    v_fwd: tuple[float, float]
    v_lat: tuple[float, float]
    v_up: tuple[float, float]
    yaw_rate: tuple[float, float]

    def as_tuple(self) -> tuple[tuple[float, float], ...]:
      return (self.v_fwd, self.v_lat, self.v_up, self.yaw_rate)

  ranges: Ranges

  rel_standing_envs: float = 0.0
  """Fraction of envs commanded to hold still. The reference clips barely contain
  standing, so anything above a few percent asks for motion the discriminator has no
  example of."""

  def build(self, env: ManagerBasedRlEnv) -> TwistCommand:
    return TwistCommand(self, env)


class amp_style_reward:
  """Scores each transition against the discriminator, and keeps training it.

  A reward term that learns is unusual, but it keeps every AMP moving part inside
  mjlab's manager system instead of forking the PPO runner. The discriminator is not
  written into the rsl-rl checkpoint, so resuming training restarts it from scratch
  and this reward is briefly meaningless while it recovers. Inference is unaffected:
  only the policy is needed to run a trained skill.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    params = cfg.params
    self._entity: Entity = env.scene[params["entity_name"]]
    self._key_body_indexes = torch.tensor(
      [self._entity.body_names.index(name) for name in G1_KEY_BODY_NAMES],
      dtype=torch.long,
      device=env.device,
    )
    self.amp = Amp(params["clip_dir"], self._key_body_indexes, env.device)
    print(
      f"[amp] {params['clip_dir']}: {self.amp.num_clips} clip(s), "
      f"{self.amp.num_transitions} reference transitions, "
      f"feature dim {self.amp.feature_dim}"
    )

    self._prev_features = torch.zeros(
      env.num_envs, self.amp.feature_dim, device=env.device
    )
    self._has_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._step = 0

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    # A reset teleports the robot, so the pair spanning it is not a transition the
    # robot performed and must not reach the discriminator.
    if env_ids is None:
      env_ids = slice(None)
    self._has_prev[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    clip_dir: str,
    entity_name: str,
    update_every: int,
    num_updates: int,
    batch_size: int,
  ) -> torch.Tensor:
    del clip_dir, entity_name
    # rsl-rl collects its rollout under torch.inference_mode, and everything born in
    # there is an inference tensor autograd refuses to touch -- training the
    # discriminator from a reward term means stepping back out first. It has to cover
    # the replay-buffer writes too, not just the update: a buffer mutated inside
    # inference mode is no longer safe to feed to a backward pass.
    with torch.inference_mode(False):
      features = features_from_entity(self._entity, self._key_body_indexes)
      reward = torch.zeros(env.num_envs, device=env.device)
      if self._has_prev.any():
        prev = self._prev_features[self._has_prev]
        current = features[self._has_prev]
        reward[self._has_prev] = self.amp.style_reward(prev, current)
        self.amp.push(prev, current)

      self._prev_features = features.detach()
      self._has_prev[:] = True

      self._step += 1
      if self._step % update_every == 0:
        self.amp.update(num_updates, batch_size)
    return reward


def track_planar_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track the commanded forward/lateral velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  lin_vel, _ = heading_frame_velocity(asset)
  error = torch.sum(torch.square(command[:, :2] - lin_vel[:, :2]), dim=1)
  return torch.exp(-error / std**2)


def track_vertical_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track the commanded vertical velocity, which is how a jump is asked for.

  Held at a positive value this rewards taking off, not levitating: the robot cannot
  keep rising, so the only way to score repeatedly is to leave the ground
  repeatedly. A sustained v_up command therefore reads as "hop", and a brief pulse
  reads as "one jump".
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  lin_vel, _ = heading_frame_velocity(asset)
  error = torch.square(command[:, 2] - lin_vel[:, 2])
  return torch.exp(-error / std**2)


def track_yaw_rate(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track the commanded turn rate."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  _, ang_vel = heading_frame_velocity(asset)
  error = torch.square(command[:, 3] - ang_vel[:, 2])
  return torch.exp(-error / std**2)
