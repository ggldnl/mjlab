"""Environments for the two differential-drive expert skills: drive and turn.
- drive: a high forward speed with the heading target pinned to the heading the
  robot starts the episode with, so the skill only ever learns to hold a straight
  line at speed. It never sees a turn command.
- turn: a genuine heading change to realize (a nonzero target angle) while the
  forward speed target is zero, always starting from rest. It never sees a fast
  approach, so it never has to cope with momentum it did not build up itself.

Both are trained against a single command [forward_speed, heading_error],
where heading_error is the live, wrapped difference to a target heading fixed
at reset. drive keeps that error near zero (hold heading); turn drives a large
initial error down to zero (realize the angle). This mirrors the analytical
experts in dynamics.py exactly: same behavior, so a trained checkpoint and a
hand-written controller are interchangeable in the composition.

Handing turn control at the speed drive cruises at is therefore expected to trace
a wide arc and slip rather than pivot cleanly, which is exactly the failure the
bridge has to remove: commanding four 90 deg turns should walk a square and
return to the start, and a naive hand-off drifts off it instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

# import numpy as np
import torch

from mjlab.asset_zoo.robots.diffdrive.diffdrive_constants import (
  get_diffdrive_robot_cfg,
)
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  bad_orientation,
  base_ang_vel,
  base_lin_vel,
  generated_commands,
  joint_torques_l2,
  joint_vel_rel,
  last_action,
  reset_joints_by_offset,
  reset_root_state_uniform,
  time_out,
)
from mjlab.envs.mdp.actions import JointVelocityAction, JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import wrap_to_pi
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  # from mjlab.viewer.debug_visualizer import DebugVisualizer

##
# Chassis geometry. Mirrors `asset_zoo/robots/diffdrive/xmls/diffdrive.xml`.
##

# Wheel radius, from <geom class="wheel" size="0.06 0.02"/>
WHEEL_RADIUS = 0.06

# Lateral offset of each wheel from the chassis centerline
HALF_TRACK = 0.11

# Bounded acceleration for the wheel velocity command. Expressed as a chassis
# linear acceleration and converted to a wheel angular acceleration; the action
# term ramps the commanded wheel speed at no more than this rate, so the robot
# eases up to a target speed instead of demanding it instantly.
MAX_LIN_ACCEL = 2.5  # [m/s^2]
MAX_WHEEL_ACCEL = MAX_LIN_ACCEL / WHEEL_RADIUS  # [rad/s^2]

# Height of the chassis frame with the wheels resting on the ground
# The asset's initial root state sits at z=0, so resets have to lift the chassis
# by this much or the wheels spawn buried in the floor
CHASSIS_HEIGHT = 0.06

_ROBOT = SceneEntityCfg("robot")
_WHEELS = SceneEntityCfg(
  "robot", joint_names=("left_wheel", "right_wheel"), preserve_order=True
)

# The single command term's name, shared by both skills
COMMAND_NAME = "goal"


##
# Actions.
##


class RateLimitedJointVelocityAction(JointVelocityAction):
  """A wheel velocity action whose target is acceleration-limited.

  The commanded wheel velocity cannot change by more than ``max_accel`` per
  second, so the robot ramps to a target speed rather than demanding it at once.
  This is what gives the diff drive a bounded acceleration: the drive skill eases
  up to cruise (keeping the velocity servo out of saturation, so it always keeps
  the authority to hold heading and does not spin out), and a skill handed a
  fast-moving robot can only spin the wheels down gradually, so forward momentum
  rides through a hand-off as an arc -- the failure the bridge must remove.
  """

  cfg: RateLimitedJointVelocityActionCfg

  def __init__(
    self, cfg: RateLimitedJointVelocityActionCfg, env: ManagerBasedRlEnv
  ) -> None:
    super().__init__(cfg, env)
    # Max change of the velocity target per control step.
    self._max_delta = cfg.max_accel * env.step_dt
    self._applied_target = torch.zeros(
      self.num_envs, self.action_dim, device=self.device
    )

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    delta = torch.clamp(
      self._processed_actions - self._applied_target,
      -self._max_delta,
      self._max_delta,
    )
    self._applied_target = self._applied_target + delta
    self._processed_actions = self._applied_target

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self._applied_target[env_ids] = 0.0


@dataclass(kw_only=True)
class RateLimitedJointVelocityActionCfg(JointVelocityActionCfg):
  max_accel: float
  """Maximum rate of change of the wheel velocity command [rad/s^2]."""

  def build(self, env: ManagerBasedRlEnv) -> RateLimitedJointVelocityAction:
    return RateLimitedJointVelocityAction(self, env)


##
# Command.
##


class HeadingSpeedCommand(CommandTerm):
  """A ``[forward_speed, heading_error]`` command.

  The forward speed is drawn once per episode. The heading target is fixed at
  reset (the heading the robot spawns with, plus a sampled offset), and the
  command exposes the *live* wrapped error to that target every step. A skill
  that drives the error to zero realizes the sampled turn; with a zero offset it
  simply holds its starting heading.
  """

  cfg: HeadingSpeedCommandCfg

  def __init__(self, cfg: HeadingSpeedCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    # command[:, 0] = forward speed, command[:, 1] = wrapped heading error.
    self._command = torch.zeros(self.num_envs, 2, device=self.device)
    self.speed = torch.zeros(self.num_envs, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_speed"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.speed[env_ids] = torch.empty(n, device=self.device).uniform_(
      *self.cfg.ranges.lin_vel_x
    )
    offset = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.heading)
    self.heading_target[env_ids] = wrap_to_pi(
      self.robot.data.heading_w[env_ids] + offset
    )

  def _update_command(self) -> None:
    self._command[:, 0] = self.speed
    self._command[:, 1] = wrap_to_pi(self.heading_target - self.robot.data.heading_w)

  def _update_metrics(self) -> None:
    max_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    self.metrics["error_speed"] += (
      torch.abs(self.speed - self.robot.data.root_link_lin_vel_b[:, 0]) / max_step
    )
    self.metrics["error_heading"] += torch.abs(self._command[:, 1]) / max_step

  """
  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    base_pos = self.robot.data.root_link_pos_w.cpu().numpy()
    target = self.heading_target.cpu().numpy()
    for batch in env_indices:
      pos = base_pos[batch]
      if np.linalg.norm(pos) < 1e-6:
        continue
      start = pos + np.array([0.0, 0.0, 0.15])
      direction = np.array([np.cos(target[batch]), np.sin(target[batch]), 0.0]) * 0.4
      visualizer.add_arrow(
        start, start + direction, color=(0.9, 0.4, 0.1, 0.8), width=0.02
      )
  """


@dataclass(kw_only=True)
class HeadingSpeedCommandCfg(CommandTermCfg):
  entity_name: str

  @dataclass
  class Ranges:
    # Forward speed command range [m/s]
    lin_vel_x: tuple[float, float]

    # Heading offset range [rad], added to the spawn heading to set the target
    heading: tuple[float, float]

  ranges: Ranges

  def build(self, env: ManagerBasedRlEnv) -> HeadingSpeedCommand:
    return HeadingSpeedCommand(self, env)


##
# Rewards.
##


def track_lin_vel_x(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reward tracking of the commanded forward speed."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  error = torch.square(command[:, 0] - asset.data.root_link_lin_vel_b[:, 0])
  return torch.exp(-error / std**2)


def track_heading(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
) -> torch.Tensor:
  """Reward driving the heading error to zero.

  ``command[:, 1]`` already is the live, wrapped error to the target heading, so
  holding it near zero is both "hold a straight line" (drive) and "reach and keep
  the target angle" (turn). The kernel is deliberately wide: a turn starts a full
  target angle away, and a sharp kernel would be flat there and leave nothing to
  climb.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  return torch.exp(-torch.square(command[:, 1]) / std**2)


def lateral_velocity_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Penalize sideways drift, which a rolling differential drive cannot have."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 1])


def wheel_slip_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _WHEELS
) -> torch.Tensor:
  """Penalize the mismatch between wheel surface speed and chassis speed.

  Under pure rolling, each wheel's surface speed r * w_wheel equals the
  forward speed of the chassis at that wheel's contact point. The residual is
  the longitudinal slip, and it is the physical quantity behind "the robot
  slipped and spun".
  """
  asset: Entity = env.scene[asset_cfg.name]
  wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  lin_vel_b = asset.data.root_link_lin_vel_b
  yaw_rate = asset.data.root_link_ang_vel_b[:, 2]

  # Forward speed of the chassis at each wheel contact point. The left wheel
  # sits at y=+HALF_TRACK, the right one at y=-HALF_TRACK.
  chassis_at_left = lin_vel_b[:, 0] - yaw_rate * HALF_TRACK
  chassis_at_right = lin_vel_b[:, 0] + yaw_rate * HALF_TRACK

  slip_left = WHEEL_RADIUS * wheel_vel[:, 0] - chassis_at_left
  slip_right = WHEEL_RADIUS * wheel_vel[:, 1] - chassis_at_right
  return torch.square(slip_left) + torch.square(slip_right)


##
# Environment.
##

# Episode length. The command is resampled once per episode, so a skill sees
# exactly one command per rollout and its regime stays unambiguous
EPISODE_LENGTH_S = 10.0

"""Tilt of the chassis from upright [rad] above which it counts as tipped over.

The termination now fires on the robot falling over, not on spinning out, so a
skill can be pushed to high forward speed (well past 5 m/s) without the episode
ending underneath it. At ~57 deg the chassis is past the point of recovery and
clearly going over; flat high-speed driving never tilts it this far, so it only
triggers when the robot is actually knocked down.
"""
TIP_LIMIT = 1.0


def _make_env_cfg(
  lin_vel_x: tuple[float, float],
  heading: tuple[float, float],
) -> ManagerBasedRlEnvCfg:
  """Build a diffdrive skill environment for one command distribution."""
  robot_cfg = SceneEntityCfg("robot")
  wheels_cfg = SceneEntityCfg(
    "robot", joint_names=("left_wheel", "right_wheel"), preserve_order=True
  )

  ##
  # Observations
  ##

  # The actor and the critic see exactly the same thing. No privileged terms:
  # the trained critic is meant to be reusable as a value estimate on states
  # produced by a bridge in the future, and a critic that needs privileged
  # inputs could not be queried there.
  #
  # The command term (index 10:12) carries the live heading error, so heading is
  # observed here and nowhere else. dynamics.py's analytical turn expert reads the
  # yaw rate at index 5, so the leading base-velocity terms must stay first for
  # that index constant to hold.
  obs_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=base_lin_vel,
      params={"asset_cfg": robot_cfg},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=base_ang_vel,
      params={"asset_cfg": robot_cfg},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "wheel_vel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": wheels_cfg},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "actions": ObservationTermCfg(func=last_action),
    "command": ObservationTermCfg(
      func=generated_commands,
      params={"command_name": COMMAND_NAME},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(terms=obs_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(terms={**obs_terms}, enable_corruption=False),
  }

  ##
  # Actions
  ##

  # The wheels are velocity-servo actuators (see diffdrive_constants.py): the
  # action is a target wheel angular velocity [rad/s], realized as a
  # torque-limited velocity servo. The target is acceleration-limited, so the
  # robot ramps to speed instead of demanding it instantly -- which keeps the
  # servo out of saturation (so it holds heading) and makes forward momentum
  # persist through a skill hand-off.
  actions: dict[str, ActionTermCfg] = {
    "wheel_velocity": RateLimitedJointVelocityActionCfg(
      entity_name="robot",
      actuator_names=("left_wheel", "right_wheel"),
      scale=1.0,
      use_default_offset=False,
      max_accel=MAX_WHEEL_ACCEL,
    ),
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    COMMAND_NAME: HeadingSpeedCommandCfg(
      entity_name="robot",
      resampling_time_range=(EPISODE_LENGTH_S, EPISODE_LENGTH_S),
      debug_vis=True,
      ranges=HeadingSpeedCommandCfg.Ranges(
        lin_vel_x=lin_vel_x,
        heading=heading,
      ),
    ),
  }

  ##
  # Events
  ##

  # Both skills always start from rest. No pushes, no initial velocity: a skill
  # that had been taught to recover from momentum it did not create would
  # already be doing the bridge's job.
  events = {
    "reset_base": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "z": (CHASSIS_HEIGHT, CHASSIS_HEIGHT),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
        "asset_cfg": robot_cfg,
      },
    ),
    "reset_wheels": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": wheels_cfg,
      },
    ),
  }

  ##
  # Rewards
  ##

  # The tracking kernels are deliberately wide. Both are exp(-error^2/std^2),
  # which goes flat once the error is a couple of std away, and an untrained
  # policy starts every episode a full command's worth of error away from the
  # target. Sharper kernels leave nothing to climb.
  rewards = {
    "track_lin_vel_x": RewardTermCfg(
      func=track_lin_vel_x,
      weight=1.5,
      params={"std": 0.5, "command_name": COMMAND_NAME, "asset_cfg": robot_cfg},
    ),
    "track_heading": RewardTermCfg(
      func=track_heading,
      weight=1.5,
      params={"std": 1.0, "command_name": COMMAND_NAME},
    ),
    "lateral_velocity": RewardTermCfg(
      func=lateral_velocity_l2,
      weight=-0.5,
      params={"asset_cfg": robot_cfg},
    ),
    "wheel_slip": RewardTermCfg(
      func=wheel_slip_l2,
      weight=-0.1,
      params={"asset_cfg": wheels_cfg},
    ),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.02),
    "wheel_torques": RewardTermCfg(
      func=joint_torques_l2,
      weight=-0.01,
      params={"asset_cfg": robot_cfg},
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "tipped_over": TerminationTermCfg(
      func=bad_orientation,
      params={"limit_angle": TIP_LIMIT, "asset_cfg": robot_cfg},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_diffdrive_robot_cfg()},
      num_envs=1,
      env_spacing=4.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base",
      distance=2.5,
      elevation=-25.0,
      azimuth=135.0,
    ),
    sim=SimulationCfg(
      nconmax=8,  # Two wheels and a caster against the plane.
      njmax=32,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,  # 50 Hz control.
    episode_length_s=EPISODE_LENGTH_S,
  )


def _apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  cfg.episode_length_s = 1e10
  cfg.observations["actor"].enable_corruption = False
  return cfg


def drive_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Drive straight at a high forward speed, holding heading.

  The heading offset is pinned to zero, so the target heading is the one the
  robot spawns with and the skill only ever learns to hold a straight line. It
  is never once asked to turn. The forward speed is high (relative to turn's),
  so it builds the momentum a naive hand-off then fails to shed.
  """
  cfg = _make_env_cfg(lin_vel_x=(0.2, 10.0), heading=(0.0, 0.0))
  return _apply_play_overrides(cfg) if play else cfg


def turn_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Realize a target heading while creeping forward on an arc.

  The forward speed target is low but nonzero, so the skill turns while moving
  and the robot describes an arc rather than pivoting in place. It stays well
  below what drive cruises at, and every episode starts from rest, so the skill
  never has to turn while carrying drive's momentum. The heading offset spans
  both directions up to ~150 deg, which covers the 90 deg turns the demo
  commands.
  """
  cfg = _make_env_cfg(lin_vel_x=(0.05, 0.12), heading=(-2.6, 2.6))
  return _apply_play_overrides(cfg) if play else cfg


##
# RL config.
##


def carlike_ppo_runner_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
  """PPO config shared by both diffdrive skills.

  obs_normalization is off on purpose. It keeps the critic a plain function
  of the raw observation, so a trained skill's value function can later be
  evaluated on states a bridge produces without having to carry a running
  normalizer whose statistics were collected on a different distribution.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=500,
  )
