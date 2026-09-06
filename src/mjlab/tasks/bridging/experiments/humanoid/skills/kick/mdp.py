"""What the kick adds to a tracking task: a ball, and a reason to hit it.

The tracking terms come in through the star import below and are not touched. Everything
here is about the ball, and there is less of it than the pass needs, because the reference
already says how to kick. There is no approach reward, no aim command and no phase gate: the
clip covers the walk in, the wind up and the strike, and the ball is placed where that clip
puts the foot. What the terms below add is the difference between a policy that mimes a kick
and one that connects.

The command term is the jump's, subclassed twice over:

    it carries the strike   where the ball goes, which way it should leave and which frame
                            the strike happens on are facts about the clip, measured by
                            dataset.py and stored in it. Reading them off the command is
                            what keeps them out of the config
    it caps its sampling    an episode may not start after the foot has closed on the ball.
                            Reference state initialization draws a frame anywhere in the
                            clip, and the frames around the strike put the foot inside the
                            ball, so a reset there teleports the two together. Contact then
                            happens because of the reset and the strike terms pay for
                            nothing. Frames past the cap are still played through every
                            episode, they are just never started at

Three pieces of episode state, latched:

    touched      a foot has been in contact with the ball. The discrete event that separates
                 miming from kicking
    max_speed    the fastest the ball has gone along the clip's kick direction since that
                 contact. Fastest rather than first, so an early nudge is not locked in as
                 the answer
    spawn_xy     where the ball was put at reset, which the distance metric measures against

Speed along the kick direction rather than raw speed. Raw would pay the same for sending the
ball sideways, and speed along the robot's live heading would let a policy fix a bad strike
by turning to face wherever the ball happened to go.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.asset_zoo.objects.ball import BALL_RADIUS
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

# The tracking terms, the jump's command and mjlab's generic terms, all through one
# namespace. Same layering the martial task uses, which is the environment this one extends
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.mdp import *  # noqa: F401, F403
from mjlab.tasks.bridging.experiments.humanoid.skills.jump_continuous.mdp.commands import (
  JumpCommand,
  JumpCommandCfg,
)
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  sample_uniform,
  yaw_quat,
)

if TYPE_CHECKING:
  from collections.abc import Callable

  import viser

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


##
# Names.
##

ROBOT = "robot"
BALL = "ball"

COMMAND_NAME = "motion"

# Contact sensor, defined in kick_env_cfg.py
FOOT_BALL_SENSOR = "foot_ball_contact"

# Ball speed above which the ball counts as struck rather than jostled, in m/s. Below this
# the ball has been leant on, not kicked
LAUNCH_SPEED = 0.5

_BALL_CFG = SceneEntityCfg(BALL)


##
# Command.
##


class KickCommand(JumpCommand):
  """The jump's motion tracker, plus what the clip says about the ball.

  Adds nothing to the observation and changes nothing about the reference. The two things it
  does are to read the strike out of the clips it was given, and to keep a reset from
  landing on top of the ball. See the module docstring.
  """

  cfg: KickCommandCfg

  def __init__(self, cfg: KickCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)

    required = ("ball_pos", "kick_dir", "reset_limit")
    strikes = []
    for path in cfg.motion_files:
      data = np.load(path)
      missing = [key for key in required if key not in data.files]
      if missing:
        raise KeyError(
          f"{path} carries no {', '.join(missing)}. It predates this task and has to be "
          "converted again: uv run python -m mjlab.tasks.bridging.experiments.humanoid"
          ".skills.kick.dataset"
        )
      strikes.append(data)

    def stack(key: str) -> torch.Tensor:
      return torch.tensor(
        np.stack([np.atleast_1d(d[key]) for d in strikes]),
        dtype=torch.float32,
        device=self.device,
      )

    self.motion_ball_pos = stack("ball_pos")  # (num_motions, 3)
    self.motion_kick_dir = stack("kick_dir")  # (num_motions, 2)
    self.motion_reset_limit = stack("reset_limit").squeeze(-1).long()

    # Set by the viewer's sliders, see create_gui. None means the clip decides
    self.ball_override: torch.Tensor | None = None

    print(
      "Kick clips: "
      + ", ".join(
        f"{m.name} ball ({self.motion_ball_pos[i, 0]:.2f}, "
        f"{self.motion_ball_pos[i, 1]:.2f}) reset to {int(self.motion_reset_limit[i])}"
        for i, m in enumerate(self.motion.metadata)
      )
    )

  @property
  def ball_target(self) -> torch.Tensor:
    """Where the ball goes, per env, in the clip's frame. Shaped (num_envs, 3)."""
    if self.ball_override is not None:
      return self.ball_override.expand(self.num_envs, 3)
    return self.motion_ball_pos[self.motion_ids]

  def move_ball(self, position: torch.Tensor | None) -> None:
    """Put the ball somewhere else, or None to go back to what the clip says.

    For the viewer's sliders. The override is what ball_target reads, so it survives the
    next reset rather than being washed away by the spawn event.
    """
    self.ball_override = position
    if position is None:
      return
    ball: Entity = self._env.scene[BALL]
    root_state = torch.zeros(self.num_envs, 13, device=self.device)
    root_state[:, 0:3] = position + self._env.scene.env_origins
    root_state[:, 2] = BALL_RADIUS
    root_state[:, 3] = 1.0
    ball.write_root_state_to_sim(root_state)

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Two sliders that move the ball, for checking the converter's choice by eye.

    The reference ghost plays the clip whatever the policy is doing, so this is a way to see
    the swing and the ball together and drag one against the other. What the sliders report
    is the position in the clip's own frame, which is what dataset.py's Clip.ball takes: pin
    a position found here by writing it there and converting again.
    """
    del get_env_idx, request_action
    home = self.motion_ball_pos[0].tolist()

    with server.gui.add_folder("Ball"):
      x = server.gui.add_slider(
        "Forward (m)",
        min=round(home[0] - 0.5, 2),
        max=round(home[0] + 0.5, 2),
        step=0.01,
        initial_value=round(home[0], 2),
      )
      y = server.gui.add_slider(
        "Lateral (m)",
        min=round(home[1] - 0.5, 2),
        max=round(home[1] + 0.5, 2),
        step=0.01,
        initial_value=round(home[1], 2),
      )
      readout = server.gui.add_text(
        "Clip.ball", initial_value=f"({home[0]:.2f}, {home[1]:.2f})", disabled=True
      )
      reset_btn = server.gui.add_button("Back to the measured position")

      def _apply() -> None:
        readout.value = f"({x.value:.2f}, {y.value:.2f})"
        self.move_ball(
          torch.tensor(
            [x.value, y.value, BALL_RADIUS], dtype=torch.float32, device=self.device
          )
        )
        if on_change is not None:
          on_change()

      @x.on_update
      def _(_) -> None:
        _apply()

      @y.on_update
      def _(_) -> None:
        _apply()

      @reset_btn.on_click
      def _(_) -> None:
        x.value, y.value = round(home[0], 2), round(home[1], 2)
        readout.value = f"({home[0]:.2f}, {home[1]:.2f})"
        self.move_ball(None)
        if on_change is not None:
          on_change()

  @property
  def kick_direction(self) -> torch.Tensor:
    """Which way a struck ball should leave, per env. Shaped (num_envs, 2)."""
    return self.motion_kick_dir[self.motion_ids]

  def _cap_to_backswing(self, env_ids: torch.Tensor) -> None:
    """Squeeze a sampled frame into the stretch of clip the foot is clear of the ball in.

    Squeezed rather than clipped. Clipping is the obvious way to keep a reset off the ball
    and it is wrong: the strike is halfway through these clips, so every frame after it
    lands on the last legal one, and half of all episodes then begin one frame from contact
    with the foot already travelling at five metres a second. The ball is struck by the
    teleport, and the strike terms pay in full for it.

    Scaling keeps the draw spread over the frames a reset can legally use, and leaves the
    adaptive sampler's weighting intact: a bin that keeps failing still maps to its own
    stretch, just a shorter one.
    """
    motion_ids = self.motion_ids[env_ids]
    limit = self.motion_reset_limit[motion_ids]
    length = torch.clamp(self.motion.time_step_total_per_motion[motion_ids] - 1, min=1)
    self.time_steps[env_ids] = self.time_steps[env_ids] * limit // length

  def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
    super()._adaptive_sampling(env_ids)
    self._cap_to_backswing(env_ids)

  def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
    super()._uniform_sampling(env_ids)
    self._cap_to_backswing(env_ids)


@dataclass(kw_only=True)
class KickCommandCfg(JumpCommandCfg):
  """The jump's command config. Same fields, different term."""

  def build(self, env: ManagerBasedRlEnv) -> KickCommand:
    return KickCommand(self, env)


def command(env: ManagerBasedRlEnv, command_name: str = COMMAND_NAME) -> KickCommand:
  """The kick's command term."""
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, KickCommand)
  return term


##
# Scene readings.
##


def ball_pos_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball position in world coordinates, shaped (num_envs, 3)."""
  ball: Entity = env.scene[BALL]
  return ball.data.root_link_pos_w


def ball_vel_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball linear velocity in world axes, shaped (num_envs, 3)."""
  ball: Entity = env.scene[BALL]
  return ball.data.root_link_lin_vel_w


def root_pos_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Robot root position in world coordinates, shaped (num_envs, 3)."""
  robot: Entity = env.scene[ROBOT]
  return robot.data.root_link_pos_w


def heading_quat(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The robot's yaw only orientation, shaped (num_envs, 4).

  Yaw only, never the full root orientation. The torso pitches hard through a kick, and a
  vector in the full base frame swings with it, so the ball would appear to move when only
  the robot leaned.
  """
  robot: Entity = env.scene[ROBOT]
  return yaw_quat(robot.data.root_link_quat_w)


def touching_ball(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Whether either foot is in contact with the ball, shaped (num_envs,) bool.

  Either foot rather than the striking one. Which leg a clip kicks with is in the clip, not
  in this config, and a support foot that reaches the ball has already lost the reference,
  so the distinction buys nothing and would cost a per-clip sensor.
  """
  sensor: ContactSensor = env.scene[FOOT_BALL_SENSOR]
  found = sensor.data.found
  assert found is not None
  return (found > 0).any(dim=-1)


##
# Episode phase.
##

_PHASE_ATTR = "_kick_phase"


class KickPhase:
  """Per env episode state, refreshed at most once per environment step.

  Rewards and terminations run before the command manager, so this refreshes lazily on
  first access within a step, guarded by the env's own step counter. It is then correct
  whichever manager touches it first.
  """

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    n, device = env.num_envs, env.device
    self._step = -1

    self.touching = torch.zeros(n, dtype=torch.bool, device=device)
    self.touched = torch.zeros(n, dtype=torch.bool, device=device)

    # Fastest the ball has travelled along the kick direction since contact, in m/s
    self.max_speed = torch.zeros(n, device=device)

    # Where the ball was put at reset. Handed in by the event that put it there rather than
    # read back off the ball: a reset event writes to the simulation and the entity buffers
    # are not refreshed until the next forward, so reading here would give the position the
    # ball rolled to last episode
    self.spawn_xy = torch.zeros(n, 2, device=device)

  def reset(self, env_ids: torch.Tensor, spawn_xy: torch.Tensor) -> None:
    """Clear the state for envs that just restarted."""
    self.touching[env_ids] = False
    self.touched[env_ids] = False
    self.max_speed[env_ids] = 0.0
    self.spawn_xy[env_ids] = spawn_xy
    # A refresh is owed: the scene moved under us, so the cached step is stale
    self._step = -1

  def refresh(self, env: ManagerBasedRlEnv, direction: torch.Tensor) -> None:
    if self._step == env.common_step_counter:
      return
    self._step = env.common_step_counter

    self.touching = touching_ball(env)
    self.touched = self.touched | self.touching

    # Gated on a foot having touched the ball, and updated after that flag, so a strike
    # registers on the step it happens. Without the gate anything that moved the ball counts,
    # including the robot falling onto it
    speed = torch.sum(ball_vel_w(env)[:, :2] * direction, dim=-1)
    self.max_speed = torch.where(
      self.touched & (speed > self.max_speed), speed, self.max_speed
    )


def _tracker(env: ManagerBasedRlEnv) -> KickPhase:
  """The env's phase tracker, created on first use. Not refreshed."""
  tracker = getattr(env, _PHASE_ATTR, None)
  if tracker is None:
    tracker = KickPhase(env)
    setattr(env, _PHASE_ATTR, tracker)
  return tracker


def phase(env: ManagerBasedRlEnv, command_name: str = COMMAND_NAME) -> KickPhase:
  """The env's phase tracker, refreshed once per step."""
  tracker = _tracker(env)
  tracker.refresh(env, command(env, command_name).kick_direction)
  return tracker


##
# Observations.
##


def ball_pos_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball position seen from the robot, in its heading frame, shaped (num_envs, 3).

  This is what makes the ball part of the task rather than scenery. The reset scatters both
  the robot and the ball, so where the ball sits relative to the foot that will swing
  changes every episode, and the reference alone does not say by how much.
  """
  return quat_apply_inverse(heading_quat(env), ball_pos_w(env) - root_pos_w(env))


def ball_vel_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball velocity in the robot's heading frame, shaped (num_envs, 3)."""
  return quat_apply_inverse(heading_quat(env), ball_vel_w(env))


def ball_contact(
  env: ManagerBasedRlEnv, command_name: str = COMMAND_NAME
) -> torch.Tensor:
  """Whether the ball has been touched yet this episode, shaped (num_envs, 1).

  A foot mounted contact switch is something a real G1 can have, so this is ordinary rather
  than privileged. It tells the policy which side of the strike it is on.
  """
  return phase(env, command_name).touched.float().unsqueeze(-1)


##
# Rewards.
##


def ball_touched(
  env: ManagerBasedRlEnv, command_name: str = COMMAND_NAME
) -> torch.Tensor:
  """Latched bonus for having made contact at all, shaped (num_envs,).

  The lower rung. A one-shot bonus at the moment of contact is a rounding error against a
  few hundred steps of tracking reward, so it cannot compete with simply tracking well and
  missing. Paying it for the rest of the episode makes connecting worth finding.
  """
  return phase(env, command_name).touched.float()


def kick_power(
  env: ManagerBasedRlEnv, std: float, command_name: str = COMMAND_NAME
) -> torch.Tensor:
  """How hard the ball was sent along the kick direction, shaped (num_envs,).

  Latched on the fastest the ball has gone, so it is paid every step after the strike rather
  than once at it, for the same reason the contact bonus is.

  Saturating rather than linear in speed. A linear term has no ceiling, so the cheapest way
  to raise it is always to swing harder, and past some point that means abandoning the
  reference and falling over. This one is worth most between a nudge and a real strike and
  flattens after, which leaves the tracking terms deciding how the kick looks.
  """
  speed = torch.clamp(phase(env, command_name).max_speed, min=0.0)
  return 1.0 - torch.exp(-torch.square(speed / std))


##
# Metrics.
##


def contact_rate(
  env: ManagerBasedRlEnv, command_name: str = COMMAND_NAME
) -> torch.Tensor:
  """Whether the ball was struck this episode. Report with reduce="last"."""
  return phase(env, command_name).touched.float()


def launch_rate(
  env: ManagerBasedRlEnv, command_name: str = COMMAND_NAME
) -> torch.Tensor:
  """Whether the ball actually left, rather than being leant on. reduce="last"."""
  return (phase(env, command_name).max_speed >= LAUNCH_SPEED).float()


def ball_speed(
  env: ManagerBasedRlEnv, command_name: str = COMMAND_NAME
) -> torch.Tensor:
  """Ball speed along the kick direction, in m/s. Report with reduce="max"."""
  return phase(env, command_name).max_speed


def ball_distance(
  env: ManagerBasedRlEnv, command_name: str = COMMAND_NAME
) -> torch.Tensor:
  """How far the ball has travelled from where it spawned, in m. reduce="max"."""
  offset = ball_pos_w(env)[:, :2] - phase(env, command_name).spawn_xy
  return torch.norm(offset, dim=-1)


##
# Events.
##


def reset_ball_at_strike(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  forward_range: tuple[float, float] = (0.0, 0.0),
  lateral_range: tuple[float, float] = (0.0, 0.0),
  command_name: str = COMMAND_NAME,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
) -> None:
  """Put the ball where the reference's striking foot will pass through it.

  Placed against the clip, not against the robot. The reference is pinned at the env origin
  facing +x and the reset writes the robot onto it, so those are the same frame up to the
  reset noise. Against the clip is the one that stays right whatever order the managers reset
  in: this event runs before the command term has moved the robot, so a foot read here is
  still where the last episode left it.

  Which clip is read is the same story, and the reason a task is one clip. The command has
  not resampled yet either, so the motion ids this indexes with are the ones the last episode
  ran under. With a single clip they are the only ones there are.

  The offsets are what the ball observation is for. At zero the ball sits exactly where the
  converter measured the sole to close on it fastest, and tracking the reference is enough.
  Widened by the curriculum, the reference still connects every time but meets some spawns
  square and grazes others, and a reference-perfect swing's worst case drops from 4.4 m/s to
  2.6. The only thing saying which way to adjust is the ball's position in the observation.
  See the table in kick_env_cfg.py.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

  n = len(env_ids)
  offset = torch.zeros(n, 3, device=env.device)
  offset[:, 0] = sample_uniform(*forward_range, n, env.device)
  offset[:, 1] = sample_uniform(*lateral_range, n, env.device)

  target = command(env, command_name).ball_target[env_ids]
  root_state = torch.zeros(n, 13, device=env.device)
  root_state[:, 0:3] = target + env.scene.env_origins[env_ids] + offset
  root_state[:, 2] = BALL_RADIUS  # Resting on the ground, whatever the offset did.
  root_state[:, 3] = 1.0  # Identity quaternion.

  ball: Entity = env.scene[ball_cfg.name]
  ball.write_root_state_to_sim(root_state, env_ids=env_ids)

  # Handed straight to the tracker, since this is the only place the new position is known
  # without waiting for a forward pass
  _tracker(env).reset(env_ids, root_state[:, :2])


##
# Curriculum helper.
##


class event_curriculum:
  """Widen an event term's parameters as training goes on.

  mjlab's curriculums move reward weights and termination thresholds; nothing moves an
  event's parameters, and where the ball spawns is an event parameter.

  Stages are {"step": int, "params": {...}} and later ones win, matching reward_curriculum.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
    self._term_cfg = env.event_manager.get_term_cfg(cfg.params["event_name"])
    self._stages: list[dict] = cfg.params["stages"]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    stages: list[dict],
  ) -> dict[str, torch.Tensor]:
    del env_ids, event_name, stages
    applied: dict[str, Any] = {}
    for stage in self._stages:
      if env.common_step_counter >= stage["step"]:
        applied = stage["params"]
    self._term_cfg.params.update(applied)
    # Logged so the stage in force is visible next to the metrics it moves. Only the ranges
    # are reported, as their upper bound, which is the number that means anything here
    return {
      key: torch.tensor(float(value[1]))
      for key, value in applied.items()
      if isinstance(value, (tuple, list)) and len(value) == 2
    }
