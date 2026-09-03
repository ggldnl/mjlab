"""What the kick adds to the pass: two marked points on the foot, and a strike that has to
be made with the front one while the foot is off the ground.

Everything else comes from `passing.mdp` unchanged and is re-exported here, so a config
reaches both through one namespace the way the pass's own does.

The pass learned to shove the ball along the ground with the sole, because that is the
cheapest way for a standing humanoid to move a 0.425 kg ball and nothing in its reward
distinguished it from a strike. Three things separate the two motions and all three are
measured here:

    where on the foot   the toe leads a strike, the flat of the foot leads a shove
    how fast            a pushing foot moves at the ball's speed, a striking one much faster
    planted or not      a shove needs the foot on the floor, a strike happens in the air

The last does the most work, and it is worth saying why the first cannot do it alone. The
foot's collision geometry is seven capsules lying flat along the sole, every one of them
running the full length of the foot, so there is no separate toe geom a contact could be
sensed on. Worse, a flat footed forward push meets the ball at the front of the foot, which
makes the toe genuinely the nearest point even when the motion is a shove. Asking which
part of the foot touched the ball cannot tell the two apart. Asking whether the foot was
standing on the floor at the time can.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.sensor import ContactSensor

# The pass's whole namespace, which itself re-exports the velocity task's and mjlab's. The
# kick changes a handful of terms out of thirty and inherits the rest
from mjlab.tasks.bridging.experiments.humanoid.skills.passing.mdp import *  # noqa: F401, F403
from mjlab.tasks.bridging.experiments.humanoid.skills.passing.mdp import (
  BALL_RADIUS,
  COMMAND_NAME,
  ROBOT,
  PassCommand,
  ball_pos_w,
  heading_quat,
  pass_window_open,
  phase,
  touching_ball,
)
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


##
# The two points on the foot.
##

STRIKE_BODY = "right_ankle_roll_link"
"""The body both sites hang off. The pass measures from a `right_foot` site sitting at
(0.04, 0, -0.037) in this frame, which is the middle of the sole."""

TOE_SITE = "strike_toe"
SOLE_SITE = "strike_sole"

TOE_OFFSET = (0.12, 0.0, -0.02)
"""Where the toe is, in the striking body's frame, in metres.

Read off the collision geometry rather than chosen: the foot capsules run forward to
x = 0.132 at z = -0.025 with a 0.01 radius, so the front surface of the foot is at 0.142 and
this sits 0.022 inside it, 0.015 above the sole.

Standing flat the ankle body sits 0.035 above the floor, which puts this point at 0.015.
Worth knowing before reading any height off it."""

SOLE_OFFSET = (0.04, 0.0, -0.034)
"""Where the sole is, in the striking body's frame, in metres.

The middle of the flat underside, on the surface rather than the 0.002 below it where the
pass's own site sits. Directly under the ankle and eight centimetres behind the toe, which
is the whole distance this task is trying to tell apart."""

SITE_SIZE = 0.012
"""Radius of both markers, in metres. Large enough to find in the viewer, small enough not
to swallow the foot it is drawn on."""

STRIKE_SPEED = 1.5
"""Toe speed, in m/s, above which a contact counts as a strike rather than a shove.

A foot pushing a ball moves at the ball's speed, which is under a metre per second for
anything that stays a push. Launching a 0.425 kg ball at the low end of the command range
takes a toe doing two to three, so this sits between the two with room either side."""

FOOT_GROUND_SENSOR = "strike_foot_ground"
"""Contact sensor for the striking foot against the floor, declared in kick_env_cfg.py.

Its own sensor rather than a slice of the pass's two footed one, because that one's primary
matches both ankles and the order the two arrive in is a fact about the model's body
ordering. Reading a left or a right out of it by index is the kind of thing that keeps
working until somebody reorders a leg."""


def add_strike_sites(cfg: EntityCfg) -> EntityCfg:
  """Return the robot config with the toe and sole markers added to its spec.

  Python side, by wrapping `spec_fn`, rather than by editing g1.xml. One skill's idea of
  where it strikes from does not belong in an asset that eleven other environments load,
  and a site added here is visible to exactly the task that wants it.

  Both are real sites, so every term below reads position and velocity straight out of
  forward kinematics instead of rotating a hand written offset, and both draw in the
  viewer. They are group 4, which MuJoCo shows by default, unlike the group 5 the G1's own
  sites sit in. Either can be toggled from the viewer's Groups tab.
  """
  build_spec = cfg.spec_fn

  def spec_fn() -> mujoco.MjSpec:
    spec = build_spec()
    foot = spec.body(STRIKE_BODY)
    for name, pos, rgba in (
      (TOE_SITE, TOE_OFFSET, (1.0, 0.25, 0.1, 1.0)),
      (SOLE_SITE, SOLE_OFFSET, (0.1, 0.45, 1.0, 1.0)),
    ):
      foot.add_site(
        name=name,
        pos=pos,
        size=(SITE_SIZE,) * 3,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        rgba=rgba,
        group=4,
      )
    return spec

  return replace(cfg, spec_fn=spec_fn)


##
# Scene readings.
##


def _site_pos_w(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  robot: Entity = env.scene[ROBOT]
  return robot.data.site_pos_w[:, robot.site_names.index(name)]


def toe_pos_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """World position of the toe site, shaped (num_envs, 3)."""
  return _site_pos_w(env, TOE_SITE)


def sole_pos_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """World position of the sole site, shaped (num_envs, 3)."""
  return _site_pos_w(env, SOLE_SITE)


def toe_vel_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """World velocity of the toe site, shaped (num_envs, 3).

  Off the site rather than off the ankle body. A kick is mostly hip, knee and ankle
  rotation, so a point twelve centimetres out from the body frame takes most of its speed
  from the rotation and the body's own linear velocity is the smaller half.
  """
  robot: Entity = env.scene[ROBOT]
  return robot.data.site_lin_vel_w[:, robot.site_names.index(TOE_SITE)]


def toe_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """How fast the toe is moving, shaped (num_envs,)."""
  return torch.norm(toe_vel_w(env), dim=-1)


def toe_pos_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The toe seen from the robot's heading frame, shaped (num_envs, 3)."""
  robot: Entity = env.scene[ROBOT]
  return quat_apply_inverse(
    heading_quat(env), toe_pos_w(env) - robot.data.root_link_pos_w
  )


def toe_vel_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The toe's velocity in the robot's heading frame, shaped (num_envs, 3).

  In the observation because it is what `shove_cost` gates on, for the reason the pass
  gives about its stance window: a term that switches on at a speed the policy cannot see
  changes for no visible reason.
  """
  return quat_apply_inverse(heading_quat(env), toe_vel_w(env))


def toe_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Height of the toe above the floor, shaped (num_envs, 1).

  Foot clearance decides whether a swing meets the ball or the floor first, and the z in
  `toe_pos_b` is measured from a root that moves, so it does not answer the same question.
  """
  return toe_pos_w(env)[:, 2:3]


def _surface_gap(env: ManagerBasedRlEnv, point_w: torch.Tensor) -> torch.Tensor:
  gap = torch.norm(ball_pos_w(env) - point_w, dim=-1) - BALL_RADIUS
  return torch.clamp(gap, min=0.0)


def toe_gap(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Distance from the toe to the ball's surface, shaped (num_envs,).

  The pass's `ball_surface_gap` with the toe in place of the mid-sole site. Same floor at
  zero and same reason for subtracting the radius: a kernel on the centre distance peaks at
  a value no foot can reach.
  """
  return _surface_gap(env, toe_pos_w(env))


def sole_gap(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Distance from the sole to the ball's surface, shaped (num_envs,)."""
  return _surface_gap(env, sole_pos_w(env))


def strike_foot_planted(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Whether the striking foot is standing on the floor, shaped (num_envs,) bool."""
  sensor: ContactSensor = env.scene[FOOT_GROUND_SENSOR]
  found = sensor.data.found
  assert found is not None
  return (found > 0).any(dim=-1)


##
# Episode phase.
##

_KICK_PHASE_ATTR = "_kick_phase"

_FAR = 10.0
"""Stand-in for the smallest gap seen so far, before anything has been seen. Finite rather
than an infinity, so the kernel it feeds returns a clean zero instead of a nan."""


class KickPhase:
  """The closest the toe has come to the ball this episode.

  Latched, for the reason the pass latches its launch velocity, and for one more that
  matters here. The pass pays its approach term on the live gap, so every centimetre the
  foot moves away from the ball is reward the policy has just given up. A backswing is
  exactly that motion: a leg cannot be loaded without first taking the foot further from
  the ball. A live approach kernel therefore charges for the windup, and what the policy
  learns instead is to poke at the ball from wherever it already stands.

  Latching the best gap makes the retreat free. The reward keeps whatever the toe has
  already reached, the leg is free to swing back as far as it likes, and the only thing
  still asking for speed is the ball, which is the thing that should be asking.

  Refreshed lazily on first access within a step, guarded by the env's own step counter,
  exactly like the pass's own tracker.
  """

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self._step = -1
    self.min_gap = torch.full((env.num_envs,), _FAR, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.min_gap[env_ids] = _FAR
    self._step = -1

  def refresh(self, env: ManagerBasedRlEnv) -> None:
    if self._step == env.common_step_counter:
      return
    self._step = env.common_step_counter
    # Only once the stance window is open. Before that the approach pays nothing anyway,
    # and a robot that crept toward the ball during the stance would otherwise bank the
    # small gap and be paid for it from the moment the window opened
    self.min_gap = torch.where(
      pass_window_open(env), torch.minimum(self.min_gap, toe_gap(env)), self.min_gap
    )


def kick_phase(env: ManagerBasedRlEnv) -> KickPhase:
  """The env's kick tracker, created on first use and refreshed once per step."""
  tracker = getattr(env, _KICK_PHASE_ATTR, None)
  if tracker is None:
    tracker = KickPhase(env)
    setattr(env, _KICK_PHASE_ATTR, tracker)
  tracker.refresh(env)
  return tracker


def reset_kick_phase(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Clear the latched gap for envs that just restarted."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  tracker = getattr(env, _KICK_PHASE_ATTR, None)
  if tracker is None:
    tracker = KickPhase(env)
    setattr(env, _KICK_PHASE_ATTR, tracker)
  tracker.reset(env_ids)


##
# Observations.
##


def approach_best(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The smallest toe to ball gap so far this episode, shaped (num_envs, 1).

  What `approach_toe` is actually paid on, so the policy can see it. Without it the
  approach reward stops responding to the live gap once a record is set, which from the
  inside looks like a term that went quiet for no reason.
  """
  return kick_phase(env).min_gap.clamp(max=1.0).unsqueeze(-1)


##
# Rewards.
##


def approach_toe(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
  """Paid on the closest the toe has come to the ball, not where it is now. (num_envs,).

  Gated the way the pass gates its own: paid after the stance window and until the ball has
  been touched, so a foot parked against the ball earns nothing.

  See `KickPhase` for why this reads a latched minimum. The short version is that a live
  kernel charges for a backswing, and the backswing is the motion this task wants.
  """
  score = torch.exp(-torch.square(kick_phase(env).min_gap) / std**2)
  active = pass_window_open(env) & ~phase(env).touched
  return torch.where(active, score, torch.zeros_like(score))


def launch_progress(
  env: ManagerBasedRlEnv, command_name: str = COMMAND_NAME
) -> torch.Tensor:
  """How near the ball got to the commanded speed, ignoring direction. (num_envs,).

  The rung the pass is missing. Its ladder steps from touching the ball, worth one, to
  matching a two dimensional velocity, worth five, and a policy that has just learned to
  touch has no gradient telling it that hitting harder is the way up. This fills the gap
  with the one thing that separates a strike from a shove after the fact: how fast the ball
  left.

  Against the commanded speed rather than a constant, so it saturates at the goal instead
  of asking for ever harder contact, and so a slow command is answered by a slow ball.
  Ignoring direction on purpose, because aim is what `pass_quality` is for and asking for
  both at once is the step that was too large.

  Reads the same latched maximum `pass_quality` does, which the tracker only writes once
  the striking foot has touched, so this needs no gating of its own.
  """
  p = phase(env)
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, PassCommand)
  target = torch.norm(command.target_vel_w, dim=-1).clamp(min=1.0e-3)
  return torch.clamp(p.max_speed / target, max=1.0)


def shove_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Touching the ball with a slow toe. Shaped (num_envs,), to be weighted down.

  A shove is a slow contact held for as long as it takes to move the ball, and a strike is
  a fast one over in a step or two, so this charges per step for the first and costs a
  clean kick almost nothing. Charging the trailing steps of a real strike, once the foot
  has given its speed to the ball and is still in contact, is intentional and is the
  duration half of the argument.
  """
  return (touching_ball(env) & (toe_speed(env) < STRIKE_SPEED)).float()


def sole_strike(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Meeting the ball with the sole rather than the toe. (num_envs,), weighted down.

  Two ways for that to be true and the term charges for either.

  The foot is standing on the floor. This is the one that fires on the motion being stamped
  out, and it is the honest test for it. A flat footed forward push meets the ball at the
  front of the foot, so the toe is the nearest point even though the motion is a shove, and
  no comparison of contact points says otherwise. What does say otherwise is that the foot
  never left the ground. A strike is made in the air.

  The sole is nearer the ball than the toe. Covers the other reading of the same failure,
  where the robot gets over the ball and drags or stands on it rather than pushing it
  along. Rare next to the first, cheap to include, and both sites draw in the viewer, so
  either can be watched rather than trusted.
  """
  planted = strike_foot_planted(env)
  sole_first = sole_gap(env) <= toe_gap(env)
  return (touching_ball(env) & (planted | sole_first)).float()
