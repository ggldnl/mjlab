"""Observation, reward, termination and event terms for the table tennis skills.

The reference implementation (`reference/jaco_*.py`, from "Composing Complex Skills by
Learning Transition Policies") is inspiration only. It drives a Jaco arm whose fingers
grasp a box, and almost none of its reward machinery survives contact with a racket:
there is nothing to grasp, so "pick" and "hold" have no analogue, and a ball meeting a
rigid blade bounces where a grasped box simply stays put. What is kept is the shape of
the idea -- a small set of independently trained primitives, each with a dense approach
term and a sparse outcome -- not its formulas.

**The ball may only ever touch the racket.** Not the floor, not the arm, in any of the
four tasks. This is the constraint the whole set is built around, and it is enforced
with contact sensors (see `table_tennis_scene_cfg`): one watching ball-against-racket,
one watching ball-against-anything. An illegal contact is anything the second sees that
the first does not. Nothing is exempt -- `hit` ends its episode at the ball's apex,
precisely so the ball never gets the chance to reach the floor.

**The unified command.** Every task observes one 3D command, and it always means the
same thing: *where the ball should end up*. Catch settles it at that point, balance
holds it there, and toss and hit both send the ball's apex there. That shared meaning
is what lets one observation space serve all four skills, which a bridge needs.

The reference reward functions are stage machines carrying per-episode flags that
latch as the episode progresses. That idea does survive, in `TaskPhase` below: mjlab
reward terms are plain stateless functions, so per-episode state lives there, one
instance per env, refreshed once per step.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactSensor

##
# Geometry and thresholds.
##

BALL_RADIUS = 0.02

# The racket's blade centre. Not "racket_attachment_site", which sits at the handle
# where the racket welds to the arm, about 0.2 m short of the blade.
FACE_SITE = "racket_face"

# Ball centre offset along the blade normal when resting on it: blade half-thickness
# plus the ball radius, plus a little clearance.
BLADE_REST_OFFSET = 0.026

# The blade is ~0.15 m across, so this is roughly its usable radius. Used as "the ball
# is over the face" rather than merely somewhere near the racket.
BLADE_RADIUS = 0.07

# Ball centre height at which it is resting on the floor.
GROUND_HEIGHT = BALL_RADIUS + 0.005

# Contact sensor names, defined in `table_tennis_scene_cfg`.
RACKET_CONTACT_SENSOR = "ball_racket_contact"
ANY_CONTACT_SENSOR = "ball_any_contact"

_ROBOT = SceneEntityCfg("robot")
_BALL = SceneEntityCfg("ball")

COMMAND_NAME = "target"


##
# Scene readings.
##


def _face_index(robot: Entity) -> int:
  """Index of the blade-centre site among the robot's sites.

  Looked up by name rather than through a `SceneEntityCfg`, whose `site_ids` are only
  resolved when the cfg is passed through a term's `params`; these helpers are called
  from rewards, terminations and the phase tracker, which take no such params.
  """
  return robot.site_names.index(FACE_SITE)


def face_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
  """World position of the blade centre, shaped (num_envs, 3)."""
  robot: Entity = env.scene["robot"]
  return robot.data.site_pos_w[:, _face_index(robot)]


def face_normal(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Unit normal of the blade face (its site frame's local z), shaped (num_envs, 3)."""
  robot: Entity = env.scene["robot"]
  quat = robot.data.site_quat_w[:, _face_index(robot)]
  w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
  return torch.stack(
    [
      2.0 * (x * z + w * y),
      2.0 * (y * z - w * x),
      1.0 - 2.0 * (x * x + y * y),
    ],
    dim=-1,
  )


def ball_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball position, shaped (num_envs, 3).

  No `env_origins` offset, and there must not be one: the arm is fixed-base, so it is
  anchored at the model origin in every world and its racket sits at identical
  coordinates in all of them. World coordinates are therefore already the arm's own
  frame. Offsetting the ball by `env_origins` would fling it away from the arm in
  every env whose origin is not zero.
  """
  ball: Entity = env.scene["ball"]
  return ball.data.root_link_pos_w


def ball_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball linear velocity in world axes, shaped (num_envs, 3)."""
  ball: Entity = env.scene["ball"]
  return ball.data.root_link_lin_vel_w


def ball_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball speed, shaped (num_envs,)."""
  return torch.norm(ball_vel(env), dim=-1)


def dist_ball_face(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Distance from the ball to the blade centre, shaped (num_envs,)."""
  return torch.norm(ball_pos(env) - face_pos(env), dim=-1)


##
# Contact.
##


def _sensor_found(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[name]
  found = sensor.data.found
  assert found is not None, f"Contact sensor '{name}' does not report 'found'."
  return found.reshape(env.num_envs, -1).any(dim=-1)


def touching_racket(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The ball is in contact with the racket, shaped (num_envs,)."""
  return _sensor_found(env, RACKET_CONTACT_SENSOR)


def touching_anything(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The ball is in contact with any geom at all, shaped (num_envs,)."""
  return _sensor_found(env, ANY_CONTACT_SENSOR)


def ball_on_ground(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The ball has reached the floor.

  Height rather than a contact test, so it fires the moment the ball arrives rather
  than on whichever substep the solver happens to register a contact.
  """
  return ball_pos(env)[:, 2] < GROUND_HEIGHT


##
# Phase tracking.
##

# The env has no slot for this, so the tracker is attached dynamically under a
# namespaced attribute. Accessed via getattr/setattr rather than `env._table_tennis...`
# so it does not read as an attribute the env class is expected to declare.
_PHASE_ATTR = "_table_tennis_phase"


class TaskPhase:
  """Per-env episode state, refreshed at most once per environment step.

  mjlab computes terminations and rewards *before* the command manager runs, so this
  cannot live in a `CommandTerm` without every reward reading a one-step-stale stage.
  It updates lazily on first access within a step, guarded by the env's own step
  counter, which makes it correct no matter which manager touches it first.
  """

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    n, device = env.num_envs, env.device
    self._step = -1
    # Live, recomputed every step.
    self.on_racket = torch.zeros(n, dtype=torch.bool, device=device)
    # Latched for the rest of the episode once they happen.
    self.touched = torch.zeros(n, dtype=torch.bool, device=device)
    self.illegal = torch.zeros(n, dtype=torch.bool, device=device)
    self.launched = torch.zeros(n, dtype=torch.bool, device=device)
    self.falling = torch.zeros(n, dtype=torch.bool, device=device)
    # Running quantities.
    self.max_height = torch.zeros(n, device=device)
    # The apex of a struck/tossed ball: where it stopped rising, and how fast it was
    # still moving there. `hit` is scored on both.
    self.apex_recorded = torch.zeros(n, dtype=torch.bool, device=device)
    self.apex_pos = torch.zeros(n, 3, device=device)
    self.apex_speed = torch.zeros(n, device=device)
    self.settled_steps = torch.zeros(n, dtype=torch.long, device=device)
    self.off_racket_steps = torch.zeros(n, dtype=torch.long, device=device)
    self.unsettled_steps = torch.zeros(n, dtype=torch.long, device=device)

  def reset(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    """Clear the state for the given envs, at the start of their new episode."""
    for flag in (
      self.on_racket,
      self.touched,
      self.illegal,
      self.launched,
      self.falling,
      self.apex_recorded,
    ):
      flag[env_ids] = False
    self.apex_pos[env_ids] = 0.0
    self.apex_speed[env_ids] = 0.0
    self.max_height[env_ids] = ball_pos(env)[env_ids, 2]
    self.settled_steps[env_ids] = 0
    self.off_racket_steps[env_ids] = 0
    self.unsettled_steps[env_ids] = 0
    # A refresh is owed: the scene moved under us, so the cached step is stale.
    self._step = -1

  def refresh(self, env: ManagerBasedRlEnv) -> None:
    if self._step == env.common_step_counter:
      return
    self._step = env.common_step_counter

    pos = ball_pos(env)
    height = pos[:, 2]
    vel = ball_vel(env)
    speed = torch.norm(vel, dim=-1)

    self.on_racket = touching_racket(env)
    self.touched = self.touched | self.on_racket

    # Illegal: the ball touched something that was not the racket -- the floor or the
    # arm, both equally a failure. No task is exempt.
    self.illegal = self.illegal | (touching_anything(env) & ~self.on_racket)

    # Launched: the ball has left the racket travelling upward, having touched it.
    self.launched = self.launched | (self.touched & ~self.on_racket & (vel[:, 2] > 0.2))

    # Apex tracking, and whether the ball is past it.
    self.max_height = torch.maximum(self.max_height, height)
    self.falling = self.falling | (self.launched & (height < self.max_height - 0.01))

    # Apex: the first step after a launch where the ball stops rising. Recorded once,
    # because that instant is the whole outcome for `hit` -- where the ball ended up
    # and how much speed it still carried when it got there.
    just_apexed = self.launched & ~self.apex_recorded & (vel[:, 2] <= 0.0)
    if bool(just_apexed.any()):
      self.apex_pos[just_apexed] = pos[just_apexed]
      self.apex_speed[just_apexed] = speed[just_apexed]
      self.apex_recorded = self.apex_recorded | just_apexed

    # How long the ball has been off the bat. A ball that is merely bouncing comes
    # back within a few steps; one that has been gone for longer is gone for good.
    self.off_racket_steps = torch.where(
      self.on_racket,
      torch.zeros_like(self.off_racket_steps),
      self.off_racket_steps + 1,
    )

    # How long it has failed to simply sit there: not touching, or touching but moving.
    # This is what catches a juggled ball, which stays nominally in contact.
    resting = self.on_racket & (speed < BALANCE_SETTLED_SPEED)
    self.unsettled_steps = torch.where(
      resting, torch.zeros_like(self.unsettled_steps), self.unsettled_steps + 1
    )

    # Settled: resting quietly on the blade. This is what `catch` is trying to reach.
    settled = self.on_racket & (speed < CATCH_SPEED_THRESHOLD)
    self.settled_steps = torch.where(
      settled, self.settled_steps + 1, torch.zeros_like(self.settled_steps)
    )


def phase(env: ManagerBasedRlEnv) -> TaskPhase:
  """The env's phase tracker, created on first use and refreshed once per step."""
  tracker = getattr(env, _PHASE_ATTR, None)
  if tracker is None:
    tracker = TaskPhase(env)
    setattr(env, _PHASE_ATTR, tracker)
  tracker.refresh(env)
  return tracker


def reset_phase(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Event term: clear the episode state for envs that just restarted.

  Registered last among the reset events, so the ball and arm are already in their new
  positions when the apex baseline is seeded from them.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  tracker = getattr(env, _PHASE_ATTR, None)
  if tracker is None:
    tracker = TaskPhase(env)
    setattr(env, _PHASE_ATTR, tracker)
  tracker.reset(env, env_ids)


##
# Task thresholds.
##

# Ball speed below which a catch counts as having dissipated the ball's energy, and
# how many consecutive steps it must hold before the catch is called a success.
CATCH_SPEED_THRESHOLD = 0.15
CATCH_SETTLE_STEPS = 20

# How many consecutive steps the ball may be off the bat before `balance` counts it
# as dropped.
#
# Sized from measurement, and the exact value matters. A ball genuinely resting on the
# blade still shows brief contact chatter, but never more than 3 steps in a row. A
# policy that keeps the ball alive by *tapping* it -- little upward flicks that send it
# hopping -- leaves it airborne for about 14. Anything in between separates the two;
# 6 sits clear of the chatter with room to spare and ends a tapping episode almost at
# once. Set this generously (it was 15) and juggling becomes the best strategy in the
# environment, because each hop costs nothing and keeps the episode alive.
BALANCE_GRACE_STEPS = 6

# What counts as the ball actually *resting* on the bat, and how long it may fail to
# before `balance` gives up on it.
#
# Contact alone is not enough, and this is the difference between holding a ball and
# juggling one. A policy can keep a ball nominally "in contact" most of the time by
# flicking it upward repeatedly -- each hop is shorter than the grace above, so the
# episode never ends, and the contact reward keeps paying. What separates the two is
# the ball's speed: resting it sits at ~0.007 m/s, while even a gentle shimmy that
# keeps it nominally in contact runs it at ~0.08. Requiring it to be slow *and*
# touching is what makes juggling a losing strategy rather than merely a worse one.
#
# The threshold has to sit well below that shimmy rather than merely below a vigorous
# juggle, or the mildest tapping still qualifies as resting. The grace is sized from
# the spawn: the ball arrives with a little drift and damps below the threshold on its
# own within ~7 steps, so 25 leaves room to settle it without ever tolerating a juggle.
BALANCE_SETTLED_SPEED = 0.05
BALANCE_SETTLE_GRACE = 25


##
# Observations.
##


def ball_pos_rel_face(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball position as seen from the blade centre, shaped (num_envs, 3).

  Relative rather than absolute, so it reads identically in every env and does not
  encode where the arm happens to stand.
  """
  return ball_pos(env) - face_pos(env)


def ball_lin_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Ball velocity. Privileged in the sense that a real system would have to estimate
  it, but every skill here needs it: catching and hitting are both about meeting a
  moving ball, and neither is solvable from position alone at this control rate."""
  return ball_vel(env)


##
# Rewards.
##


def reach_ball(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
  """Dense, bounded approach: how close the blade centre is to the ball.

  A positive kernel rather than a difference of distances. A difference telescopes to
  `start_dist - end_dist` over the episode, so any rollout where the ball ends up far
  away scores net negative no matter what the arm did, which teaches the policy to end
  episodes early instead of trying. This term is always >= 0.
  """
  return torch.exp(-torch.square(dist_ball_face(env)) / std**2)


def ball_over_face(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
  """How close the ball is to the blade centre in the blade's own plane.

  Distance measured perpendicular to the face normal, so it asks "is the ball over the
  middle of the bat" rather than "is it close to the bat". That is what an *approach*
  wants -- get the bat underneath the ball -- which is why catch and hit use it.

  Note what it deliberately ignores: the gap along the normal. A ball hovering well
  above the blade but dead centre scores full marks here. That makes it useless on its
  own as a "keep the ball on the bat" reward, because a policy can max it out by
  tracking a falling ball it never touches. Anything that means "on the bat" must gate
  on contact instead -- see `ball_on_face`.
  """
  offset = ball_pos(env) - face_pos(env)
  normal = face_normal(env)
  along = torch.sum(offset * normal, dim=-1, keepdim=True) * normal
  lateral = torch.norm(offset - along, dim=-1)
  return torch.exp(-torch.square(lateral) / std**2)


def ball_on_face(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
  """Reward the ball for sitting on the blade, near its centre. Zero unless touching.

  The full 3D distance to the face site, gated on contact -- which together are the
  literal statement of `balance`: keep the ball on the racket, as close as possible to
  the middle of it. The gate is the important half. Without it the same kernel is
  maximised by a bat held under a falling ball it never catches, which is a strictly
  easier behaviour to find and pays almost as well.
  """
  p = phase(env)
  score = torch.exp(-torch.square(dist_ball_face(env)) / std**2)
  return torch.where(p.on_racket, score, torch.zeros_like(score))


def ball_at_command_on_racket(
  env: ManagerBasedRlEnv, std: float, command_name: str = COMMAND_NAME
) -> torch.Tensor:
  """`ball_at_command`, but only while the ball is actually on the bat.

  Same reasoning as `ball_on_face`: an ungated position reward is collectable by
  hovering near the target without ever making contact.
  """
  p = phase(env)
  score = ball_at_command(env, std, command_name)
  return torch.where(p.on_racket, score, torch.zeros_like(score))


def racket_speed_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared linear speed of the blade centre. Pair with a negative weight.

  The cost of waving the bat around. For a task whose whole content is holding still,
  arm motion is not a means to the goal but evidence of failing at it: a blade resting
  under a ball moves ~0.006 m/s, while one flicking the ball back up moves ~0.09.
  Penalising it directly is what makes stillness worth more than busyness, rather than
  leaving that to be inferred from the contact terms alone.
  """
  robot: Entity = env.scene["robot"]
  return torch.sum(
    torch.square(robot.data.site_lin_vel_w[:, _face_index(robot)]), dim=-1
  )


def ball_slow(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
  """Reward a slow ball. This is the energy dissipation `catch` is judged on."""
  return torch.exp(-torch.square(ball_speed(env)) / std**2)


def ball_slow_on_racket(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
  """Reward a slow ball, but only while it is actually on the racket.

  Ungated, a policy could collect this by simply never going near the ball while it
  hangs at the top of its arc. Gating on contact means the only way to earn it is to
  take the ball's energy out through the bat.
  """
  p = phase(env)
  return torch.where(
    p.on_racket, ball_slow(env, std), torch.zeros(env.num_envs, device=env.device)
  )


def ball_at_command(
  env: ManagerBasedRlEnv, std: float, command_name: str = COMMAND_NAME
) -> torch.Tensor:
  """Reward the ball for being where the command says it should end up."""
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  error = torch.sum(torch.square(ball_pos(env) - command), dim=-1)
  return torch.exp(-error / std**2)


def in_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 while the ball is touching the racket."""
  return phase(env).on_racket.float()


def apex_matches_command(
  env: ManagerBasedRlEnv, std: float, command_name: str = COMMAND_NAME
) -> torch.Tensor:
  """Reward a tossed ball for peaking at the commanded height.

  Paid only once the ball is past its apex, when `max_height` is final; before that it
  would reward a ball that is merely still rising.
  """
  p = phase(env)
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  error = torch.square(p.max_height - command[:, 2])
  return torch.where(
    p.falling, torch.exp(-error / std**2), torch.zeros(env.num_envs, device=env.device)
  )


def toss_is_vertical(
  env: ManagerBasedRlEnv, std: float, command_name: str = COMMAND_NAME
) -> torch.Tensor:
  """Penalise horizontal drift during a toss, so the ball goes straight up.

  Measured against the command's x/y, which for `toss` is the blade's own resting
  position: a toss that drifts sideways is one the arm cannot catch again.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  drift = torch.norm(ball_pos(env)[:, :2] - command[:, :2], dim=-1)
  return torch.exp(-torch.square(drift) / std**2)


def caught_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sparse: 1 on the step `catch` succeeds. The outcome the dense terms lead up to."""
  return caught(env).float()


def illegal_contact_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 once the ball has touched anything but the racket. Pair with a negative weight.

  Needed as a reward and not just a termination: without a cost, ending the episode by
  dropping the ball is a way to stop accumulating any negative term, which is exactly
  the degenerate strategy to avoid.
  """
  return phase(env).illegal.float()


def apex_at_command(
  env: ManagerBasedRlEnv, std: float, command_name: str = COMMAND_NAME
) -> torch.Tensor:
  """How near the ball's apex came to the commanded point.

  Paid once, on the step the apex is recorded. Gating on the apex rather than scoring
  position continuously is what makes this "put the ball *there*" instead of "spend as
  long as possible near there".
  """
  p = phase(env)
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  error = torch.sum(torch.square(p.apex_pos - command), dim=-1)
  return torch.where(
    p.apex_recorded,
    torch.exp(-error / std**2),
    torch.zeros(env.num_envs, device=env.device),
  )


def apex_is_still(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
  """How nearly the ball was at a standstill at its apex.

  A projectile's vertical speed is zero at its apex for free, so what this actually
  rewards is killing the *horizontal* speed: the ball has to be sent straight up, and
  arrive at the commanded point rather than sail through it.
  """
  p = phase(env)
  return torch.where(
    p.apex_recorded,
    torch.exp(-torch.square(p.apex_speed) / std**2),
    torch.zeros(env.num_envs, device=env.device),
  )


def apex_in_goal(
  env: ManagerBasedRlEnv,
  goal_radius: float,
  speed_threshold: float,
  command_name: str = COMMAND_NAME,
) -> torch.Tensor:
  """Sparse: 1 on the step the ball apexes inside the goal, nearly stationary.

  The success `hit` is actually after, and the reason the two graded terms above exist:
  on their own this fires far too rarely to learn from.
  """
  p = phase(env)
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  dist = torch.norm(p.apex_pos - command, dim=-1)
  return (
    p.apex_recorded & (dist < goal_radius) & (p.apex_speed < speed_threshold)
  ).float()


def caught(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Success for `catch`: the ball has been resting quietly on the blade long enough.

  Held for several consecutive steps rather than judged on a single one, so a ball
  passing through zero velocity at the top of a bounce does not read as a catch.
  """
  return phase(env).settled_steps >= CATCH_SETTLE_STEPS


def ball_not_resting(
  env: ManagerBasedRlEnv, grace_steps: int = BALANCE_SETTLE_GRACE
) -> torch.Tensor:
  """The ball has not been sitting still on the bat for `grace_steps` in a row.

  The termination that makes `balance` mean what it says. `ball_left_bat` below only
  notices a ball that is *gone*; a policy that keeps one alive by tapping it upward
  never triggers it, because each hop is shorter than that grace. This one asks for the
  ball to be touching *and* slow, which a tapped ball never is.
  """
  return phase(env).unsettled_steps >= grace_steps


def ball_left_bat(env: ManagerBasedRlEnv, grace_steps: int = 15) -> torch.Tensor:
  """The ball has been off the racket for `grace_steps` in a row.

  The direct statement of failure for `balance`, and a much tighter one than waiting
  for the ball to drift a set distance away: it ends the episode the moment the ball
  is genuinely gone, which closes the window in which a policy could be paid for
  shadowing a ball in free fall. The grace period is what keeps a small bounce -- a
  couple of centimetres, back on the bat within ~0.1 s -- from counting as a drop.
  """
  return phase(env).off_racket_steps >= grace_steps


def illegal_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The ball touched something that is not the racket.

  The constraint the whole task set is built around. `hit` overrides the ground part
  of it, since there the ball is meant to land.
  """
  return phase(env).illegal


def apex_reached(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminal for `hit`: the struck ball has peaked, so the outcome is settled.

  Ending here also keeps the ball off the floor, which is what lets `hit` obey the
  same "only ever touch the racket" rule as the other three instead of carving out an
  exception for a landing.
  """
  return phase(env).apex_recorded


def toss_returned(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminal for `toss`: a launched ball has fallen back to the blade's height.

  The toss is finished at that point; catching it again is another skill's job.
  """
  p = phase(env)
  return p.falling & (ball_pos(env)[:, 2] < face_pos(env)[:, 2])


def ball_out_of_reach(
  env: ManagerBasedRlEnv, max_distance: float = 4.0
) -> torch.Tensor:
  """The ball has left the working volume entirely."""
  return torch.norm(ball_pos(env), dim=-1) > max_distance


##
# Events.
##


def reset_ball_on_blade(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  offset: float = BLADE_REST_OFFSET,
  lateral_range: float = 0.0,
  speed_range: tuple[float, float] = (0.0, 0.0),
  ball_cfg: SceneEntityCfg = _BALL,
) -> None:
  """Place the ball on (or just above) the blade face.

  Cannot be a plain uniform root reset: where the blade is depends on the arm's pose,
  so the ball is positioned from the blade site's live pose. `lateral_range` scatters
  it across the face and `speed_range` gives it a small random drift, which is what
  keeps `balance` from being solvable by standing perfectly still.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  ball: Entity = env.scene[ball_cfg.name]

  # The arm reset that ran before this only wrote joint targets into the entity
  # buffers; site poses come out of forward kinematics and are still whatever they
  # were. Flush and re-run, or the ball is placed on where the blade used to be (on
  # the very first reset, on the model's zero pose, nowhere near the ready stance).
  env.scene.write_data_to_sim()
  env.sim.forward()

  n = len(env_ids)
  positions = (face_pos(env) + face_normal(env) * offset)[env_ids]
  if lateral_range > 0.0:
    scatter = (torch.rand(n, 3, device=env.device) * 2.0 - 1.0) * lateral_range
    scatter[:, 2] = 0.0  # Scatter across the face, not through it.
    positions = positions + scatter

  root_state = torch.zeros(n, 13, device=env.device)
  root_state[:, 0:3] = positions
  root_state[:, 3] = 1.0  # identity quaternion
  lo, hi = speed_range
  if hi > 0.0:
    direction = torch.randn(n, 3, device=env.device)
    direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-6)
    speed = lo + (hi - lo) * torch.rand(n, 1, device=env.device)
    root_state[:, 7:10] = direction * speed
  ball.write_root_state_to_sim(root_state, env_ids=env_ids)


def reset_ball_above_face(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  height_range: tuple[float, float] = (0.0, 0.0),
  lateral_range: float = 0.0,
  vel_range: dict[str, tuple[float, float]] | None = None,
  ball_cfg: SceneEntityCfg = _BALL,
) -> None:
  """Spawn the ball above the blade centre, in the blade's own x/y.

  Every task spawns this way, which is the point: an absolute box in world coordinates
  silently stops being reachable the moment the ready stance changes, and a ball the
  arm cannot get to is not a task, just a guaranteed failure. Anchoring to the live
  face pose keeps position coupled to the arm's configuration, so whatever stance the
  robot starts in, the ball starts somewhere it can actually be played.

  `height_range` is measured straight up from the resting point on the blade, so zero
  puts the ball on the bat (balance, toss) and a large value drops it in from above
  (catch, hit). `lateral_range` is the jitter around the blade's x/y -- deliberately
  small, so the ball stays in front of the racket rather than off to one side.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  ball: Entity = env.scene[ball_cfg.name]
  n = len(env_ids)

  # The arm reset that ran before this only wrote joint targets into the entity
  # buffers; site poses come out of forward kinematics and are still whatever they
  # were. Flush and re-run, or the ball is placed relative to where the blade used to
  # be (on the very first reset, the model's zero pose, nowhere near the stance).
  env.scene.write_data_to_sim()
  env.sim.forward()

  # Resting point on the blade, then straight up from there in world Z.
  positions = (face_pos(env) + face_normal(env) * BLADE_REST_OFFSET)[env_ids]
  if lateral_range > 0.0:
    jitter = (torch.rand(n, 2, device=env.device) * 2.0 - 1.0) * lateral_range
    positions = positions.clone()
    positions[:, :2] += jitter
  lo, hi = height_range
  if hi > 0.0:
    positions = positions.clone()
    positions[:, 2] += lo + (hi - lo) * torch.rand(n, device=env.device)

  root_state = torch.zeros(n, 13, device=env.device)
  root_state[:, 0:3] = positions
  root_state[:, 3] = 1.0  # identity quaternion
  if vel_range:
    bounds = torch.tensor(
      [vel_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z")], device=env.device
    )
    root_state[:, 7:10] = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * torch.rand(
      n, 3, device=env.device
    )
  ball.write_root_state_to_sim(root_state, env_ids=env_ids)


def reset_ball_uniform(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pos_range: dict[str, tuple[float, float]],
  vel_range: dict[str, tuple[float, float]],
  ball_cfg: SceneEntityCfg = _BALL,
) -> None:
  """Put the ball into flight from a sampled position with a sampled velocity.

  The ranges are absolute positions in the arm's frame rather than offsets from the
  asset's default root state, so each skill states plainly where its ball comes from.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  ball: Entity = env.scene[ball_cfg.name]
  n = len(env_ids)

  pos_bounds = torch.tensor(
    [pos_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z")], device=env.device
  )
  vel_bounds = torch.tensor(
    [vel_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z")], device=env.device
  )

  def _sample(bounds: torch.Tensor) -> torch.Tensor:
    lo, hi = bounds[:, 0], bounds[:, 1]
    return lo + (hi - lo) * torch.rand(n, 3, device=env.device)

  root_state = torch.zeros(n, 13, device=env.device)
  # No env_origins offset: see `ball_pos`.
  root_state[:, 0:3] = _sample(pos_bounds)
  root_state[:, 3] = 1.0  # identity quaternion
  root_state[:, 7:10] = _sample(vel_bounds)
  ball.write_root_state_to_sim(root_state, env_ids=env_ids)
