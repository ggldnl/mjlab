"""
The goal is the twist. Walking already has one: the velocity task's commanded linear
and angular velocity, which the policy already sees and is already scored on. Pushing
does not need a different goal, it needs the same goal applied to the box, so the push
reward asks how much of the commanded velocity the box is actually carrying. That keeps
the conditioning real on both sides without inventing a command term whose relationship
to the locomotion one would have to be negotiated every step.

The hands are the only legal contact. Every other collision geom on the robot is illegal
and carries a penalty. Without that gate the task is locomotion with a box in front of it,
and the cheapest solution is to walk into the crate chest first.
`reach_box` is a distance kernel that reaches one exactly when a hand is against the box,
`hands_on_box` pays while the contact holds, and `push_tracking` pays for the box moving
the way it was asked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

# The velocity task's terms come in first and the push specific ones are layered on top,
# so a config can reach everything through this one namespace. This also brings in
# mjlab's generic terms, which the velocity task re-exports.
from mjlab.tasks.velocity.mdp import *  # noqa: F401, F403
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  sample_uniform,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


##
# Names and geometry.
##

ROBOT = "robot"
BOX = "box"

# The locomotion command, reused as the push goal. This is the name the velocity task
# gives it, and nothing here renames it.
COMMAND_NAME = "twist"

# Contact sensors, defined in push_env_cfg.py. The first watches the two hand spheres
# against the crate, the second watches the whole robot against it. Both are needed:
# the difference between their contact counts is what an illegal touch is, and there is
# no way to ask a subtree sensor to skip two of its geoms.
HANDS_BOX_SENSOR = "hands_box_contact"
ROBOT_BOX_SENSOR = "robot_box_contact"

# The dome collision geoms that stand in for hands. Left first, matching the order
# find_geoms returns and therefore the order the hand sensor's primaries are in, so a
# per-hand tensor from the sensor lines up with a per-hand tensor from the model.
HAND_GEOMS: tuple[str, ...] = (r"^(left|right)_hand_collision$",)

# Half-extents of the crate, in metres. A one metre cube: tall enough that a G1 meets it
# with its hands at chest height rather than stepping over it, wide enough that walking
# around it is a detour rather than a sidestep.
BOX_HALF_SIZE: tuple[float, float, float] = (0.5, 0.5, 0.5)

_BOX_CFG = SceneEntityCfg(BOX)


def hands_cfg() -> SceneEntityCfg:
  """A fresh entity config selecting the two hand collision geoms.

  A function rather than a module constant, and passed explicitly by every term that
  needs hand positions. Explicitly, because a SceneEntityCfg is only resolved when it
  appears in a term's params, so a default argument would arrive with geom_ids still
  slice(None) and quietly select every geom on the robot. Fresh, because resolving one
  writes the ids of a particular scene into it, and this config is built twice, once for
  training and once for play.
  """
  return SceneEntityCfg(ROBOT, geom_names=HAND_GEOMS)


##
# Scene readings.
##


def box_pos_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Box center in world coordinates, shaped (num_envs, 3)."""
  box: Entity = env.scene[BOX]
  return box.data.root_link_pos_w


def box_quat_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Box orientation, shaped (num_envs, 4).

  The full orientation, not the yaw. A cube shoved by a walking humanoid tips as often
  as it slides, and the surface distance below is only right if the box's own axes are
  where it thinks they are.
  """
  box: Entity = env.scene[BOX]
  return box.data.root_link_quat_w


def box_vel_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Box linear velocity in world axes, shaped (num_envs, 3)."""
  box: Entity = env.scene[BOX]
  return box.data.root_link_lin_vel_w


def root_pos_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Robot root position in world coordinates, shaped (num_envs, 3)."""
  robot: Entity = env.scene[ROBOT]
  return robot.data.root_link_pos_w


def hand_pos_w(env: ManagerBasedRlEnv, hands_cfg: SceneEntityCfg) -> torch.Tensor:
  """Centers of the two hand spheres in world coordinates, (num_envs, 2, 3).

  The geom center rather than the wrist body origin. The version of the robot I'm using
  doesn't have rubber hands, it has domes instead. The dome is seated 0.04 m out along
  the wrist axis, which is most of a hand radius, and the reach kernel below is measuring
  the last few centimetres before contact.
  """
  robot: Entity = env.scene[hands_cfg.name]
  ids = hands_cfg.geom_ids
  assert isinstance(ids, list) and len(ids) == 2, (
    "hands_cfg must be passed through a term's params so its geom_names resolve to the "
    f"two hand collision geoms; got {ids}."
  )
  return robot.data.geom_pos_w[:, ids]


def heading_quat(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The robot's yaw only orientation, shaped (num_envs, 4).

  Yaw only, never the full root orientation. A robot leaning its weight into a crate
  pitches hard, and a vector expressed in the full base frame swings around with it, so
  the box would appear to move when only the robot had leaned.
  """
  robot: Entity = env.scene[ROBOT]
  return yaw_quat(robot.data.root_link_quat_w)


def to_heading_frame(env: ManagerBasedRlEnv, vec_w: torch.Tensor) -> torch.Tensor:
  """Rotate a world frame vector, (num_envs, 3) or (num_envs, K, 3), into heading."""
  quat = heading_quat(env)
  if vec_w.dim() == 3:
    quat = quat.unsqueeze(1).expand(-1, vec_w.shape[1], -1)
  return quat_apply_inverse(quat, vec_w)


def box_surface_gap(
  env: ManagerBasedRlEnv,
  points_w: torch.Tensor,
  half_size: tuple[float, float, float] = BOX_HALF_SIZE,
) -> torch.Tensor:
  """Distance from each of K world points to the box's surface, (num_envs, K).

  The exterior distance to a box, computed in the box's own frame: take the offset,
  subtract the half-extents from its absolute value per axis, floor each at zero, and
  take the norm of what is left. Zero anywhere inside or on the surface.

  Center to center distance would not do. It bottoms out at half a metre for a crate this
  size, so a kernel built on it peaks at a value nothing can reach and spends its useful
  range on offsets that all mean the same thing. Subtracting the extents puts the peak
  where contact actually happens.
  """
  offset_w = points_w - box_pos_w(env).unsqueeze(1)
  quat = box_quat_w(env).unsqueeze(1).expand(-1, points_w.shape[1], -1)
  offset_b = quat_apply_inverse(quat, offset_w)
  extents = torch.tensor(half_size, device=env.device)
  outside = torch.clamp(torch.abs(offset_b) - extents, min=0.0)
  return torch.norm(outside, dim=-1)


def hand_box_gap(env: ManagerBasedRlEnv, hands_cfg: SceneEntityCfg) -> torch.Tensor:
  """Distance from each hand sphere to the crate's surface, (num_envs, 2).

  Measured to the sphere's center, so it floors at the hand radius rather than at zero
  when the hand is resting on the crate. That offset is 0.03 m against a kernel whose
  width is half a metre, which is nothing, and leaving it in keeps this a plain distance
  that can be read off a log without remembering a correction.
  """
  return box_surface_gap(env, hand_pos_w(env, hands_cfg))


def hands_found(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Number of crate contacts on each hand, (num_envs, 2)."""
  sensor: ContactSensor = env.scene[HANDS_BOX_SENSOR]
  found = sensor.data.found
  assert found is not None
  return found


def hands_on_box(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Whether each hand is touching the crate, (num_envs, 2) bool."""
  return hands_found(env) > 0


def body_on_box(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Whether anything that is not a hand is touching the crate, (num_envs,) bool.

  Counted rather than sensed directly. A contact sensor takes one primary pattern and a
  subtree primary is a single element, so there is no way to say "the whole robot except
  these two geoms" in one sensor, and spelling the other thirty collision geoms out as
  thirty primaries costs thirty sensors and their match buffers at four thousand
  environments. Both sensors report ``found`` as the number of matching contacts, over
  the same contact list, so the whole robot's count minus the two hands' counts is
  exactly the number of contacts made by something that is not a hand.
  """
  sensor: ContactSensor = env.scene[ROBOT_BOX_SENSOR]
  total = sensor.data.found
  assert total is not None
  return total.sum(dim=-1) > hands_found(env).sum(dim=-1)


def commanded_vel_xy(
  env: ManagerBasedRlEnv, command_name: str = COMMAND_NAME
) -> tuple[torch.Tensor, torch.Tensor]:
  """The commanded horizontal velocity and its magnitude, in the robot's frame.

  Returns the (num_envs, 2) command and the (num_envs,) speed, floored away from zero so
  callers can divide by it. The floor is a guard and nothing more: the command ranges in
  push_env_cfg.py are strictly forward, and no environment is asked to stand still.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  vel = command[:, :2]
  return vel, torch.clamp(torch.norm(vel, dim=-1), min=1.0e-3)


##
# Episode state.
##

# The env has no slot for this, so where the box started is attached under a namespaced
# attribute and reached with getattr and setattr, so it does not read as something the
# env class is expected to declare. Nothing here is latched or gated, so unlike the
# kick's phase tracker it needs no per-step refresh: the reset event writes it and the
# displacement metric reads it.
_HOME_ATTR = "_push_box_home"


def box_home_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Where the box was placed at the start of the episode, shaped (num_envs, 2)."""
  home = getattr(env, _HOME_ATTR, None)
  if home is None:
    home = box_pos_w(env)[:, :2].clone()
    setattr(env, _HOME_ATTR, home)
  return home


##
# Observations.
##


def box_pos_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Box center seen from the robot, in its heading frame, shaped (num_envs, 3).

  Relative, so it reads identically in every env and carries nothing about where in the
  world this particular robot happens to be standing.
  """
  return to_heading_frame(env, box_pos_w(env) - root_pos_w(env))


def box_vel_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Box velocity in the robot's heading frame, shaped (num_envs, 3)."""
  return to_heading_frame(env, box_vel_w(env))


def hand_box_offset_b(
  env: ManagerBasedRlEnv, hands_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Vector from each hand to the box center, heading frame, (num_envs, 6).

  Strictly speaking redundant: the policy is told where the box is and it knows its own
  joint angles, so where its hands are relative to the crate is derivable. It is given
  anyway because it is the exact quantity the reach kernel scores, and asking a network
  to rediscover its own forward kinematics before it can start earning the first rung is
  a slow way to begin.
  """
  offset_w = box_pos_w(env).unsqueeze(1) - hand_pos_w(env, hands_cfg)
  return to_heading_frame(env, offset_w).flatten(start_dim=1)


def hands_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Whether each hand is on the crate right now, shaped (num_envs, 2).

  A contact switch on each palm is something a real robot can have, so this is an
  ordinary observation rather than a privileged one. It is what separates "reach for the
  crate" from "lean on it and walk".
  """
  return hands_on_box(env).float()


def hands_contact_force(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Net crate-on-hand contact force per hand, log compressed, (num_envs, 6).

  Privileged. How hard the robot is leaning is most of what decides whether the crate is
  about to move, and it stands in for the box's mass, which the reset randomizes and the
  actor is never told. Compressed the way the velocity task compresses foot forces,
  because the raw number spans two orders of magnitude between a brush and a shove.
  """
  sensor: ContactSensor = env.scene[HANDS_BOX_SENSOR]
  force = sensor.data.force
  assert force is not None
  flat = force.flatten(start_dim=1)
  return torch.sign(flat) * torch.log1p(torch.abs(flat))


def body_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Whether an illegal part is on the crate, shaped (num_envs, 1).

  Privileged, because a G1 has no skin. The actor is penalized for this without being
  told it directly, which is fair: it knows where the crate is and where its own limbs
  are, so a torso against the crate is something it can predict rather than something it
  has to be informed of. The critic is given it because it is a step change in the
  return and guessing it from the rest of the state is wasted capacity.
  """
  return body_on_box(env).float().unsqueeze(-1)


##
# Rewards.
##


def reach_box(
  env: ManagerBasedRlEnv, std: float, hands_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Dense pull of the hands toward the box, shaped (num_envs,).

  A positive kernel on each hand's distance to the crate's surface, averaged over the
  two, never a difference of distances. A difference telescopes over the episode to start
  minus end, so any rollout that ends away from the box scores net negative however well
  it pushed.

  Averaging the two kernels rather than kernelling the mean gap is what makes it worth
  bringing the second hand up: with the mean gap inside the exponential, one hand planted
  and one hanging scores the same as two hands halfway there, and the policy has no
  gradient telling it which of those it should prefer.

  Measured at the hands rather than at the root, which is the whole difference between
  this and the walking task it is built on. A root kernel is maximized by standing as
  close to the crate as possible, which for a G1 means chest against cardboard; a hand
  kernel is maximized by reaching, and reaching is the skill.

  Ungated, unlike the kick's version. There the term had to stop paying on contact or the
  policy would learn to park a foot against the ball, which is the opposite of a kick.
  Here parking a palm against the box is the skill, and the kernel simply saturates at
  one for as long as the hand stays there. It also comes back on its own if the contact
  is lost, which is what makes it a chase rather than a one-off approach.
  """
  gap = hand_box_gap(env, hands_cfg)
  return torch.mean(torch.exp(-torch.square(gap) / std**2), dim=-1)


def hands_on_box_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Fraction of the hands that are on the box, shaped (num_envs,).

  The middle rung, and it is a fraction rather than a flag so that the second hand is
  worth as much as the first. Contact is the discrete step between a policy that can
  reach for a crate and one that can move it, and the distance kernel above cannot
  express it: the last few centimetres of gap and the first newton of contact look nearly
  identical to it.
  """
  return hands_on_box(env).float().mean(dim=-1)


def body_contact_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """One while anything but a hand is on the crate, shaped (num_envs,).

  Paired with a negative weight, and the term the whole task turns on. A G1 walking into
  a metre of cube meets it with chest, hips and shins long before it meets it with its
  palms, and shoving with the torso is both easier to discover and easier to balance
  through than shoving with two 3 cm spheres at the end of two arms.

  A flag rather than a count of offending parts. Counting would make a policy that has
  already fallen against the crate pay a penalty that grows with how much of it is
  touching, which is a large negative spike arriving at the moment the episode is already
  lost, and the parkour skills' notes on reward saturation are about exactly that. The
  flag is a flat cost for a state the policy should never be in.
  """
  return body_on_box(env).float()


def push_tracking(
  env: ManagerBasedRlEnv, command_name: str = COMMAND_NAME
) -> torch.Tensor:
  """How much of the commanded velocity the box is carrying, shaped (num_envs,).

  The fraction of the commanded speed the box is moving at along the commanded direction,
  clamped to [0, 1], and paid only while a hand is on it. This is the task: the robot is
  told to travel at a velocity, and the crate in front of it has to travel at that
  velocity too, driven from the palms.

  Clamped above at one, so there is nothing to gain from shoving the box away and letting
  it slide off; clamped below at zero, so a box drifting backwards is worth the same as a
  box standing still rather than being a penalty the policy can run up.

  The contact gate is what stops the ungated version from paying for a crate that is
  still coasting from a body check the policy has already been penalized for. It costs a
  little smoothness -- a contact that flickers pays only for the steps it holds -- and
  that is the right trade, because a push the hands are not part of is not this skill.

  The command lives in the robot's full base frame and the box's velocity is taken in its
  heading frame, which are not the same frame when the robot is leaning. The difference is
  a cosine of the lean, a few percent at the pitch a push involves, and the heading frame
  is the right one for something sliding on the ground: a crate does not tilt because the
  robot pushing it did.
  """
  vel, speed = commanded_vel_xy(env, command_name)
  along = torch.sum(box_vel_b(env)[:, :2] * vel / speed.unsqueeze(-1), dim=-1)
  fraction = torch.clamp(along / speed, 0.0, 1.0)
  return fraction * hands_on_box(env).any(dim=-1).float()


##
# Metrics.
##


def hands_contact_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Fraction of hands on the box, averaged over the episode, shaped (num_envs,).

  The first thing to read. Everything below it is meaningless until it is high: a policy
  that never reaches the crate is being scored on a box nothing is touching. One half is
  a one handed push and is a partial success; near zero with the velocity tracking reward
  climbing is a policy that has learned to walk around the crate.
  """
  return hands_on_box(env).float().mean(dim=-1)


def body_contact_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Fraction of the episode spent touching the box with something else, (num_envs,).

  The second thing to read, and the one that says whether the hands-only constraint took.
  It should fall toward zero as training goes on. If it stays high while the displacement
  metric also climbs, the policy is paying the penalty and body checking the crate
  anyway, and the answer is a larger body_contact weight rather than a smaller push one.
  """
  return body_on_box(env).float()


def box_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Horizontal speed of the box, shaped (num_envs,)."""
  return torch.norm(box_vel_w(env)[:, :2], dim=-1)


def box_displacement(env: ManagerBasedRlEnv) -> torch.Tensor:
  """How far the box has travelled from where it spawned, shaped (num_envs,).

  Logged with ``reduce="max"``, so it reads as the furthest the crate got this episode
  rather than an average over a trajectory that started at zero. This is the headline
  number: metres of crate moved.
  """
  return torch.norm(box_pos_w(env)[:, :2] - box_home_w(env), dim=-1)


##
# Events.
##


def reset_box_in_front(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  forward_range: tuple[float, float],
  lateral_range: tuple[float, float],
  yaw_range: tuple[float, float],
  half_size: tuple[float, float, float] = BOX_HALF_SIZE,
  box_cfg: SceneEntityCfg = _BOX_CFG,
) -> None:
  """Place the box ahead of the robot, in the robot's own heading frame.

  Relative to the robot, never at an absolute point. The robot resets with a random yaw
  and a scattered position, so a fixed world pose would put the crate behind it as often
  as in front, and a crate the robot never meets is not a task but a guaranteed miss.

  The forward offset is measured center to center, so the near face sits one half-extent
  closer than the number says: at 1.4 m the robot starts roughly 0.9 m from the face, and
  its hands, held out in front by the push keyframe, start about 0.5 m from it.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

  # The robot reset that ran before this wrote root pose and joint state into the entity
  # buffers, but nothing has pushed them through the simulator yet. Flush and re-run, or
  # the crate is placed relative to where the robot used to be, which on the very first
  # reset is the model's zero pose.
  env.scene.write_data_to_sim()
  env.sim.forward()

  robot: Entity = env.scene[ROBOT]
  box: Entity = env.scene[box_cfg.name]
  n = len(env_ids)

  root_w = robot.data.root_link_pos_w[env_ids]
  heading = yaw_quat(robot.data.root_link_quat_w[env_ids])

  offset_b = torch.zeros(n, 3, device=env.device)
  offset_b[:, 0] = sample_uniform(*forward_range, n, env.device)
  offset_b[:, 1] = sample_uniform(*lateral_range, n, env.device)
  offset_w = quat_apply(heading, offset_b)

  # A yaw of its own, so the policy meets a face at an angle as often as square on and
  # cannot learn one approach. Small, because a crate turned far enough presents a corner
  # and the task stops being a push.
  yaw = sample_uniform(*yaw_range, n, env.device)
  half = 0.5 * yaw

  root_state = torch.zeros(n, 13, device=env.device)
  root_state[:, 0:3] = root_w + offset_w
  root_state[:, 2] = half_size[2]  # Resting on the ground, whatever the robot is doing.
  root_state[:, 3] = torch.cos(half)
  root_state[:, 6] = torch.sin(half)
  box.write_root_state_to_sim(root_state, env_ids=env_ids)

  # Where it went, for the displacement metric. Written after the box has been placed and
  # read straight off it, so it needs no second forward pass.
  box_home_w(env)[env_ids] = root_state[:, :2]


##
# Curriculum helper.
##


def ramp(final_weight: float, stage_steps: int) -> list[dict]:
  """Stage list for the velocity task's reward_weight curriculum: off, half, full.

  The half step in the middle is not decoration: turning a large weight on in one go
  hands the policy a reward it has no idea how to earn, and the cheapest way to discover
  it is to throw itself at the crate, which undoes the walking it just learned.
  """
  return [
    {"step": 0, "weight": 0.0},
    {"step": stage_steps, "weight": 0.5 * final_weight},
    {"step": 2 * stage_steps, "weight": final_weight},
  ]
