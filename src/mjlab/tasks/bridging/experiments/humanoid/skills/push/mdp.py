"""The pushing skill: what the robot sees, what it is paid for, and where the box goes.

The task in one sentence: a G1 walks into a one metre crate sitting in its path and
drives it along at the velocity it was commanded.

Three ideas run through everything below.

The goal is the twist, and there is no second command. Walking already has one: the
velocity task's commanded linear and angular velocity, which the policy already sees and
is already scored on. Pushing does not need a different goal, it needs the same goal
applied to the box, so the push reward asks how much of the commanded velocity the box
is actually carrying. That keeps the conditioning real on both sides without inventing a
command term whose relationship to the locomotion one would have to be negotiated every
step.

The ladder is continuous rather than latched. The kick is an instant, so its rungs latch:
touch the ball once and the floor rises for good. A push is not an instant, it is a
sustained contact that can be lost and has to be regained, so every rung here is live.
`approach_box` is a distance kernel that reaches one exactly when the robot is against the
box, `box_contact` pays while the contact holds, and `push_tracking` pays for the box
moving the way it was asked. Nothing about parking against the box has to be discouraged,
because parking against the box is the skill.

`push_tracking` is a projection, not a kernel. Every other goal term in this repo is an
exponential of a squared error, and a stack of them is how a policy discovers that dying
early beats trying (see the parkour skills' notes on reward saturation). This one is the
fraction of the commanded speed the box is carrying in the commanded direction, clamped
to [0, 1]: zero for a box that is not moving, one for a box keeping up, and nothing extra
for overshooting. A stationary box scores zero rather than the near-miss credit an
exponential would hand it at low commanded speeds.
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

# Contact sensor, defined in push_env_cfg.py.
ROBOT_BOX_SENSOR = "robot_box_contact"

# Half-extents of the crate, in metres. A one metre cube: tall enough that a G1 meets it
# with its arms and chest rather than stepping over it, wide enough that walking around
# it is a detour rather than a sidestep.
BOX_HALF_SIZE: tuple[float, float, float] = (0.5, 0.5, 0.5)

_ROBOT_CFG = SceneEntityCfg(ROBOT)
_BOX_CFG = SceneEntityCfg(BOX)


##
# Scene readings.
##


def box_pos_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Box centre in world coordinates, shaped (num_envs, 3)."""
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


def heading_quat(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The robot's yaw only orientation, shaped (num_envs, 4).

  Yaw only, never the full root orientation. A robot leaning its weight into a crate
  pitches hard, and a vector expressed in the full base frame swings around with it, so
  the box would appear to move when only the robot had leaned.
  """
  robot: Entity = env.scene[ROBOT]
  return yaw_quat(robot.data.root_link_quat_w)


def to_heading_frame(env: ManagerBasedRlEnv, vec_w: torch.Tensor) -> torch.Tensor:
  """Rotate a world frame vector into the robot's heading frame."""
  return quat_apply_inverse(heading_quat(env), vec_w)


def box_surface_gap(
  env: ManagerBasedRlEnv, half_size: tuple[float, float, float] = BOX_HALF_SIZE
) -> torch.Tensor:
  """Distance from the robot's root to the box's surface, shaped (num_envs,).

  The exterior distance to a box, computed in the box's own frame: take the offset,
  subtract the half-extents from its absolute value per axis, floor each at zero, and
  take the norm of what is left. Zero anywhere inside or on the surface.

  Centre to centre distance would not do. It bottoms out at half a metre for a crate this
  size, so a kernel built on it peaks at a value the robot can never reach and spends its
  useful range on offsets that all mean the same thing. Subtracting the extents puts the
  peak where contact actually happens.
  """
  offset_w = root_pos_w(env) - box_pos_w(env)
  offset_b = quat_apply_inverse(box_quat_w(env), offset_w)
  extents = torch.tensor(half_size, device=env.device)
  outside = torch.clamp(torch.abs(offset_b) - extents, min=0.0)
  return torch.norm(outside, dim=-1)


def touching_box(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Whether any part of the robot is in contact with the box, (num_envs,) bool."""
  sensor: ContactSensor = env.scene[ROBOT_BOX_SENSOR]
  found = sensor.data.found
  assert found is not None
  return (found > 0).any(dim=-1)


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
  """Box centre seen from the robot, in its heading frame, shaped (num_envs, 3).

  Relative, so it reads identically in every env and carries nothing about where in the
  world this particular robot happens to be standing.
  """
  return to_heading_frame(env, box_pos_w(env) - root_pos_w(env))


def box_vel_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Box velocity in the robot's heading frame, shaped (num_envs, 3)."""
  return to_heading_frame(env, box_vel_w(env))


def box_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Whether the robot is touching the box right now, shaped (num_envs, 1).

  A contact switch is something a real robot can have, so this is an ordinary
  observation rather than a privileged one. It is what separates "go and find the crate"
  from "lean on it and walk".
  """
  return touching_box(env).float().unsqueeze(-1)


def box_contact_force(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Net robot-on-box contact force, log compressed, shaped (num_envs, 3).

  Privileged. How hard the robot is leaning is most of what decides whether the crate is
  about to move, and it stands in for the box's mass, which the reset randomizes and the
  actor is never told. Compressed the way the velocity task compresses foot forces,
  because the raw number spans two orders of magnitude between a brush and a shove.
  """
  sensor: ContactSensor = env.scene[ROBOT_BOX_SENSOR]
  force = sensor.data.force
  assert force is not None
  flat = force.flatten(start_dim=1)
  return torch.sign(flat) * torch.log1p(torch.abs(flat))


##
# Rewards.
##


def approach_box(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
  """Dense pull of the robot toward the box, shaped (num_envs,).

  A positive kernel on the surface distance, never a difference of distances. A
  difference telescopes over the episode to start minus end, so any rollout that ends
  away from the box scores net negative however well it pushed.

  Ungated, unlike the kick's version. There the term had to stop paying on contact or the
  policy would learn to park a foot against the ball, which is the opposite of a kick.
  Here parking against the box is the skill, and the kernel simply saturates at one for as
  long as the robot stays there. It also comes back on its own if the contact is lost,
  which is what makes it a chase rather than a one-off approach.
  """
  return torch.exp(-torch.square(box_surface_gap(env)) / std**2)


def box_contact_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Paid for every step the robot is against the box, shaped (num_envs,).

  The middle rung. Contact is the discrete step between a policy that can walk up to a
  crate and one that can move it, and the distance kernel above cannot express it: the
  last few centimetres of gap and the first newton of contact look nearly identical to it.
  """
  return touching_box(env).float()


def push_tracking(
  env: ManagerBasedRlEnv, command_name: str = COMMAND_NAME
) -> torch.Tensor:
  """How much of the commanded velocity the box is carrying, shaped (num_envs,).

  The fraction of the commanded speed the box is moving at along the commanded direction,
  clamped to [0, 1]. This is the task: the robot is told to travel at a velocity, and the
  crate in front of it has to travel at that velocity too.

  Clamped above at one, so there is nothing to gain from shoving the box away and letting
  it slide off; clamped below at zero, so a box drifting backwards is worth the same as a
  box standing still rather than being a penalty the policy can run up.

  The command lives in the robot's full base frame and the box's velocity is taken in its
  heading frame, which are not the same frame when the robot is leaning. The difference is
  a cosine of the lean, a few percent at the pitch a push involves, and the heading frame
  is the right one for something sliding on the ground: a crate does not tilt because the
  robot pushing it did.
  """
  vel, speed = commanded_vel_xy(env, command_name)
  along = torch.sum(box_vel_b(env)[:, :2] * vel / speed.unsqueeze(-1), dim=-1)
  return torch.clamp(along / speed, 0.0, 1.0)


##
# Metrics.
##


def contact_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Fraction of the episode spent touching the box, shaped (num_envs,).

  The first thing to read. Everything below it is meaningless until this is high: a
  policy that never reaches the crate is being scored on a box nothing is touching.
  """
  return touching_box(env).float()


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

  The forward offset is measured centre to centre, so the near face sits one half-extent
  closer than the number says: at 1.4 m the robot starts roughly 0.9 m from the face,
  which is a step or two of walking rather than an immediate collision.
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
