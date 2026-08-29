"""What the bridge is paid for: being in the target state when the clock runs out.

There is no reference in the middle and nothing scores the middle. That is the point of
the task and it is also its difficulty: a reward that only pays at the deadline is a
reward a policy has to stumble into before it can improve on it, and thirty to sixty steps
of a 29-joint humanoid is a lot of ways to not stumble into anything.

Two terms carry it, and they carry different parts.

`arrival` is the objective. It is the six-channel kernel, paid every step but weighted by
a ramp that is near zero for most of the window and one at the deadline, so almost all of
its mass sits where the question is actually asked. Paying it early as well as late is
worth the small distortion it introduces: a policy that reaches the target ahead of time
and holds it collects more than one that arrives exactly on the buzzer, and holding a
state is a better hand-over than passing through it.

`approach` is the shaping, and it is the only gradient available for the first half of a
window. Distance to a fixed target through a wide kernel, flat across the episode. It is a
function of the state and not of progress, which matters: a per-step term paying for
ground covered would have to charge for ground given up, and a body that has to turn
around or gather itself before it can go anywhere gives up ground first every time. A
function of where you are has no such history to exploit and forbids no such motion.

The gait terms that were added after the first training runs -- `action_acc`, `air_time`
and `feet_slip` -- are not about arriving at all. They are there because a reward that only
says "be in this state by then, and stay upright" is indifferent to how the robot gets
there, and what PPO found was a high-frequency hop: it moves the body, it does not fall
over, and no term objected. Nothing in an initialization fixes that, because hopping is
what the objective was paying for. These make it cost something.

Every arrival term here is positive. That is not an accident either: this project has already
watched a bridge discover that falling over promptly beat trying, which is what happens
when a reward can go negative and the cheapest way to stop losing is to stop. With every
term positive, ending an episode early is always worse than continuing it, and what stops
a policy from parking somewhere safe and mediocre instead is the strayed termination, not
the reward.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.commands import (
  BridgeCommand,
  arrival_score,
)
from mjlab.utils.lab_api.math import quat_error_magnitude


def _command(env: ManagerBasedRlEnv, command_name: str) -> BridgeCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, BridgeCommand)
  return term


def arrival(
  env: ManagerBasedRlEnv, command_name: str, sharpness: float = 3.0
) -> torch.Tensor:
  """The six-channel arrival kernel under a ramp that peaks at the deadline.

  `sharpness` is the exponent on progress. At 3 the last fifth of a window is worth about
  half of everything the term pays over the whole of it, which concentrates the reward
  where the question is without leaving the first two thirds of the episode carrying no
  signal from it at all.
  """
  command = _command(env, command_name)
  errors = command.errors_now()
  ramp = command.progress.pow(sharpness)
  return ramp * arrival_score(errors, command.tolerances)


def approach(
  env: ManagerBasedRlEnv, command_name: str, pos_std: float = 1.5, ori_std: float = 1.0
) -> torch.Tensor:
  """How much of the gap to the target is left, through two deliberately wide kernels.

  Wide because this is the only thing to follow while the target is still metres away, and
  a kernel that has already saturated at the distance a window opens at leaves the policy
  with no gradient for the part of the episode where it needs one most. Narrow enough to
  matter is `arrival`'s job.
  """
  command = _command(env, command_name)
  data = command.robot.data
  distance = (data.root_link_pos_w - command.target[:, 0:3]).square().sum(dim=-1)
  turn = quat_error_magnitude(data.root_link_quat_w, command.target[:, 3:7]).square()
  return 0.5 * (
    torch.exp(-distance / (pos_std * pos_std)) + torch.exp(-turn / (ori_std * ori_std))
  )


def feet_below_ground(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.0
) -> torch.Tensor:
  """How far the lowest foot is under the floor, as a cost.

  Physics makes this nearly impossible, which is most of the reason the middle of a window
  is produced by a policy in a simulator rather than by a model that regresses frames. The
  supervised attempt this architecture replaced put a foot through the floor in two windows
  out of five. Kept as a term because a number pinned at zero in the training log is how we
  know the move worked.
  """
  asset = env.scene[asset_cfg.name]
  assert asset_cfg.body_ids is not None
  heights = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  return (threshold - heights.min(dim=-1).values).clamp(min=0.0)


def feet_slip(
  env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Penalize a foot sliding along the floor: its horizontal speed while in contact.

  The velocity tasks have a version of this already and it is not reused, for one reason:
  theirs takes a command name and scales itself by how fast the robot was told to go,
  reading channels 0 and 1 of that command as a linear velocity and channel 2 as a yaw
  rate. The bridge's command has an offset to the target in those channels. The numbers
  would come out plausible and mean nothing, and the coupling would be invisible until
  somebody reordered the observation.

  Ungated here, which is what a bridge wants anyway: a planted foot has no horizontal
  speed, so standing still already costs nothing, and there is no moment in a window when
  dragging a foot is the right answer.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  assert asset_cfg.site_ids is not None
  in_contact = (sensor.data.found > 0).float()
  speed = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2].square().sum(dim=-1)
  return (speed * in_contact).sum(dim=1)


def knees_inward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  left_joint: str,
  right_joint: str,
  threshold: float,
) -> torch.Tensor:
  """Penalize the two legs counter-rotating so the knees face each other.

  Both hip yaw joints turn about the same axis, so the two legs turning *together* is a
  twist of the whole lower body and the two turning *against* each other is the knees
  converging. The difference `right - left` separates them: it is zero for any twist,
  however far the robot has turned, and grows only as the knees come to face one another.
  A term built on each joint's own magnitude could not tell the two apart, and would tax
  every turning jump in the bank to catch a thing none of them do.

  The threshold is measured, not chosen. Across 84705 states of walk, run and jump this
  quantity has a 99th percentile of 0.24 and a 99.9th of 0.43; walk never exceeds 0.40 and
  run never exceeds 0.18. A bridge trained 2600 iterations sits at a *median* of 0.47,
  which is to say it spends over half its time more knock-kneed than 99.9% of everything
  the skills do. So a bound in that gap costs the objective essentially nothing -- the
  frames it touches are the top fraction of a percent of the jump, and no target is ever
  drawn from anywhere else -- while the behaviour it is aimed at is on the wrong side of it
  most of the time.

  Quadratic past the bound and exactly zero inside it. Nothing is asked of the legs while
  they are doing what the skills do.

  Joints are resolved by name here rather than taken in the order `asset_cfg` resolved
  them. Which of the two comes first decides the sign, and a term that silently rewards
  the thing it was written to forbid is not a failure anybody would catch by reading a
  reward curve.
  """
  asset: Entity = env.scene[asset_cfg.name]
  names = asset.joint_names
  converge = (
    asset.data.joint_pos[:, names.index(right_joint)]
    - asset.data.joint_pos[:, names.index(left_joint)]
  )
  return (converge - threshold).clamp(min=0.0).square()
