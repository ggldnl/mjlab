"""What the bridge is paid for: being in the target state when the clock runs out.

Nothing scores the middle. That is the task, and it is also the difficulty: a reward paid
only at the deadline has to be stumbled into first, and 30 to 60 steps of a 29-joint
humanoid is a lot of ways to stumble into nothing.

Two terms carry the objective:

    arrival    the six-channel kernel, paid every step but weighted by a ramp that is near
               zero for most of the window and one at the deadline. Paying early too costs
               a small distortion and buys something: a policy that arrives ahead of time
               and holds the state collects more than one that arrives on the buzzer, and
               holding is a better hand-over than passing through.
    approach   distance to the target through a wide kernel, flat across the episode. The
               only gradient available for the first half of a window. A function of state,
               not of progress: a per-step term paying for ground covered would have to
               charge for ground given up, and a body that turns around or gathers itself
               gives up ground first every time.

The gait terms (action_acc, air_time, feet_slip, knees_inward) say nothing about arriving.
They exist because "be in this state by then, and stay upright" is indifferent to how, and
what PPO found was a high-frequency hop. Nothing in an initialization fixes that, because
hopping is what the objective pays for. These make it cost something.

Every arrival term is positive. A bridge here once discovered that falling over promptly
beat trying, which is what happens when a reward can go negative and the cheapest way to
stop losing is to stop. With everything positive, ending early is always worse than
continuing. What stops the policy parking somewhere safe and mediocre is the `strayed`
termination, not the reward.
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
  half of what the term pays over the whole of it, so the reward concentrates where the
  question is without leaving the first two thirds of the episode with no signal.

  The kernel runs against the curriculum's tolerances, not the calibrated ones. See
  `BridgeCommand._tighten`: a channel more than about three tolerances out contributes
  exactly zero, and a channel that teaches nothing gets spent on the ones that do.
  """
  command = _command(env, command_name)
  errors = command.errors_now()
  ramp = command.progress.pow(sharpness)
  # The curriculum's tolerances, not the fixed ones. arrived and every reported number use
  # command.tolerances, so only what is being taught moves. One shared object would make the
  # score chase the error and hold constant by construction, leaving no instrument at all
  return ramp * arrival_score(errors, command.reward_tolerances)


def approach(
  env: ManagerBasedRlEnv, command_name: str, pos_std: float = 1.5, ori_std: float = 1.0
) -> torch.Tensor:
  """How much of the gap to the target is left, through two deliberately wide kernels.

  Wide because this is the only thing to follow while the target is still metres away. A
  kernel already saturated at the distance a window opens at leaves the policy without a
  gradient exactly where it needs one most. Being narrow enough to matter is `arrival`'s
  job.
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
  comes from a policy in a simulator rather than from a model that regresses frames. The
  supervised attempt this replaced put a foot through the floor in two windows out of five.
  Kept as a term because a number pinned at zero in the log is the evidence it is fixed.
  """
  asset = env.scene[asset_cfg.name]
  assert asset_cfg.body_ids is not None
  heights = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  return (threshold - heights.min(dim=-1).values).clamp(min=0.0)


def feet_slip(
  env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Penalize a foot sliding along the floor: its horizontal speed while in contact.

  The velocity tasks have a version of this, not reused for one reason: theirs takes a
  command name and scales by how fast the robot was told to go, reading channels 0 and 1 of
  that command as a linear velocity and channel 2 as a yaw rate. The bridge's command holds
  an offset to the target in those channels. The numbers would look plausible and mean
  nothing, and the coupling would stay invisible until somebody reordered the observation.

  Ungated here, which is what a bridge wants anyway: a planted foot has no horizontal
  speed, so standing still already costs nothing, and no moment in a window makes dragging
  a foot the right answer.
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

  Both hip yaw joints turn about the same axis. The legs turning together is a twist of the
  whole lower body; the legs turning against each other is the knees converging. The
  difference right minus left separates the two: zero for any twist, however far the robot
  has turned, and growing only as the knees come to face one another. A term built on each
  joint's own magnitude cannot tell them apart and would tax every turning jump in the
  dataset to catch something none of them do.

  The threshold is measured, not chosen. Over 84705 states of walk, run and jump:

      99th percentile     0.24
      99.9th percentile   0.43
      walk maximum        0.40
      run maximum         0.18
      bridge at 2600 iterations, median   0.47

  So the bridge spends over half its time more knock-kneed than 99.9% of what the skills
  do, and a bound in that gap costs the objective almost nothing: the only frames it
  touches are the top fraction of a percent of the jump, and no target is drawn from there.

  Quadratic past the bound, exactly zero inside it. Nothing is asked of the legs while they
  do what the skills do.

  Joints are resolved by name, not in the order `asset_cfg` resolved them. Which one comes
  first decides the sign, and a term that silently rewards what it was written to forbid is
  not visible in a reward curve.
  """
  asset: Entity = env.scene[asset_cfg.name]
  names = asset.joint_names
  converge = (
    asset.data.joint_pos[:, names.index(right_joint)]
    - asset.data.joint_pos[:, names.index(left_joint)]
  )
  return (converge - threshold).clamp(min=0.0).square()
