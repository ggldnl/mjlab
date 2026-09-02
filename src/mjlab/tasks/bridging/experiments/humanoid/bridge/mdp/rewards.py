"""What the bridge is paid for.

    arrival    8-channel kernel against the target, under a ramp peaking at the deadline
    approach   same channels through a wide kernel, flat across the window
    guidance   nearness to the recorded crossing. Shaping, anneals to zero

    action_acc     second difference of the action
    action_rate    first difference
    feet_slip      foot sliding while in contact
    feet_chatter   contact or flight phase too short to be a step
    joint_limits   joints against their soft limits

The first three carry the objective. The rest are regularizers and are all non-positive.

Why the objective needs two terms
---------------------------------

Nothing scores the middle of a window, so `arrival` alone is a reward that has to be
stumbled into: 15 to 60 steps of a 29-joint humanoid is a lot of ways to hit nothing.
`approach` is the only gradient over the first half.

`approach` is a function of state, not of progress. A term paying for ground covered would
have to charge for ground given up, and a body that turns around or gathers itself gives up
ground first every time.

Why every term is signed the way it is
--------------------------------------

The three objective terms are strictly positive, so ending an episode early is always worse
than continuing. A bridge here once found that falling over promptly beat trying, which is
what happens when a reward can go negative and the cheapest way to stop losing is to stop.
What stops the policy parking somewhere safe is the `strayed` termination, not the reward.

The regularizers are strictly non-positive. This task used to carry a positive air_time
reward borrowed from locomotion. It stopped the feet chattering and it also paid the bridge
to walk while it was being asked to crouch or hold a stance, so on every non-locomotion
target the objective and the gait prior pulled against each other and the arrival lost. A
penalty forbids the artifact without wanting anything, which is what a regularizer is for.

Why the gait terms exist at all
-------------------------------

"Be in this state by then, and stay upright" is indifferent to how. Left alone, PPO found a
high-frequency hop: it covers ground, it does not fall, nothing objected. No initialization
fixes that, because hopping is what the objective pays for.
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
  channel_errors,
)


def _command(env: ManagerBasedRlEnv, command_name: str) -> BridgeCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, BridgeCommand)
  return term


def arrival(
  env: ManagerBasedRlEnv, command_name: str, sharpness: float = 3.0
) -> torch.Tensor:
  """The 8-channel kernel against the target, under a ramp peaking at the deadline.

  `sharpness` is the exponent on progress. At 3, the last fifth of a window is worth about
  half of what this pays over the whole of it, so the reward concentrates where the question
  is without leaving the first two thirds with no signal.

  Paying early is a small distortion that buys something: arriving ahead of time and holding
  collects more than arriving on the buzzer, and holding is the better hand-over.

  Runs against the curriculum's tolerances, not the requirements. See `BridgeCommand._retune`
  for why: a channel more than about 3 tolerances out contributes exactly zero, and a channel
  that teaches nothing gets spent on the ones that do.
  """
  command = _command(env, command_name)
  errors = command.errors_now()
  ramp = command.progress.pow(sharpness)
  # The curriculum's tolerances, not the requirements. arrived and every reported number use
  # command.tolerances, so only what is being taught moves. One shared object would make the
  # score chase the error and hold constant by construction, leaving no instrument at all
  return ramp * arrival_score(errors, command.reward_tolerances)


def guidance(
  env: ManagerBasedRlEnv, command_name: str, tolerance_scale: float = 4.0
) -> torch.Tensor:
  """How near the robot is to the crossing the tracker recorded across this window.

  Every window is a contiguous slice of one rollout, so the frames between its endpoints are
  a motion this robot performed under this physics. `BridgeCommand.reference_now` reads the
  one for this tick.

  Shaping, not an objective. `guide_scale` anneals it to zero, after which the reward is
  arrival and approach alone. It has to go away: there is no reference at inference, and
  during training the start is perturbed off the recorded one on purpose, so the recorded
  crossing is often unreachable from where the robot actually is. A term that scored it
  would pay for imitating one answer instead of for arriving.

  Deliberately wide and light: `tolerance_scale` times the requirements, and `approach`'s
  bottleneck weight. A hint about a whole motion, not a requirement about a state.

  Measured against the fixed requirements, not the reward curriculum's tolerances. The two
  arrival terms chase terminal accuracy and should sharpen as the policy does; this one asks
  whether the robot is on the recorded motion at all, which has a fixed answer. Hanging it
  off the curriculum would open it at 10x the requirement, where a metre of root error and a
  centimetre score alike.

  Zero where there is no crossing, which is a window aimed from outside through
  `open_window`. Multiplied rather than branched, so the shapes match either way.
  """
  command = _command(env, command_name)
  scale = command.guide_scale
  if scale <= 0.0:
    return torch.zeros(env.num_envs, device=env.device)
  errors = channel_errors(command.state_now(), command.reference_now(), command.arms)
  score = arrival_score(
    errors, command.tolerances * tolerance_scale, bottleneck_weight=0.2
  )
  return scale * score * command.has_reference


def approach(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Broad full-state gradient, flat across the window. Arrival handles precision.

  4x the tolerances, with a light bottleneck instead of arrival's heavy one. This term has
  to say something useful about a state nowhere near the target, and a hard bottleneck would
  flatten it to zero over the whole first half of a window, which is the half with no other
  signal.

  All 8 channels, not just the root. A root-only version made every other channel wait for
  the last few ticks, and the policy had a rational incentive to use its arms for balance
  and never bring them back.
  """
  command = _command(env, command_name)
  errors = command.errors_now()
  return arrival_score(errors, command.reward_tolerances * 4.0, bottleneck_weight=0.2)


def feet_below_ground(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.0
) -> torch.Tensor:
  """How far the lowest foot is under the floor. Should read exactly zero, always.

  Physics makes this nearly impossible, which is most of the reason the middle of a window
  comes from a policy in a simulator instead of a model that regresses frames: the
  supervised attempt this replaced put a foot through the floor in 2 windows out of 5. Kept
  as a term because a number pinned at zero in the log is the evidence it stays fixed.
  """
  asset = env.scene[asset_cfg.name]
  assert asset_cfg.body_ids is not None
  heights = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  return (threshold - heights.min(dim=-1).values).clamp(min=0.0)


def feet_slip(
  env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Foot sliding along the floor: horizontal speed while in contact.

  Local, not reused from the velocity tasks. Theirs scales by how fast the robot was told to
  go, reading command channels 0-1 as a linear velocity and 2 as a yaw rate. The bridge's
  command holds a target offset in those channels, so the numbers would look plausible and
  mean nothing, and the coupling would stay invisible until somebody reordered the
  observation.

  Ungated, which is what a bridge wants: a planted foot has no horizontal speed, so standing
  still already costs nothing, and no moment in a window makes dragging a foot right.
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
  """The two legs counter-rotating so the knees face each other.

  Both hip yaw joints turn about the same axis, so legs turning together is a twist of the
  whole lower body and legs turning against each other is the knees converging. Right minus
  left separates them: zero for any twist however far the robot turned, growing only as the
  knees converge. A term built on each joint's own magnitude cannot tell the two apart and
  would tax every turning jump in the corpus to catch something none of them do.

  Threshold measured, not chosen. Over 84705 states of walk, run and jump:

      p99                                0.24
      p99.9                              0.43
      walk max                           0.40
      run max                            0.18
      bridge at 2600 iterations, median  0.47

  The bridge spends over half its time more knock-kneed than 99.9% of what the skills do, so
  a bound in that gap costs the objective almost nothing. Quadratic past it, exactly zero
  inside.

  Joints resolved by name, not in the order `asset_cfg` gave them. Which comes first decides
  the sign, and a term that silently rewards what it forbids is invisible in a reward curve.
  """
  asset: Entity = env.scene[asset_cfg.name]
  names = asset.joint_names
  converge = (
    asset.data.joint_pos[:, names.index(right_joint)]
    - asset.data.joint_pos[:, names.index(left_joint)]
  )
  return (converge - threshold).clamp(min=0.0).square()


def feet_chatter(
  env: ManagerBasedRlEnv, sensor_name: str, min_time: float = 0.2
) -> torch.Tensor:
  """A contact or flight phase that ends sooner than a real step's would.

  Charged at the transition, once per phase, and only for the part below `min_time`. Longer
  than that costs exactly zero, so a planted foot is free and a proper step is free. The only
  thing priced is a phase too short to be either.

  That one-sidedness is the design. The term this replaced paid for air time, which is an
  instruction to walk, and a bridge asked to arrive in a crouch was being paid to do
  something else at the same time.

  A zero previous phase means there was no previous phase, which is what a foot looks like on
  the step after a teleport. Charging it would fine every window for its own start.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  air = sensor.data.last_air_time
  stance = sensor.data.last_contact_time
  assert air is not None and stance is not None
  landed = sensor.compute_first_contact(env.step_dt) & (air > 0.0)
  lifted = sensor.compute_first_air(env.step_dt) & (stance > 0.0)
  short = _shortfall(air, min_time) * landed.float()
  short = short + _shortfall(stance, min_time) * lifted.float()
  return short.sum(dim=1)


def _shortfall(elapsed: torch.Tensor, floor: float) -> torch.Tensor:
  """How far short of `floor` each phase fell. Zero for every phase that reached it."""
  return (elapsed.clamp(max=floor) - floor).neg()
