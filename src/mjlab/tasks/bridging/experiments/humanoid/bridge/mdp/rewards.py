"""What the bridge is paid for.

Objective, all strictly positive:

    arrival    how much the 8 channel kernel against the target beat its own best
    guidance   nearness to the recorded crossing. Shaping, anneals to zero

Regularizers, all non positive:

    action_acc     second difference of the action
    action_rate    first difference
    feet_slip      foot sliding while in contact
    feet_chatter   contact or flight phase too short to be a step
    joint_limits   joints against their soft limits

arrival is the whole objective and it is dense, which is what it was not before. It used
to be a kernel under a progress cubed ramp, which put most of its mass in the last fifth of
a window and left the rest of the episode to a broad `approach` term evaluated at four
times the requirement. That term then ate the run: at convergence it was collecting 0.238
against arrival's 0.114, root linear velocity error had not moved in twelve thousand
iterations, and `arrived` read 0.000 all run. The policy had correctly learned the thing it
was being paid for, which was to hover in the neighbourhood and not fall over.

Paying the improvement instead makes every step that gets closer worth something and needs
no second term to fill the middle of the window, so `approach` is gone rather than
reweighted. The broad early gradient it was there for now comes from the tolerance
curriculum, which samples broad precision profiles and tightens their range. See
BridgeCommandCfg.tolerance_initial_range and tolerance_final_range.

Both are positive, so ending an episode early is always worse than continuing. An earlier
version found that falling over promptly beat trying, which is what happens when reward can
go negative and the cheapest way to stop losing is to stop. The strayed termination, not the
reward, is what stops the policy parking somewhere safe.
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


def arrival(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """How much the arrival score beat the best already reached this window.

  Summed over an episode this is the best arrival score the crossing ever achieved, paid
  once, whenever it happened. Three things follow, and all three are the point.

  Dense. Any step that gets closer than the policy has ever been pays immediately, so
  there is a gradient from the first step of a window rather than only near its end.

  No instant to hit. A target carries momentum, so it is a state the robot passes through
  and not one it can sit in. Scored at a fixed tick, a crossing that went through the
  target perfectly three ticks early reads as a miss, which is what a deadline was doing
  to every dynamic target in the corpus.

  Nothing paid for coming back. A plain per step score would pay a policy aiming at a
  moving target to turn round and re-approach it, over and over, which is the opposite of
  a hand-over. Once the best is set, leaving costs nothing and returning earns nothing.

  BridgeCommand.advance is what computes it, because it also moves the best of the window,
  writes the metrics. This is the only reward term that calls
  it, and calling it twice in one step would pay the same improvement twice.
  """
  increment = _command(env, command_name).advance()
  # Arrival is an event amount, while the manager integrates reward rates
  return increment / env.step_dt if env.cfg.scale_rewards_by_dt else increment


def guidance(
  env: ManagerBasedRlEnv, command_name: str, tolerance_scale: float = 4.0
) -> torch.Tensor:
  """How near the robot is to the crossing the tracker recorded across this window.

  Every window is a contiguous slice of one rollout, so the frames between its endpoints
  are a motion this robot performed under this physics. BridgeCommand.reference_now reads
  the one for this tick.

  Shaping, not an objective. guide_scale anneals it to zero, after which the reward is
  arrival alone. It has to go away: there is no reference at inference, and
  during training the start is perturbed off the recorded one on purpose, so the recorded
  crossing is often unreachable from where the robot actually is. A term that scored it
  would pay for imitating one answer instead of for arriving.

  Deliberately wide and light: tolerance_scale times the requirements, and the approach
  bottleneck weight. A hint about a whole motion, not a requirement about a state.

  Measured against the fixed requirements, not the reward curriculum. The two arrival
  terms chase terminal accuracy and should sharpen as the policy does. This one asks
  whether the robot is on the recorded motion at all, which has a fixed answer. Hanging it
  off the curriculum would open it at 10x the requirement, where a metre of root error and
  a centimetre score alike.

  Zero where there is no crossing, which is a window aimed from outside through
  open_window. Multiplied rather than branched, so the shapes match either way.
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


def feet_below_ground(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.0
) -> torch.Tensor:
  """How far the lowest foot is under the floor. Should read exactly zero, always.

  Physics makes this nearly impossible, which is most of the reason the middle of a window
  comes from a policy in a simulator rather than a model that regresses frames: the
  supervised attempt this replaced put a foot through the floor in 2 windows out of 5.
  Kept as a term because a number pinned at zero in the log is the evidence it stays
  fixed.
  """
  asset = env.scene[asset_cfg.name]
  assert asset_cfg.body_ids is not None
  heights = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  return (threshold - heights.min(dim=-1).values).clamp(min=0.0)


def feet_slip(
  env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Foot sliding along the floor: horizontal speed while in contact.

  Local, not reused from the velocity tasks. Theirs scales by how fast the robot was told
  to go, reading command channels 0-1 as a linear velocity and 2 as a yaw rate. The bridge
  command holds a target offset in those channels, so the numbers would look plausible and
  mean nothing, and the coupling would stay invisible until somebody reordered the
  observation.

  Ungated, which is what a bridge wants: a planted foot has no horizontal speed, so
  standing still already costs nothing, and no moment in a window makes dragging a foot
  right.
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
  """The two legs counter rotating so the knees face each other.

  Both hip yaw joints turn about the same axis, so legs turning together is a twist of the
  whole lower body and legs turning against each other is the knees converging. Right
  minus left separates them: zero for any twist however far the robot turned, growing only
  as the knees converge. A term built on each joint's own magnitude cannot tell the two apart
  and would tax every turning jump in the corpus to catch something none of them do.

  Threshold measured, not chosen. Over 84705 states of walk, run and jump:

      p99                                0.24
      p99.9                              0.43
      walk max                           0.40
      run max                            0.18
      bridge at 2600 iterations, median  0.47

  The bridge spends over half its time more knock kneed than 99.9% of what the skills do,
  so a bound in that gap costs the objective almost nothing. Quadratic past it, exactly
  zero inside.

  Joints resolved by name, not in the order asset_cfg gave them. Which comes first decides
  the sign, and a term that silently rewards what it forbids is invisible in a reward
  curve.
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
  """A contact or flight phase that ends sooner than a real step would.

  Charged at the transition, once per phase, and only for the part below min_time. Longer
  than that costs exactly zero, so a planted foot is free and a proper step is free. The
  only thing priced is a phase too short to be either.

  That one sidedness is the design. The term this replaced paid for air time, which is an
  instruction to walk, and a bridge asked to arrive in a crouch was being paid to do
  something else at the same time.

  A zero previous phase means there was no previous phase, which is what a foot looks like
  on the step after a teleport. Charging it would fine every window for its own start.
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
  """How far short of the floor each phase fell. Zero for every phase that reached it."""
  return (elapsed.clamp(max=floor) - floor).neg()
