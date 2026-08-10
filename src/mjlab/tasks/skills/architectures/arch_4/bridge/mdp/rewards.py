"""What the bridge is paid for: reproducing the motion that was cut out, under physics.

One exponential kernel per group, because the groups have incompatible units and a tenth
of a radian of joint error and a tenth of a metre per second of velocity error are not the
same mistake. Kernels rather than a plain negative distance for the usual reason a tracking
reward uses them: a policy that cannot yet track should not be handed a gradient dominated
by how spectacularly it is failing.

Every tracking term is strictly positive and multiplied by the arrival ramp, which is the
one place the architecture's priority is written down. Two properties follow and both are
deliberate.

The ramp comes through `tracking_weight`, which is the ramp with a gate on it: zero for
any step where the reference is not a recording of anything. That is the hole of a spliced
window and nothing else (see mdp/commands.py). Those steps are paid by `approach` instead,
which knows only where the robot has to end up, and the two gates are complements, so
every step of every episode is paid by exactly one of them.

Positive per-step reward means ending an episode early is always worse than continuing it,
so there is no suicide trap: this project has already seen a bridge discover that falling
over promptly beats trying, and that came from summed kernels that could go slack. What
keeps the policy from parking in a mediocre-but-survivable state instead is termination on
lost tracking, not the reward.

The ramp means the frames near the far end of a hole count for several times what the
frames near the hand-off do. Filling a hole plausibly and arriving somewhere else is the
exact failure this whole architecture exists to prevent, and a flat reward would score it
the same as arriving correctly.

The tracking terms run through the resume as well as the hole, ungated, so a bridge that
arrives badly still has every reason to recover rather than give up: at inference it will
be handing a real skill a real state, and a state it has partly salvaged is worth more
than one it has abandoned.

What is gated is `resumed`, and that is the term carrying the argument the resume exists
for. It pays a flat amount for every frame of the second window that actually gets played
out, multiplied by how good the hand-off was, so the second window is worth what the
arrival earned and nothing more. Two ways of losing it, both of them the point: hand over
badly and each frame is worth little, or fall over during the resume and the frames stop
arriving. Neither is available to a policy scored on the hole alone.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.skills.architectures.arch_4.bridge.dataset import ROOT_STATE_DIM
from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.commands import BridgeCommand
from mjlab.utils.lab_api.math import quat_error_magnitude


def _command(env: ManagerBasedRlEnv, command_name: str) -> BridgeCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, BridgeCommand)
  return term


def _kernel(error: torch.Tensor, std: float, weight: torch.Tensor) -> torch.Tensor:
  return weight * torch.exp(-error / (std * std))


def joint_pos_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """The pose itself: are the limbs where the body had them."""
  command = _command(env, command_name)
  reference = command.reference_now()
  target = reference[:, ROOT_STATE_DIM : ROOT_STATE_DIM + command.num_joints]
  error = torch.square(command.robot.data.joint_pos - target).mean(dim=-1)
  return _kernel(error, std, command.tracking_weight())


def joint_vel_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """The pose is not enough: a limb passing through the right angle at the wrong speed
  is on its way somewhere else."""
  command = _command(env, command_name)
  reference = command.reference_now()
  target = reference[:, ROOT_STATE_DIM + command.num_joints :]
  error = torch.square(command.robot.data.joint_vel - target).mean(dim=-1)
  return _kernel(error, std, command.tracking_weight())


def root_pos_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Where the body ended up. The reference is rebased onto this environment's origin at
  reset, so this is a plain world-space distance and it measures exactly the thing the
  descriptor will care about: did the robot actually travel."""
  command = _command(env, command_name)
  error = torch.square(
    command.robot.data.root_link_pos_w - command.reference_now()[:, 0:3]
  ).sum(dim=-1)
  return _kernel(error, std, command.tracking_weight())


def root_ori_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _command(env, command_name)
  error = torch.square(
    quat_error_magnitude(
      command.robot.data.root_link_quat_w, command.reference_now()[:, 3:7]
    )
  )
  return _kernel(error, std, command.tracking_weight())


def root_lin_vel_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """The momentum term. Arriving at the right place stopped is not arriving."""
  command = _command(env, command_name)
  error = torch.square(
    command.robot.data.root_link_lin_vel_w - command.reference_now()[:, 7:10]
  ).sum(dim=-1)
  return _kernel(error, std, command.tracking_weight())


def root_ang_vel_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _command(env, command_name)
  error = torch.square(
    command.robot.data.root_link_ang_vel_w
    - command.reference_now()[:, 10:ROOT_STATE_DIM]
  ).sum(dim=-1)
  return _kernel(error, std, command.tracking_weight())


def approach(
  env: ManagerBasedRlEnv, command_name: str, pos_std: float, ori_std: float
) -> torch.Tensor:
  """The hole of a spliced window, where the only thing known is the destination.

  Zero everywhere `tracking_weight` is not, which is everywhere with a reference. Where
  there is none, the frames that were recorded in the hole led to a continuation that has
  been replaced, so scoring against them would be asking the bridge to head somewhere it is
  no longer going. What survives the splice is the arrival, and this is the distance to it:
  how far the root still has to travel and how far it still has to turn, one kernel each,
  averaged.

  Distance to a fixed target rather than ground covered per step, which is the other
  obvious shaping and is worse. A per-step term pays for closing distance and would have to
  charge for opening it, or a policy could farm it by swinging toward the target and away
  again; charging for it forbids the run-up, and a body that has to turn around or gather
  itself before it can go anywhere opens the distance first every time. A function of the
  state has no such history to exploit and no such motion to forbid.

  The standard deviations are wide on purpose. This is the only gradient in a spliced hole
  and a kernel that has already saturated at the distances a hole starts from would leave
  the bridge with nothing to follow for the first half of it.

  It carries the same arrival ramp as the tracking terms, so the frames near the hand-off
  are worth several times the frames near the start, and it is weighted in `env_cfg` to sum
  to what they sum to. A bridge should not be able to prefer one kind of window to the
  other, and the critic cannot see which kind it is in.
  """
  command = _command(env, command_name)
  target = command.arrival_target()
  robot = command.robot.data
  distance = torch.square(robot.root_link_pos_w - target[:, 0:3]).sum(dim=-1)
  turn = torch.square(quat_error_magnitude(robot.root_link_quat_w, target[:, 3:7]))
  closeness = 0.5 * (
    torch.exp(-distance / (pos_std * pos_std)) + torch.exp(-turn / (ori_std * ori_std))
  )
  return closeness * command.approach_weight()


def resumed(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """One frame of the second window, worth what the hand-off earned.

  Zero inside the hole and up to the hand-off, `hand_off_score` for every frame after it.
  Summed over an episode this is the resume's length times how usable the arrival was,
  which is as close as a corpus can come to the question that matters at inference: how
  far does the next motion get, given the state the bridge left it in.

  The gate is deliberately not something the resume itself can move. `hand_off_score` is
  measured once, at the frame the next skill would take over, and held; a bridge cannot
  arrive wrong and then earn the second window back by tracking it well, because at
  inference it will not be the one tracking it. What it can still do is keep the robot
  alive, which is what the ungated tracking terms above are for.
  """
  command = _command(env, command_name)
  return command.resume_credit()


def feet_below_ground(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.0
) -> torch.Tensor:
  """How far the lowest foot is under the floor, as a cost.

  Physics makes this nearly impossible, which is the entire reason for moving to a
  simulator: the supervised model this replaced put a foot more than two centimetres
  through the floor in two windows out of five. Kept as a term anyway because it is the
  defect that condemned the previous approach, and a number that stays at zero in the
  training log is how we know the move worked.
  """
  asset = env.scene[asset_cfg.name]
  assert asset_cfg.body_ids is not None
  heights = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  return (threshold - heights.min(dim=-1).values).clamp(min=0.0)
