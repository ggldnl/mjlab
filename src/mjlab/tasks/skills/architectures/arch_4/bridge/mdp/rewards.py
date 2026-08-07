"""What the bridge is paid for: reproducing the motion that was cut out, under physics.

One exponential kernel per group, because the groups have incompatible units and a tenth
of a radian of joint error and a tenth of a metre per second of velocity error are not the
same mistake. Kernels rather than a plain negative distance for the usual reason a tracking
reward uses them: a policy that cannot yet track should not be handed a gradient dominated
by how spectacularly it is failing.

Every tracking term is strictly positive and multiplied by the arrival ramp, which is the
one place the architecture's priority is written down. Two properties follow and both are
deliberate.

Positive per-step reward means ending an episode early is always worse than continuing it,
so there is no suicide trap: this project has already seen a bridge discover that falling
over promptly beats trying, and that came from summed kernels that could go slack. What
keeps the policy from parking in a mediocre-but-survivable state instead is termination on
lost tracking, not the reward.

The ramp means the frames near the far end of a hole count for several times what the
frames near the hand-off do. Filling a hole plausibly and arriving somewhere else is the
exact failure this whole architecture exists to prevent, and a flat reward would score it
the same as arriving correctly.
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
  return _kernel(error, std, command.arrival_weight())


def joint_vel_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """The pose is not enough: a limb passing through the right angle at the wrong speed
  is on its way somewhere else."""
  command = _command(env, command_name)
  reference = command.reference_now()
  target = reference[:, ROOT_STATE_DIM + command.num_joints :]
  error = torch.square(command.robot.data.joint_vel - target).mean(dim=-1)
  return _kernel(error, std, command.arrival_weight())


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
  return _kernel(error, std, command.arrival_weight())


def root_ori_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _command(env, command_name)
  error = torch.square(
    quat_error_magnitude(
      command.robot.data.root_link_quat_w, command.reference_now()[:, 3:7]
    )
  )
  return _kernel(error, std, command.arrival_weight())


def root_lin_vel_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """The momentum term. Arriving at the right place stopped is not arriving."""
  command = _command(env, command_name)
  error = torch.square(
    command.robot.data.root_link_lin_vel_w - command.reference_now()[:, 7:10]
  ).sum(dim=-1)
  return _kernel(error, std, command.arrival_weight())


def root_ang_vel_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _command(env, command_name)
  error = torch.square(
    command.robot.data.root_link_ang_vel_w
    - command.reference_now()[:, 10:ROOT_STATE_DIM]
  ).sum(dim=-1)
  return _kernel(error, std, command.arrival_weight())


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
