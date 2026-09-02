"""How a window ends.

    deadline_reached   clock ran out. Time-out, so the critic bootstraps
    strayed            robot walked away from the target. Failure
    fell_over          torso tipped past recovering. Failure

Time-out vs failure matters. Bootstrapping off a failure tells the critic a robot on the
floor is worth whatever it happened to estimate.

`strayed` is what stops the policy parking somewhere safe. Every reward term is positive,
so standing still always pays something; without this, collecting a small kernel for the
whole window beats crossing it.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.commands import (
  BridgeCommand,
)


def _command(env: ManagerBasedRlEnv, command_name: str) -> BridgeCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, BridgeCommand)
  return term


def deadline_reached(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Clock ran out. Time-out, not failure.

  The final step is still scored. Terminations run before rewards, so the step that trips
  this is the one `arrival` is paid for and the one the arrival is latched from.
  """
  command = _command(env, command_name)
  return command.step >= command.deadline


def strayed(
  env: ManagerBasedRlEnv, command_name: str, margin: float = 1.5
) -> torch.Tensor:
  """Robot is further from the target than the time left can recover.

  Measured against the distance the window opened at, not an absolute bound: opening two
  metres out and opening at arm's length are different questions.

  `margin` is generous on purpose. A run-up, a turn, or gathering itself all open the
  distance before closing it. Walking away is what this catches.
  """
  command = _command(env, command_name)
  distance = (command.robot.data.root_link_pos_w - command.target[:, 0:3]).norm(dim=-1)
  return distance > command.start_distance + margin


def fell_over(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.7
) -> torch.Tensor:
  """Torso tipped past recovering. Fires past about 46 degrees of tilt.

  Read off projected gravity, not root height. A deep crouch and a fall reach the same
  height and only one is a failure, and a corpus built from jumps is full of crouches.

  Measured on the tracker corpus: walk states never exceed 30 degrees of tilt, and
  everything past 46 sits at a pelvis height of 0.20 m, which is a robot on the floor. So
  this does not currently block any posture the corpus contains. If crouch-rich clips are
  added it may start to, since a real squat pitches the torso 50 to 60 degrees.
  """
  asset = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b[:, 2] > -threshold
