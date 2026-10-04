"""Style reward: how close the robot is to the nearest mocap frame moving at the command.

A nearest neighbour lookup in place of AMP's discriminator. Every step, the robot's joint
positions and velocities and the velocity command form one feature vector. The library frames
form the same vector from their joints and their velocity label (see dataset.py). The reward
is a kernel on the distance to the closest frame:

    d = min over frames of  |q - q_i|^2 / (n sq^2) + |dq - dq_i|^2 / (n sdq^2)
                            + |v_cmd - v_i|^2 / sv^2
    r = exp(-d)

The command, not the measured velocity, picks the frames. Matching on measured velocity would
let a slow walking style score well while the command asks for a fast one. Joint velocities
are what stop the policy from freezing in one matching pose: a frozen pose matches no walking
frame, and they also fix the direction the gait cycle runs in.

The command term is a constant per frame, so a command outside the data's coverage only scales
the reward down. It does not push the robot anywhere, which is why the command ranges should
stay inside what dataset.py prints.

The pairwise distance is chunked over envs. 4096 envs against 50k frames is 800 MB in one go.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg

LIBRARY_FILE = Path("data/t1_walk_style/frames.npz")
"""Written by dataset.py, read by the reward at env construction."""

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class motion_style:
  """Nearest frame style reward, see the module docstring."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    library_file = Path(params["library_file"])
    if not library_file.is_file():
      raise FileNotFoundError(
        f"No style library at {library_file}. Build it with\n"
        "  uv run python -m mjlab.tasks.bridging.config.t1.skills.walk.dataset"
      )
    with np.load(library_file, allow_pickle=False) as library:
      names = [str(name) for name in library["joint_names"]]
      joint_pos = library["joint_pos"]
      joint_vel = library["joint_vel"]
      velocity = library["velocity"]

    asset: Entity = env.scene[params.get("asset_cfg", _DEFAULT_ASSET_CFG).name]
    joint_ids, _ = asset.find_joints(names, preserve_order=True)
    self.joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)

    count = len(names)
    self.scale = torch.tensor(
      [1 / (params["joint_pos_std"] * math.sqrt(count))] * count
      + [1 / (params["joint_vel_std"] * math.sqrt(count))] * count
      + [1 / params["lin_vel_std"]] * 2
      + [1 / params["ang_vel_std"]],
      device=env.device,
    )
    frames = np.concatenate([joint_pos, joint_vel, velocity], axis=-1)
    self.frames = torch.as_tensor(frames, device=env.device) * self.scale

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    library_file: str,
    command_name: str,
    joint_pos_std: float,
    joint_vel_std: float,
    lin_vel_std: float,
    ang_vel_std: float,
    chunk_size: int = 1024,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del library_file, joint_pos_std, joint_vel_std, lin_vel_std, ang_vel_std  # baked in
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    features = torch.cat(
      [
        asset.data.joint_pos[:, self.joint_ids],
        asset.data.joint_vel[:, self.joint_ids],
        command[:, :3],
      ],
      dim=-1,
    )
    features = features * self.scale
    nearest = torch.cat(
      [
        torch.cdist(chunk, self.frames).min(dim=-1).values
        for chunk in features.split(chunk_size)
      ]
    )
    return torch.exp(-nearest.square())
