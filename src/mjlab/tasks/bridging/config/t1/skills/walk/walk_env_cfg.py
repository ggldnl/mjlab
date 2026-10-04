"""T1 flat velocity task plus the LAFAN style reward.

Everything of the stock task is kept except three things:

    style           new, nearest mocap frame reward, see style.py
    pose            narrowed to the head, which the library leaves out. On the body it
                    pulled towards the home pose and fought the style term
    twist ranges    held inside the library's velocity coverage, no speed curriculum

The command ranges are a first guess at what LAFAN walks cover once scaled to the T1.
Compare them with the coverage dataset.py prints and narrow them where the data is thin.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.bridging.config.t1.skills.walk.style import LIBRARY_FILE, motion_style
from mjlab.tasks.velocity.config.t1.env_cfgs import booster_t1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

STYLE_WEIGHT = 1.0
"""Against 2.0 on each velocity tracking term, so tracking still wins a conflict."""

COMMAND_RANGES = {
  "lin_vel_x": (-0.5, 1.0),
  "lin_vel_y": (-0.3, 0.3),
  "ang_vel_z": (-0.8, 0.8),
}

HEAD_JOINTS = ("AAHead_yaw", "Head_pitch")


def t1_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The T1 flat velocity task with the style reward on top."""
  cfg = booster_t1_flat_env_cfg(play=play)

  cfg.rewards["style"] = RewardTermCfg(
    func=motion_style,
    weight=STYLE_WEIGHT,
    params={
      "library_file": str(LIBRARY_FILE),
      "command_name": "twist",
      "joint_pos_std": 0.2,  # rad, rms over joints
      "joint_vel_std": 2.0,  # rad/s, rms over joints
      "lin_vel_std": 0.25,  # m/s
      "ang_vel_std": 0.4,  # rad/s
      "chunk_size": 1024,
    },
  )

  pose = cfg.rewards["pose"]
  pose.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=HEAD_JOINTS)
  for key in ("std_standing", "std_walking", "std_running"):
    pose.params[key] = {".*": 0.05}

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  for name, value in COMMAND_RANGES.items():
    setattr(twist.ranges, name, value)
  twist.ranges.heading = (-math.pi, math.pi)
  cfg.curriculum.pop("command_vel", None)

  return cfg
