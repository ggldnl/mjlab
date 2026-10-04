"""The T1 push task on the G1: same command, rewards and scene, G1 robot numbers."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch
from mjlab.tasks.bridging.config.t1.skills.push.push_env_cfg import (
  PushRobot,
  make_push_env_cfg,
)
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

# Straight arms forward, the G1 elbow is straight near 1.4. Hands 0.38 m ahead of the
# pelvis, 0.30 m apart, 0.87 m high
G1_ARM_POSE = {
  "left_shoulder_pitch_joint": -1.0,
  "right_shoulder_pitch_joint": -1.0,
  "left_shoulder_roll_joint": 0.15,
  "right_shoulder_roll_joint": -0.15,
  "left_elbow_joint": 1.4,
  "right_elbow_joint": 1.4,
}

G1_ARM_STD = {
  r".*shoulder_pitch.*": 0.25,
  r".*shoulder_roll.*": 0.2,
  r".*shoulder_yaw.*": 0.15,
  r".*elbow.*": 0.2,
  r".*wrist.*": 0.3,
}

G1 = PushRobot(
  arm_pose=G1_ARM_POSE,
  arm_std=G1_ARM_STD,
  hands=ContactMatch(
    mode="geom", pattern=r"^(left|right)_hand_collision$", entity="robot"
  ),
  hand_geoms=(r"^(left|right)_hand_collision$",),
  root_body="pelvis",
  min_clearance=0.32,
  # At friction 0.4 it slides at 47 N and tips at 68 N with the hands at 0.87 m
  box_mass=12.0,
)


def g1_push_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_push_env_cfg(unitree_g1_flat_env_cfg(play=play), G1)
