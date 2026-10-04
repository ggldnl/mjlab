"""Push environment builder, shared by the T1 and G1 push tasks.

Starts from the robot's flat velocity task and keeps its walking rewards. The twist they
read is the pace command, which the push derives every step, so the robot is paid for a
steady forward walk at the box speed and never sees a twist itself.

Scene     robot at the center of a 1 m cell, facing a 1 m cube in the next cell
Default   arms straight forward. It is the posture reward target and the action zero
Actor     proprioception, push command, box state, hand offsets and hand contacts
Reward    walking terms plus box speed, final position, hand contact, minus illegal
          contact, trunk too close to the box, both feet off the ground, box off lane,
          tipped or turned
Ends      fall, box off lane, tipped, lifted, turned or pushed past the target
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from mjlab.asset_zoo.objects.box import get_box_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.bridging.config.t1.skills.push import mdp
from mjlab.tasks.bridging.config.t1.skills.push.command import (
  PushCommandCfg,
  PushPaceCommandCfg,
)
from mjlab.tasks.velocity.config.t1.env_cfgs import booster_t1_flat_env_cfg
from mjlab.utils.spec_config import CollisionCfg

# Robot spawns at the center of cell (0, 0), the box in cell (1, 0)
ROBOT_CELL = (0.5, 0.5)
BOX_CELL = (1.5, 0.5)

# Placement error the skill accepts at entry, matching the Sokoban controller checks
ROBOT_POSE_RANGE = {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "yaw": (-0.1, 0.1)}
BOX_POSE_RANGE = {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-0.04, 0.04)}


@dataclass(frozen=True)
class PushRobot:
  """What the builder needs to know about a robot"""

  arm_pose: dict[str, float]  # straight arms forward, replaces the arm defaults
  arm_std: dict[str, float]  # posture tolerance of the arm joints while walking
  hands: ContactMatch  # hands against the box, one column per hand
  hand_geoms: tuple[str, ...]  # hand geoms observed relative to the box
  root_body: str  # subtree holding every collision geom of the robot
  min_clearance: float  # root to box face, a little under the straight arm reach
  box_mass: float
  box_friction: float = 0.4
  # Furthest forward placement error, kept short of the reach so hands never spawn
  # inside the box
  max_forward_error: float = 0.08


def box_sensors(name: str, robot: PushRobot) -> tuple[ContactSensorCfg, ...]:
  box = ContactMatch(mode="geom", pattern="box_collision", entity=name)
  return (
    ContactSensorCfg(
      name=f"hands_{name}",
      primary=robot.hands,
      secondary=box,
      fields=("found",),
      reduce="netforce",
      num_slots=1,
    ),
    ContactSensorCfg(
      name=f"robot_{name}",
      primary=ContactMatch(mode="subtree", pattern=robot.root_body, entity="robot"),
      secondary=box,
      fields=("found",),
      reduce="netforce",
      num_slots=1,
    ),
  )


def make_push_env_cfg(
  cfg: ManagerBasedRlEnvCfg, robot: PushRobot
) -> ManagerBasedRlEnvCfg:
  """Turn a flat velocity task config into the push task"""
  # Scene
  entity = cfg.scene.entities["robot"]
  # Drop every default entry that names an arm joint, the keyframes mix regexes with names
  base = {
    key: value
    for key, value in (entity.init_state.joint_pos or {}).items()
    if not any(re.fullmatch(key, joint) for joint in robot.arm_pose)
  }
  entity.init_state = replace(
    entity.init_state,
    pos=(*ROBOT_CELL, entity.init_state.pos[2]),
    joint_pos={**base, **robot.arm_pose},
  )
  cfg.scene.entities["box"] = get_box_cfg(
    half_size=(0.5, 0.5, 0.5),
    init_x=BOX_CELL[0],
    mass=robot.box_mass,
    friction=robot.box_friction,
    priority=2,
    color=(0.55, 0.4, 0.25, 1.0),
  )
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + box_sensors("box", robot)
  cfg.scene.env_spacing = 10.0
  cfg.sim.njmax = 600
  cfg.sim.contact_sensor_maxmatch = 256

  # Commands
  cfg.commands = {
    "push": PushCommandCfg(resampling_time_range=(1.0e9, 1.0e9), gui=False),
    "pace": PushPaceCommandCfg(gui=False),
  }
  cfg.curriculum = {}

  # Events
  cfg.events.pop("push_robot", None)
  pose_range = cfg.events["reset_base"].params["pose_range"]
  pose_range.update(ROBOT_POSE_RANGE)
  pose_range["x"] = (
    pose_range["x"][0],
    min(pose_range["x"][1], robot.max_forward_error),
  )
  cfg.events["reset_box"] = EventTermCfg(
    func=mdp.reset_box,
    mode="reset",
    params={"pose_range": BOX_POSE_RANGE, "cell": BOX_CELL},
  )
  # Mass and inertia together, about 0.6 to 1.6 times the nominal mass
  cfg.events["box_inertia"] = EventTermCfg(
    func=dr.pseudo_inertia,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("box", body_names=("box",)),
      "alpha_range": (-0.25, 0.25),
    },
  )
  cfg.episode_length_s = 20.0

  # Observations
  hands_cfg = SceneEntityCfg("robot", geom_names=robot.hand_geoms)
  push_terms = {
    "command": ObservationTermCfg(func=mdp.push_command),
    "box": ObservationTermCfg(func=mdp.box_features),
    "hand_contact": ObservationTermCfg(func=mdp.hand_contact),
    "hands": ObservationTermCfg(func=mdp.hand_offsets, params={"hands_cfg": hands_cfg}),
  }
  for group in ("actor", "critic"):
    cfg.observations[group].terms.update(push_terms)
  cfg.observations["critic"].terms["body_contact"] = ObservationTermCfg(
    func=mdp.body_contact
  )

  # Walking rewards follow the pace, not a sampled twist
  for reward in cfg.rewards.values():
    if reward.params.get("command_name") == "twist":
      reward.params["command_name"] = "pace"
  cfg.rewards["track_linear_velocity"].params["std"] = 0.25
  cfg.rewards["track_angular_velocity"].weight = 1.0
  for regime in ("std_walking", "std_running"):
    cfg.rewards["pose"].params[regime].update(robot.arm_std)

  cfg.rewards.update(
    {
      "box_velocity": RewardTermCfg(
        func=mdp.box_velocity, weight=3.0, params={"std": 0.15}
      ),
      "box_position": RewardTermCfg(
        func=mdp.box_position, weight=1.0, params={"std": 0.1}
      ),
      "hands_on_box": RewardTermCfg(func=mdp.hands_on_box, weight=1.0),
      "body_contact": RewardTermCfg(func=mdp.illegal_contact, weight=-3.0),
      "body_clearance": RewardTermCfg(
        func=mdp.body_clearance,
        weight=-2.0,
        params={"min_clearance": robot.min_clearance},
      ),
      "flight_phase": RewardTermCfg(
        func=mdp.flight_phase,
        weight=-2.0,
        params={"sensor_name": "feet_ground_contact"},
      ),
      "box_lane": RewardTermCfg(func=mdp.box_lane_error, weight=-0.2),
      "box_tilt": RewardTermCfg(func=mdp.box_tilt_error, weight=-0.2),
      "box_yaw": RewardTermCfg(func=mdp.box_yaw_error, weight=-0.2),
      "box_angular_velocity": RewardTermCfg(func=mdp.box_angular_velocity, weight=-0.5),
      "termination": RewardTermCfg(func=envs_mdp.is_terminated, weight=-100.0),
    }
  )

  # Terminations
  cfg.terminations["invalid_box"] = TerminationTermCfg(func=mdp.invalid_box)

  # Metrics
  cfg.metrics["push_hands_contact"] = MetricsTermCfg(func=mdp.hands_contact_rate)
  cfg.metrics["push_body_contact"] = MetricsTermCfg(func=mdp.body_contact_rate)
  cfg.metrics["push_flight"] = MetricsTermCfg(
    func=mdp.flight_rate, params={"sensor_name": "feet_ground_contact"}
  )
  return cfg


##
# T1
##

# Arms rolled down, then pitched forward with straight elbows, slightly below horizontal.
# Fingertips 0.48 m ahead of the trunk at 0.76 m height, 0.37 m apart
T1_ARM_POSE = {
  "Left_Shoulder_Pitch": -1.0,
  "Left_Shoulder_Roll": -1.45,
  "Left_Elbow_Pitch": 0.0,
  "Left_Elbow_Yaw": -0.5,
  "Right_Shoulder_Pitch": -1.0,
  "Right_Shoulder_Roll": 1.45,
  "Right_Elbow_Pitch": 0.0,
  "Right_Elbow_Yaw": 0.5,
}

T1_ARM_STD = {
  r".*Shoulder_Pitch": 0.25,
  r".*Shoulder_Roll": 0.2,
  r".*Elbow_Pitch": 0.2,
  r".*Elbow_Yaw": 0.3,
}

# Named physical geoms in the T1 XML, excluding the visual meshes. The stock task collides
# feet only, the push needs the whole body against the box
T1_PHYSICAL_GEOMS = (
  r"^(Trunk|H[12]|A[LR][123]|Waist|Hip_.*|Shank_.*|Ankle_.*|left_.*link|right_.*link)$",
)

T1 = PushRobot(
  arm_pose=T1_ARM_POSE,
  arm_std=T1_ARM_STD,
  hands=ContactMatch(
    mode="subtree", pattern=r"^(left|right)_hand_link$", entity="robot"
  ),
  hand_geoms=(r"^(left|right)_hand_link$",),
  root_body="Trunk",
  min_clearance=0.4,
  # At friction 0.4 it slides at 31 N and tips at 52 N with the hands at 0.76 m
  box_mass=8.0,
  max_forward_error=0.02,
)


def t1_push_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = booster_t1_flat_env_cfg(play=play)
  robot = cfg.scene.entities["robot"]
  robot.collisions = (
    CollisionCfg(
      geom_names_expr=T1_PHYSICAL_GEOMS,
      contype=0,
      conaffinity=1,
      condim=3,
      priority=1,
      friction=(0.8,),
    ),
  )
  return make_push_env_cfg(cfg, T1)
