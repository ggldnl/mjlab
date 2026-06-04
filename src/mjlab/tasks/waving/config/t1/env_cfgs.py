"""Booster T1 waving environment configurations."""

from mjlab.asset_zoo.robots import T1_ACTION_SCALE, get_t1_robot_cfg
from mjlab.asset_zoo.robots.booster_t1.t1_constants import BASE_BODY_NAME
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.waving.waving_env_cfg import make_waving_env_cfg

# The right arm does the waving; everything else holds the standing pose.
WAVE_JOINT_NAMES = (
  "Right_Shoulder_Pitch",
  "Right_Shoulder_Roll",
  "Right_Elbow_Pitch",
  "Right_Elbow_Yaw",
)
STANDING_JOINT_NAMES = (
  r".*Hip_.*",
  r".*Knee_Pitch",
  r".*Ankle_.*",
  "Waist",
  r"Left_Shoulder_.*",
  r"Left_Elbow_.*",
  "AAHead_yaw",
  "Head_pitch",
)


def booster_t1_waving_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Booster T1 stand-and-wave configuration."""
  cfg = make_waving_env_cfg()

  cfg.scene.entities = {"robot": get_t1_robot_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  # The default 0.25 scale keeps actions near the standing pose, but the raised
  # wave target sits ~2.3 rad from the arm's default. Give the waving arm a
  # unit scale so that target is reachable with normal-magnitude actions
  # (raw action ~ the offset itself) instead of needing huge ones.
  action_scale = dict(T1_ACTION_SCALE)
  for joint_name in WAVE_JOINT_NAMES:
    action_scale[joint_name] = 1.0
  joint_pos_action.scale = action_scale

  cfg.viewer.body_name = BASE_BODY_NAME

  # Wave shape, as offsets from the default arm pose (radians). The hand is
  # raised above the head and waves side to side overhead.
  #
  # T1 right-arm kinematics: Right_Shoulder_Roll (axis x, range [-1.57, 1.74])
  # sets the arm elevation. At its default of 1.3 the arm hangs down at the
  # side; driving it strongly negative swings the arm up in the frontal plane
  # until it points straight overhead. The elbow is kept straight so the hand
  # stays at the top of the reach (bending it would drop the hand). The wave
  # itself is a side-to-side rock of the overhead arm via the same shoulder
  # roll, with a small elbow-yaw hand flourish. Ranges stay inside the 0.9
  # soft-limit factor so the arm is not fighting its joint stops.
  wave = cfg.rewards["wave"]
  wave.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=WAVE_JOINT_NAMES)
  wave.params["center"] = {
    "Right_Shoulder_Pitch": 0.0,  # Keep near-vertical (slight natural forward).
    "Right_Shoulder_Roll": -2.3,  # 1.3 default -> -1.0: arm raised overhead.
    "Right_Elbow_Pitch": 0.0,  # Forearm straight, hand at the top.
    "Right_Elbow_Yaw": 0.0,
  }
  wave.params["amplitude"] = {
    "Right_Shoulder_Pitch": 0.0,
    "Right_Shoulder_Roll": 0.3,  # Side-to-side rock of the overhead arm.
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.3,  # Hand flourish.
  }

  # Hold everything else near the standing pose. Legs/waist/left arm are tight;
  # the head is tightest since it should stay still while greeting.
  posture = cfg.rewards["posture"]
  posture.params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=STANDING_JOINT_NAMES
  )
  posture.params["std"] = {
    r".*Hip_Pitch": 0.1,
    r".*Hip_Roll": 0.1,
    r".*Hip_Yaw": 0.1,
    r".*Knee_Pitch": 0.1,
    r".*Ankle_Pitch": 0.1,
    r".*Ankle_Roll": 0.1,
    "Waist": 0.1,
    "Left_Shoulder_Pitch": 0.1,
    "Left_Shoulder_Roll": 0.1,
    "Left_Elbow_Pitch": 0.1,
    "Left_Elbow_Yaw": 0.1,
    "AAHead_yaw": 0.05,
    "Head_pitch": 0.05,
  }

  if play:
    cfg.scene.num_envs = 16
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.episode_length_s = int(1e9)  # Effectively infinite for play.

  return cfg
