"""Booster T1 velocity environment configurations."""

from dataclasses import replace

from mjlab.asset_zoo.robots import (
  T1_ACTION_SCALE,
  get_t1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.config import STAIRS_TERRAINS_CFG

# Trunk is the floating base; the IMU site and subtree angular momentum sensor
# both live on it.
BASE_BODY_NAME = "Trunk"
FOOT_SITE_NAMES = ("left_foot", "right_foot")
# Only the foot contact spheres collide with the ground (FEET_ONLY_COLLISION),
# so friction randomization targets those geoms.
FOOT_GEOM_REGEX = (r"^(left|right)_foot_sphere.*link$",)


def booster_t1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster T1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_t1_robot_cfg()}

  # Set raycast sensor frame to T1 trunk.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = BASE_BODY_NAME

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITE_NAMES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_foot_link|right_foot_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg,)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = T1_ACTION_SCALE

  cfg.viewer.body_name = BASE_BODY_NAME

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.0

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_REGEX
  cfg.events["base_com"].params["asset_cfg"].body_names = (BASE_BODY_NAME,)

  # Rationale for std values (mirrors the G1 setup):
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist stays tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - The head stays tight since it doesn't contribute to locomotion.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*Hip_Pitch": 0.3,
    r".*Hip_Roll": 0.15,
    r".*Hip_Yaw": 0.15,
    r".*Knee_Pitch": 0.35,
    r".*Ankle_Pitch": 0.25,
    r".*Ankle_Roll": 0.1,
    # Waist.
    r"Waist": 0.2,
    # Arms.
    r".*Shoulder_Pitch": 0.15,
    r".*Shoulder_Roll": 0.15,
    r".*Elbow_Pitch": 0.15,
    r".*Elbow_Yaw": 0.15,
    # Head.
    r"AAHead_yaw": 0.05,
    r"Head_pitch": 0.05,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*Hip_Pitch": 0.5,
    r".*Hip_Roll": 0.2,
    r".*Hip_Yaw": 0.2,
    r".*Knee_Pitch": 0.6,
    r".*Ankle_Pitch": 0.35,
    r".*Ankle_Roll": 0.15,
    # Waist.
    r"Waist": 0.3,
    # Arms.
    r".*Shoulder_Pitch": 0.5,
    r".*Shoulder_Roll": 0.2,
    r".*Elbow_Pitch": 0.35,
    r".*Elbow_Yaw": 0.35,
    # Head.
    r"AAHead_yaw": 0.05,
    r"Head_pitch": 0.05,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = (BASE_BODY_NAME,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (BASE_BODY_NAME,)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = FOOT_SITE_NAMES

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def booster_t1_stairs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster T1 stair-climbing velocity configuration.

  The task is just the rough-terrain velocity setup with the terrain swapped
  for a stairs-only curriculum (row 0 is flat; higher rows raise the step
  height). The policy is blind-proprioceptive plus the existing height-scan;
  the terrain-level curriculum promotes envs that walk far enough on their
  current step height.
  """

  cfg = booster_t1_rough_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = replace(STAIRS_TERRAINS_CFG)
  cfg.scene.terrain.terrain_generator.curriculum = not play
  # Start every env on the flat row and let the curriculum raise the steps.
  cfg.scene.terrain.max_init_terrain_level = 0

  if play:
    assert cfg.scene.terrain.terrain_generator is not None
    cfg.scene.terrain.terrain_generator.num_rows = 5
    cfg.scene.terrain.terrain_generator.num_cols = 5
    cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def booster_t1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster T1 flat terrain velocity configuration."""
  cfg = booster_t1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def booster_t1_running_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster T1 flat-terrain running configuration.

  This is the flat velocity-tracking task with a higher top forward speed. The
  sampled command range and the final velocity-curriculum stage are raised so
  the policy is trained to track fast forward velocities; everything else
  (rewards, events, the gentle curriculum ramp from slow speeds) is left
  identical to the flat task so it still trains from scratch.
  """
  cfg = booster_t1_flat_env_cfg(play=play)

  # Raise the top forward speed. The command curriculum still starts slow, so
  # the policy bootstraps from walking before being asked to sprint.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-2.0, 4.0)

  if not play:
    # Same staged ramp as the flat task, with one extra stage that pushes the
    # top speed from 3.0 to 4.0 m/s once the earlier stages are mastered.
    command_vel = cfg.curriculum["command_vel"]
    command_vel.params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
      {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
      {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
      {"step": 15000 * 24, "lin_vel_x": (-2.0, 4.0)},
    ]
  else:
    # Play at fast forward speeds.
    twist_cmd.ranges.lin_vel_x = (-1.5, 4.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
