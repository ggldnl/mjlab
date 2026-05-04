"""Unitree Crawler environment configurations."""

from mjlab.asset_zoo.robots import (
    CRAWLER_FOOT_SITE_NAMES,
    CRAWLER_FOOT_GEOM_NAMES,
    CRAWLER_ACTION_OFFSET,
    CRAWLER_ACTION_SCALE,
    CRAWLER_BASE_NAME,
    IMU,
    get_crawler_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    BuiltinSensorCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg, target_height
from mjlab.tasks.velocity.mdp.height_command import UniformHeightCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import math


SIM_DT = 0.001
DECIMATION = 10 # control period = 10 ms


def crawler_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Crawler rough terrain velocity configuration."""
    cfg = make_velocity_env_cfg()

    # impratio + elliptic cone give stable, physically meaningful foot forces
    # even at small mass scales.
    cfg.sim.mujoco.ccd_iterations = 200
    cfg.sim.mujoco.impratio = 10
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.mujoco.iterations = 20
    cfg.sim.mujoco.ls_iterations = 30
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 200
    cfg.sim.dt = SIM_DT
    cfg.decimation = DECIMATION

    cfg.scene.entities = {"robot": get_crawler_robot_cfg()}
    cfg.viewer.body_name = CRAWLER_BASE_NAME
    cfg.viewer.show_sites = False

    # Sensors

    # Patch sensors inherited from make_velocity_env_cfg that are left with
    # blank/empty fields expecting each robot config to fill them in.
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            sensor.frame = ObjRef(type="body", name=CRAWLER_BASE_NAME, entity="robot")
        elif sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            # Wire to the crawler's four foot sites with a small ring pattern
            # to measure ground clearance under each foot (drives foot_clearance reward)
            sensor.frame = tuple(
                ObjRef(type="site", name=s, entity="robot")
                for s in CRAWLER_FOOT_SITE_NAMES
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.008, num_samples=6)

    # Detect when each foot/tibia subtree touches terrain
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^leg_[1-4]_foot",  # one per leg - matches all 4
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    _LEG_SEGMENT_GEOM_NAMES = tuple(
        f"leg_{i}_{seg}_collision"
        for i in (1, 2, 3, 4)
        for seg in ("coxa", "femur", "tibia")
    )
    legs_ground_cfg = ContactSensorCfg(
        name="legs_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=_LEG_SEGMENT_GEOM_NAMES,
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    # Detect self collision
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(
            mode="subtree",
            pattern=CRAWLER_BASE_NAME,
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="subtree",
            pattern=CRAWLER_BASE_NAME,
            entity="robot",
        ),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    # Measure robot angular momentum
    root_angmom = BuiltinSensorCfg(
        name="root_angmom",
        sensor_type="subtreeangmom",
        obj=ObjRef(type="body", name=CRAWLER_BASE_NAME, entity="robot"),
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        legs_ground_cfg,
        self_collision_cfg,
        root_angmom,
        *IMU,
    )

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    # Actions

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = CRAWLER_ACTION_SCALE
    joint_pos_action.offset = CRAWLER_ACTION_OFFSET

    # 40% of envs get a zero-velocity command so the robot first learns to
    # stand still stably before being pushed to walk.
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.rel_standing_envs = 0.1

    # Vertical offset at which the velocity command arrow is rendered in the viewer.
    # Nothing to do with training.
    twist_cmd.viz.z_offset = 0.05

    cfg.commands["target_height"] = UniformHeightCommandCfg(
        entity_name="robot",
        resampling_time_range=(4.0, 8.0),
        ranges=UniformHeightCommandCfg.Ranges(height=(0.03, 0.08)),
    )

    # Events

    # Spawn scatter: the terrain spawner places the robot
    # at the correct absolute height; this is a perturbation on top of that.
    cfg.events["reset_base"].params["pose_range"].update({
        "x": (-0.1, 0.1),
        "y": (-0.1, 0.1),
        "z": (0.006, 0.018),
    })

    # Push disturbance: start late so the robot has a chance to learn walking
    # before being disturbed
    cfg.events["push_robot"] = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={
            "velocity_range": {
                "x": (-0.04, 0.04),
                "y": (-0.04, 0.04),
                "z": (-0.01, 0.02),
                "roll": (-0.08, 0.08),
                "pitch": (-0.08, 0.08),
                "yaw": (-0.12, 0.12),
            },
        },
    )

    # CoM offset: +/-8/10 mm is a large fraction of the crawler's body size;
    # +/-8/10 mm is still meaningful DR without shifting CoM outside the
    # support polygon at rest.
    cfg.events["base_com"].params["asset_cfg"].body_names = (CRAWLER_BASE_NAME,)
    cfg.events["base_com"].params["ranges"].update({
        0: (-0.008, 0.008),
        1: (-0.008, 0.008),
        2: (-0.010, 0.010),
    })

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = CRAWLER_FOOT_GEOM_NAMES

    """
    # Target height randomization
    # At each episode reset, every robot is assigned a fixed target body height
    # sampled uniformly from the range. The value is stored on the env
    # and read by the track_target_height reward.
    cfg.events["randomize_target_height"] = EventTermCfg(
        func=mdp.randomize_target_height,
        mode="reset",
        params={"height_range": (0.02, 0.04)},
    )
    """

    # Velocity tracking rewards (should be the primary learning signal)

    cfg.rewards["track_linear_velocity"].weight = 3.0
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.05)

    cfg.rewards["track_angular_velocity"].weight = 1.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.05)

    # Pose regularization

    # Rationale for std values:
    # Standard deviation around the default joint pose. Smaller std ->
    # stronger pull toward the default. The crawler joints are:
    #   coxa   (hip abduction / rotation, lateral sway)
    #   femur  (hip flexion / main swing joint)
    #   tibia  (knee extension, foot clearance)
    #
    # Philosophy:
    # - femur gets the loosest std: it must swing freely during stride.
    # - tibia is moderately loose: needs to lift for clearance but also
    #   provide ground contact stability.
    # - coxa is tightest: excessive lateral sway destabilizes the trot
    #   and wastes energy.
    # Running values are ~1.5–2x walking to allow larger motion range.
    cfg.rewards["pose"].weight = 0.8
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        r"base_leg_[1-4]_coxa": 0.30,
        r"leg_[1-4]_coxa_leg_[1-4]_femur": 0.50,
        r"leg_[1-4]_femur_leg_[1-4]_tibia": 0.25,
    }
    cfg.rewards["pose"].params["std_running"] = {
        r"base_leg_[1-4]_coxa": 0.30,
        r"leg_[1-4]_coxa_leg_[1-4]_femur": 0.50,
        r"leg_[1-4]_femur_leg_[1-4]_tibia": 0.40,
    }

    # Inherited weight=−1.0 from base. With action_scale=0.30 rad (fixed) the
    # robot stays well inside soft limits, so this rarely fires. Reduce anyway
    # to prevent it from dominating during any early joint limit contacts.
    cfg.rewards["dof_pos_limits"].weight = -0.1

    # Foot clearance targets scaled to actual robot geometry
    cfg.rewards["foot_clearance"].weight = -0.5
    cfg.rewards["foot_clearance"].params["target_height"] = 0.005

    cfg.rewards["foot_swing_height"].weight = -0.1
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.005

    # Slower priority than velocity tracking
    cfg.rewards["track_target_height"] = RewardTermCfg(
        func=mdp.track_target_height,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=(CRAWLER_BASE_NAME,)),
            "command_name": "target_height",
            "std": 0.01,
        },
    )

    # Other rewards

    # Upright and angular velocity rewards
    cfg.rewards["upright"].params["asset_cfg"].body_names = (CRAWLER_BASE_NAME,)
    cfg.rewards["upright"].params["std"] = math.sqrt(0.1)
    cfg.rewards["upright"].weight = 0.1

    # Body angular velocity penalty
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (CRAWLER_BASE_NAME,)
    cfg.rewards["body_ang_vel"].weight = -0.05

    # Penalizing angular momentum before the robot can walk
    # creates conflicting gradients with gait exploration.
    cfg.rewards["angular_momentum"].weight = -0.05

    # Over-penalizing action rate early in training prevents
    # the robot from discovering any motion at all.
    cfg.rewards["action_rate_l2"].weight = -0.05

    cfg.rewards["dof_vel_l2"] = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Small positive bonus for air_time.
    cfg.rewards["air_time"].weight = 0.1
    cfg.rewards["air_time"].params["command_threshold"] = 0.05
    cfg.rewards["air_time"].params["threshold_min"] = 0.015

    # Foot sensors
    # foot_clearance and foot_slip both need the four foot sites.
    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = CRAWLER_FOOT_SITE_NAMES

    # During early training the robot inevitably self-collides while exploring;
    # a higher penalty per step makes the policy learn to freeze rather than move.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.05,
        params={
            "sensor_name": self_collision_cfg.name,
            "force_threshold": 0.03,
        },
    )

    cfg.rewards["legs_ground_collision"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={
            "sensor_name": legs_ground_cfg.name,
            "force_threshold": 0.01,  # N - low threshold: even grazing costs
        },
    )

    # Observations

    # Defaults from make_velocity_env_cfg are tuned for a full-size quadruped.
    _actor_noise_overrides: dict[str, Unoise] = {
        "base_lin_vel": Unoise(n_min=-0.05, n_max=0.05),
        "base_ang_vel": Unoise(n_min=-0.05, n_max=0.05),
        "joint_vel": Unoise(n_min=-0.30, n_max=0.30),
    }
    for term_name, noise in _actor_noise_overrides.items():
        term = cfg.observations["actor"].terms.get(term_name)
        if term is not None:
            term.noise = noise

    # The policy is penalized by track_target_height but has no way to observe
    # what the sampled target is. Without this the reward is an unlearnable
    # stochastic signal.
    for obs_group_name in ("actor", "critic"):
        cfg.observations[obs_group_name].terms["target_height"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "target_height"},
        )

    # Terminations

    # The large termination penalty combined with a per-step alive bonus trained
    # the policy to prioritize survival (staying motionless) over locomotion.
    # The time_out termination already provides the implicit cost of episode
    # termination through the value function.
    """
    cfg.rewards["fell_over_penalty"] = RewardTermCfg(
        func=mdp.is_terminated,
        weight=-10.0,
    )

    cfg.rewards["alive"] = RewardTermCfg(
        func=mdp.is_alive,
        weight=2.0,  # +2.0/step * N_steps >> velocity reward from one fall
    )
    """

    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(35.0)

    # Curriculum

    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=mdp.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                # Stage 0: learn balance and minimal motion.
                {"step": 0,
                 "lin_vel_x": (-0.2, 0.2), "lin_vel_y": (-0.1, 0.1),
                 "ang_vel_z": (-0.15, 0.15)},
                # Stage 1: introduce walking velocity.
                {"step": 500 * 24,
                 "lin_vel_x": (-0.4, 0.5), "lin_vel_y": (-0.25, 0.25),
                 "ang_vel_z": (-0.25, 0.25)},
                # Stage 2: moderate locomotion.
                {"step": 1500 * 24,
                 "lin_vel_x": (-0.7, 0.9), "lin_vel_y": (-0.5, 0.5),
                 "ang_vel_z": (-0.40, 0.40)},
                # Stage 3: full range.
                {"step": 3000 * 24,
                 "lin_vel_x": (-1.0, 1.5), "lin_vel_y": (-0.7, 0.7),
                 "ang_vel_z": (-0.60, 0.60)},
            ],
        },
    )

    # Apply play mode overrides.
    if play:
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


def crawler_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Crawler flat terrain velocity configuration."""
    cfg = crawler_rough_env_cfg(play=play)

    # Flat terrain has fewer contacts, we can relax limits.
    cfg.sim.njmax = 500
    cfg.sim.mujoco.ccd_iterations = 200
    cfg.sim.contact_sensor_maxmatch = 50
    cfg.sim.nconmax = 200
    cfg.sim.dt = SIM_DT
    cfg.decimation = DECIMATION

    # Switch to flat terrain.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Drop raycast sensors: ground is flat at z=0 so terrain height and foot
    # clearance scans carry no information and waste compute.
    cfg.scene.sensors = tuple(
        s for s in cfg.scene.sensors
        if s.name not in ("terrain_scan", "foot_height_scan")
    )

    # Drop observations and rewards that depend on the removed sensors.
    # - height_scan reads terrain_scan
    # - foot_height, foot_clearance, foot_swing_height all read foot_height_scan
    for obs_group in ("actor", "critic"):
        cfg.observations[obs_group].terms.pop("height_scan", None)
        cfg.observations[obs_group].terms.pop("foot_height", None)
    cfg.rewards.pop("foot_clearance", None)
    cfg.rewards.pop("foot_swing_height", None)

    # Curriculum and termination.
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum.pop("terrain_levels", None)

    if play:
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
        twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

    return cfg