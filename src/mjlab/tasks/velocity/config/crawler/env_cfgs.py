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
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


SIM_DT = 0.001
DECIMATION = 10 # control period = 10 ms > T_settle (depends on the actuators)


def crawler_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Crawler rough terrain velocity configuration."""
    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 200
    cfg.sim.dt = SIM_DT
    cfg.decimation = DECIMATION

    cfg.scene.entities = {"robot": get_crawler_robot_cfg()}

    # Patch sensors inherited from make_velocity_env_cfg that are left with
    # blank/empty fields expecting each robot config to fill them in.
    # - terrain_scan: frame.name='' resolves to 'robot/' at scene init -> crash
    # - foot_height_scan: frame=() means no sites attached -> sensor never fires
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
            pattern=r"^leg_[1-4]_foot",  # one per leg — matches all 4
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(
            mode="subtree",
            pattern=CRAWLER_BASE_NAME,  # "base"
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

    root_angmom = BuiltinSensorCfg(
        name="root_angmom",
        sensor_type="subtreeangmom",
        obj=ObjRef(type="body", name=CRAWLER_BASE_NAME, entity="robot"),
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
        root_angmom,
        *IMU,
    )

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = CRAWLER_ACTION_SCALE
    joint_pos_action.offset = CRAWLER_ACTION_OFFSET

    cfg.viewer.body_name = CRAWLER_BASE_NAME

    # The crawler rides ~0.1–0.15 m off the ground
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 0.05

    # Randomize friction on the foot geoms only (the contact surfaces).
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = CRAWLER_FOOT_GEOM_NAMES

    # Push the center-of-mass randomization onto the base body.
    cfg.events["base_com"].params["asset_cfg"].body_names = (CRAWLER_BASE_NAME,)

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
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        r"base_leg_[1-4]_coxa": 0.15,  # lateral / rotational hip, keep tight
        r"leg_[1-4]_coxa_leg_[1-4]_femur": 0.30,  # main swing, needs freedom
        r"leg_[1-4]_femur_leg_[1-4]_tibia": 0.25,  # moderate
    }
    cfg.rewards["pose"].params["std_running"] = {
        r"base_leg_[1-4]_coxa": 0.20,
        r"leg_[1-4]_coxa_leg_[1-4]_femur": 0.50,
        r"leg_[1-4]_femur_leg_[1-4]_tibia": 0.40,
    }

    # Upright and angular velocity rewards
    cfg.rewards["upright"].params["asset_cfg"].body_names = (CRAWLER_BASE_NAME,)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (CRAWLER_BASE_NAME,)

    # foot_clearance and foot_slip both need the four foot sites.
    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = CRAWLER_FOOT_SITE_NAMES

    # The crawler is lower to the ground so angular momentum matters less;
    # air time is left at 0 until a gait style is chosen (trot, crawl, etc.).
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["air_time"].weight = 0.0

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={
            "sensor_name": self_collision_cfg.name,
            "force_threshold": 0.5,  # N
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
    cfg.sim.mujoco.ccd_iterations = 50
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