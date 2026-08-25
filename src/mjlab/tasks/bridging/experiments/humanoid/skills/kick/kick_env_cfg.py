"""The kick environment: a standing G1, a football in reach, and a commanded launch.

Built from mjlab parts rather than wrapped around the velocity task, because almost
nothing carries over. There is no commanded twist, no terrain, and the thing being
tracked is not the robot at all but the ball. What is borrowed is the velocity task's
regularizers and its upright term, which are about keeping a G1 in one piece and are the
same problem whatever the robot is doing.

The reward set is a ladder with three rungs and a floor.

  Floor      alive, upright, posture, stay_put. Paid every step from the first
             iteration. This is the only thing that pays during the standing stage of
             the curriculum, and it stays the largest per step term afterwards, which is
             what keeps a policy from discovering that falling over ends the penalties.

  Rung one   approach_ball. Dense, and switched off once the ball has been touched, so
             it points at the ball rather than rewarding a foot resting against it.

  Rung two   ball_touched. Latched: touch the ball once and the floor rises for the rest
             of the episode. Contact is the discrete event between a policy that can
             balance and one that can score the kick, so it is worth paying for on its
             own.

  Rung three kick_quality. Latched on the fastest the ball has gone, scored against the
             commanded launch velocity. This is the task.

The penalties are deliberately small. On a 29 joint humanoid a penalty set that outweighs
the positive terms makes an immediate termination the highest return trajectory available,
because a failure bootstraps zero; the tell is mean episode length falling monotonically
from the first iteration.

Run it:

    uv run train Mjlab-Parkour-Kick --env.scene.num-envs 4096
    uv run play Mjlab-Parkour-Kick

What to watch, in order of how much it tells you:

    Metrics/kick/launch_rate     fraction of episodes where the ball was struck at all
    Metrics/kick/vel_error       commanded minus achieved launch velocity, in m/s
    Metrics/kick/heading_error   aim, in radians, over the episodes that launched
    Episode/rew_kick_quality     the goal term itself
    Curriculum/kick_quality      whether the kick weights have been turned on yet

launch_rate is the one that separates "learning to kick" from "learning to stand near a
ball". vel_error and heading_error together are the check that the conditioning is real
rather than one memorised kick: a policy that ignores the command still launches, but its
errors stay flat at the spread of the command range instead of falling.
"""

from __future__ import annotations

import math

from mjlab.asset_zoo.objects.ball import get_ball_cfg
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.kick import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

##
# Where the ball goes.
##

# Forward offset of the ball's centre from the kicking foot's site, in metres. The foot's
# collision geoms reach about 0.09 m ahead of that site and the ball's rear surface sits
# one radius behind its centre, so anything under about 0.20 m spawns the ball already
# touching the toe. The upper end is bounded by reach: a standing G1 gets its toe out to
# roughly 0.45 m from the pelvis, and the site is 0.04 m ahead of it to begin with.
BALL_FORWARD_RANGE = (0.24, 0.32)

# Lateral scatter about the kicking foot's own line. Small on purpose. This is the knob
# that widens the task to either foot: opened up to cover the other foot's y as well, the
# policy has to choose a leg, which is a materially harder problem than the one trained
# here.
BALL_LATERAL_RANGE = (-0.05, 0.05)

##
# What the kick is asked for.
##

# Launch speed range in m/s. The lower end is well clear of a nudge; the upper end is a
# guess at what a balanced G1 can do to a 0.425 kg ball without falling over, and is the
# first thing to lower if launch_rate climbs while vel_error refuses to fall.
COMMAND_SPEED_RANGE = (1.5, 4.5)

# Aim, in radians off the robot's heading at reset. Wide enough that a policy ignoring
# the command is visibly wrong in the heading_error metric, narrow enough to stay a kick
# rather than a pivot.
COMMAND_HEADING_RANGE = (math.radians(-25.0), math.radians(25.0))

##
# Posture tolerances, per joint.
##

# Lopsided by design. The kicking leg is given room to swing and everything else is held
# near the stance, which is how the term says "kick with the leg, not with the whole
# body" without needing to know which phase the episode is in. Every joint must match
# exactly one pattern here or the term raises at construction.
POSTURE_STD = {
  # The kicking leg. This is where the kick actually happens.
  r"right_hip_pitch.*": 1.0,
  r"right_hip_roll.*": 0.4,
  r"right_hip_yaw.*": 0.4,
  r"right_knee.*": 1.2,
  r"right_ankle_pitch.*": 0.6,
  r"right_ankle_roll.*": 0.2,
  # The support leg, held close: this is what the robot is standing on.
  r"left_hip_pitch.*": 0.3,
  r"left_hip_roll.*": 0.15,
  r"left_hip_yaw.*": 0.15,
  r"left_knee.*": 0.35,
  r"left_ankle_pitch.*": 0.25,
  r"left_ankle_roll.*": 0.1,
  # Waist, tight, because the torso leaning is how balance is lost.
  r"waist_yaw.*": 0.3,
  r"waist_roll.*": 0.1,
  r"waist_pitch.*": 0.2,
  # Arms, which counterbalance the swing and matter little otherwise.
  r".*shoulder_pitch.*": 0.4,
  r".*shoulder_roll.*": 0.3,
  r".*shoulder_yaw.*": 0.3,
  r".*elbow.*": 0.4,
  r".*wrist.*": 0.5,
}

##
# Curriculum.
##

# Steps per curriculum stage. common_step_counter advances once per environment step, so
# at the default 24 steps per environment per iteration this is about 300 iterations a
# stage: standing for the first 300, half strength kick terms to 600, full after that.
# Standing a G1 that starts in its own stance keyframe is close to free, so 300 is
# generous rather than tight.
KICK_STAGE = 300 * 24

# Final weights of the three kick terms. Declared here rather than inline because the
# curriculum has to ramp to exactly these, and the two drifting apart would silently
# leave the task training at a weight nobody chose.
W_APPROACH = 1.0
W_TOUCH = 1.0
W_QUALITY = 5.0


def g1_kick_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the kick environment.

  Args:
    play: Drop the observation noise and the curriculum, and leave the reward weights at
      their final values. The episode clock goes effectively infinite as it does in every
      other task, but what actually cycles an episode here is the ball leaving, and its
      threshold is brought in so that a struck ball reliably crosses it. A kick happens
      once per episode, so without that the viewer shows one kick and then a robot
      standing next to a ball it has already dealt with.
  """

  ##
  # Observations
  ##

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.3, n_max=0.3),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    # The goal, in the robot's own heading frame. Without this term the task is not
    # conditioned on anything and the policy can only learn one kick.
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": mdp.COMMAND_NAME}
    ),
    # The ball. Position is what the approach is aimed at; velocity is what tells the
    # policy the kick has happened and how it went.
    "ball_pos": ObservationTermCfg(
      func=mdp.ball_pos_b, noise=Unoise(n_min=-0.02, n_max=0.02)
    ),
    "ball_vel": ObservationTermCfg(
      func=mdp.ball_vel_b, noise=Unoise(n_min=-0.1, n_max=0.1)
    ),
    "kick_foot_pos": ObservationTermCfg(func=mdp.kick_foot_pos_b),
    # When the kick terms turn on, and whether contact has already happened. Both are
    # gates the reward already applies; putting them in the observation is what makes
    # them a phase the policy can act on rather than a reward that changes for no
    # visible reason.
    "stance_phase": ObservationTermCfg(func=mdp.stance_phase),
    "ball_contact": ObservationTermCfg(func=mdp.ball_contact),
  }

  critic_terms = {
    **actor_terms,
    # True joint angles rather than the encoder biased ones.
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    # The latched launch velocity. Every step after a kick is paid on this rather than on
    # anything currently visible, so a critic without it is valuing a state it cannot
    # see.
    "launch_velocity": ObservationTermCfg(func=mdp.launch_velocity_b),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=G1_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    mdp.COMMAND_NAME: mdp.KickCommandCfg(
      # One goal per episode. There is one kick in an episode, so a command that changed
      # halfway through would be asking the policy to un-kick the ball.
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      ranges=mdp.KickCommandCfg.Ranges(
        speed=COMMAND_SPEED_RANGE,
        heading=COMMAND_HEADING_RANGE,
      ),
    )
  }

  ##
  # Events
  ##

  # Order matters and is load bearing. The robot is placed first, the ball is then placed
  # relative to where the robot's foot actually ended up, and the phase tracker reads its
  # stay put anchor off both of them last.
  events: dict[str, EventTermCfg] = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        # Yaw is free: the ball is placed in the robot's own frame, so a random heading
        # costs nothing and stops the policy learning a world direction. Position jitter
        # is small, because the env origins already separate the worlds and the task is
        # about a fixed relationship between a robot and a ball.
        "pose_range": {"z": (0.0, 0.02), "yaw": (-3.14, 3.14)},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.03, 0.03),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "reset_ball": EventTermCfg(
      func=mdp.reset_ball_near_foot,
      mode="reset",
      params={
        "forward_range": BALL_FORWARD_RANGE,
        "lateral_range": BALL_LATERAL_RANGE,
      },
    ),
    "reset_phase": EventTermCfg(func=mdp.reset_kick_phase, mode="reset", params={}),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=r"^(left|right)_foot[1-7]_collision$"
        ),
        "operation": "abs",
        "ranges": (0.4, 1.2),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "operation": "add",
        "ranges": {0: (-0.02, 0.02), 1: (-0.02, 0.02), 2: (-0.03, 0.03)},
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards: dict[str, RewardTermCfg] = {
    # The floor. Together these are worth about four per step to a robot that is simply
    # standing, which is what the kick terms have to be measured against and what keeps
    # an early termination from being the cheapest trajectory on offer.
    "alive": RewardTermCfg(func=mdp.is_alive, weight=1.0),
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      },
    ),
    "posture": RewardTermCfg(
      func=mdp.posture,
      weight=1.0,
      params={
        "std": POSTURE_STD,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "stay_put": RewardTermCfg(func=mdp.stay_put, weight=1.0, params={"std": 0.3}),
    # The ladder. All three start at zero and are raised by the curriculum.
    "approach_ball": RewardTermCfg(
      func=mdp.approach_ball, weight=W_APPROACH, params={"std": 0.15}
    ),
    "ball_touched": RewardTermCfg(func=mdp.ball_touched, weight=W_TOUCH),
    "kick_quality": RewardTermCfg(
      func=mdp.kick_quality,
      weight=W_QUALITY,
      params={"std": 1.0, "command_name": mdp.COMMAND_NAME},
    ),
    # Kicking before the stance window closes earns nothing, and this makes it cost
    # something too. Kept small: it fires for as long as the ball keeps rolling, and a
    # penalty that can run up a large negative over the opening second is exactly how a
    # policy learns that falling over early is a good idea.
    "early_kick": RewardTermCfg(func=mdp.early_disturbance, weight=-0.5),
    # Regularizers. Small, for the reason in the module docstring.
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
    "joint_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_vel": RewardTermCfg(func=mdp.joint_vel_l2, weight=-1.0e-4),
    "joint_torques": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1.0e-6),
    "foot_slip": RewardTermCfg(
      func=mdp.planted_foot_slip,
      weight=-0.2,
      params={
        "sensor_name": mdp.FEET_GROUND_SENSOR,
        "asset_cfg": SceneEntityCfg(
          "robot", site_names=(mdp.SUPPORT_SITE, mdp.KICK_SITE)
        ),
      },
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
    ),
    "collapsed": TerminationTermCfg(
      func=mdp.root_height_below_minimum, params={"minimum_height": 0.4}
    ),
    # A ball that has left the scene ends the episode, and it must stay a time out. The
    # kick is already scored and latched by then, and calling it a failure would
    # bootstrap zero onto the end of the best rollouts the policy ever produces.
    #
    # The distance is deliberately further than a good kick reaches inside an episode,
    # so in training this is a safety net and nothing else. kick_quality is latched and
    # paid to the end of the episode, so a threshold the ball actually crosses would end
    # the best rollouts soonest and pay them for fewer steps than the weak ones. rsl_rl
    # bootstraps the value on a time out, which makes that neutral in principle, but it
    # costs nothing to not rely on the value function getting it right. Play mode brings
    # the threshold in, because there the ball leaving is what cycles the episode.
    "ball_gone": TerminationTermCfg(
      func=mdp.ball_out_of_range, params={"distance": 10.0}, time_out=True
    ),
  }

  ##
  # Curriculum
  ##

  # Standing first, literally: for the opening stage the kick terms are worth nothing, so
  # the only reward available is the floor, and the only way to collect it is to stay
  # upright. See mdp.ramp for why there is a half strength stage in the middle.
  curriculum: dict[str, CurriculumTermCfg] = {
    "approach_ball": CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "approach_ball",
        "weight_stages": mdp.ramp(W_APPROACH, KICK_STAGE),
      },
    ),
    "ball_touched": CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "ball_touched",
        "weight_stages": mdp.ramp(W_TOUCH, KICK_STAGE),
      },
    ),
    "kick_quality": CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "kick_quality",
        "weight_stages": mdp.ramp(W_QUALITY, KICK_STAGE),
      },
    ),
  }

  ##
  # Scene
  ##

  # What the kick is scored through. Primary is the whole subtree below the kicking
  # ankle, which is how all seven foot collision geoms are covered at once; a secondary
  # has to resolve to a single element, so it is the ball's own collision sphere.
  foot_ball_cfg = ContactSensorCfg(
    name=mdp.FOOT_BALL_SENSOR,
    primary=ContactMatch(
      mode="subtree", pattern="right_ankle_roll_link", entity="robot"
    ),
    secondary=ContactMatch(mode="geom", pattern="ball_collision", entity="ball"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  feet_ground_cfg = ContactSensorCfg(
    name=mdp.FEET_GROUND_SENSOR,
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
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
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  # The ball asset's defaults are already a size 5 football, so only its spawn is given
  # here, and even that is overwritten by the reset event on the first step. It matters
  # only to a raw viewer that never resets.
  scene = SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={
      "robot": get_g1_robot_cfg(),
      "ball": get_ball_cfg(pos=(0.3, -0.12, mdp.BALL_RADIUS)),
    },
    sensors=(foot_ball_cfg, feet_ground_cfg, self_collision_cfg),
    num_envs=1,
    env_spacing=3.0,
  )

  cfg = ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=3.0,
      elevation=-10.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      # Do not leave this on the heuristic. The G1 puts fourteen condim=3 foot capsules
      # on the ground, adds joint limit rows as the kicking leg reaches its stops, and
      # the ball brings its own contacts, and the heuristic allocates far too few. The
      # failure is quiet: MuJoCo Warp prints "nefc overflow" and then simply drops the
      # constraints past the limit, so the run keeps going while the physics silently
      # stops being right.
      #
      # Measured over a 420 iteration run at 2048 envs, the constraint count per world
      # ran to a median of 74 and a maximum of 201. This is that maximum with half again
      # of headroom, and it matches what the G1 flat velocity task allocates. A policy
      # that strikes the ball harder than that run managed may need more, so grep a
      # training log for "nefc overflow" before trusting a result.
      njmax=300,
      contact_sensor_maxmatch=64,
      mujoco=MujocoCfg(
        timestep=0.005, iterations=10, ls_iterations=20, ccd_iterations=50
      ),
    ),
    # 0.005 * 4 gives 50 Hz control, matching the other humanoid skills.
    decimation=4,
    # One second of stance, then four to get the foot to the ball, strike it and stay
    # standing afterwards. Long enough that falling over after a good kick is still
    # visibly worse than not falling over.
    episode_length_s=5.0,
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    # The curriculum only ever lowers these; without it they sit at the final weights
    # declared above, which is what should be watched.
    cfg.curriculum = {}
    # With the clock effectively stopped, the ball leaving is the only thing that ends an
    # episode, so it has to be reachable. Even the slowest command in range carries the
    # ball past three metres, while six is a distance a weak kick can fail to reach and
    # then the viewer sits on one episode forever.
    cfg.terminations["ball_gone"].params["distance"] = 3.0

  return cfg
