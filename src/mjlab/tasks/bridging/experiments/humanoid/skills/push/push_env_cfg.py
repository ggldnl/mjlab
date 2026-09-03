"""The push environment: mjlab's G1 velocity task with a 1 m crate in front of the robot.

Reward steps:

    floor        everything mjlab's flat G1 task already pays: velocity tracking, upright,
                 posture, foot clearance and slip, the regularizers
    step one     reach_box. Dense, a kernel on each hand's distance to the crate surface,
                 averaged over the two, reaching one when a palm is against it. Ungated, so
                 it also pulls a hand back if contact is lost
    step two     hands_on_box. Fraction of the hands on the crate, paid every step contact
                 holds
    step three   push_tracking. Fraction of the commanded velocity the crate is carrying,
                 paid only while a hand is on it
    penalty      body_contact. Flat, while anything that is not a hand touches the crate

The lower two steps agree with learning to walk from iteration one, since they ask the
robot to go somewhere and put its hands on it, which costs a walking policy nothing.
A large weight on the crate velocity is a reward a robot that cannot yet walk has
no way to earn, and the cheapest way to look for it is to throw itself at the crate.
The penalty is live from iteration one for the same reason in reverse: a body
push that is ever profitable is a habit that has to be unlearned later.

The default pose puts the arms out front. The velocity task's posture reward holds every
joint near its default, so the way to hold the arms forward is to make forward the default
rather than fight that reward with a second one. It is also the reset pose and the zero
point of the action, so the policy begins every episode already reaching.

What to watch:

    Episode_Metrics/push_hands_contact_rate  fraction of the hands on the crate
    Episode_Metrics/push_body_contact_rate   fraction of the episode touching it illegally
    Episode_Metrics/push_box_displacement    furthest the crate got, in metres
    Episode_Metrics/push_box_speed           how fast it was moving
    Episode/rew_push_tracking                the goal term itself
    Curriculum/push_tracking                 whether that weight is on yet

push_hands_contact_rate separates learning to push from learning to walk around a crate.
Near zero while the velocity tracking reward climbs means the policy found the detour, and
the fix is a larger reach_box weight, not a larger push_tracking one. Climbing alongside
push_body_contact_rate means the policy is paying the penalty, and the fix is a larger
body_contact weight.

Run

    uv run train Mjlab-G1-Push --env.scene.num-envs 4096
    uv run play Mjlab-G1-Push
"""

from __future__ import annotations

import math
import re
from dataclasses import replace

from mjlab.asset_zoo.objects.box import get_box_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.bridging.experiments.humanoid.skills.push import mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

##
# The stance the robot pushes from.
##

# Arm joint angles that replace the walking defaults, in radians.
#
# These are the pose reward's target, the reset pose, and the zero point of the joint
# position action all at once, which is why they are the lever that makes a G1 hold its arms
# forward rather than a fourth reward term fighting a tuned third one.
#
# Read off the G1's forward kinematics, not guessed. Shoulder pitch -1.0 swings the upper
# arm forward and level, elbow 1.4 folds the forearm back out ahead of it, and a small
# shoulder roll spreads the hands to shoulder width. The result:
#
#     hands 0.38 m in front of the pelvis, 0.30 m apart, 0.87 m off the ground
#
# which is a metre crate's upper face. Everything else stays at the walking keyframe
PUSH_ARM_POSE: dict[str, float] = {
  "left_shoulder_pitch_joint": -1.0,
  "right_shoulder_pitch_joint": -1.0,
  "left_shoulder_roll_joint": 0.15,
  "right_shoulder_roll_joint": -0.15,
  "left_elbow_joint": 1.4,
  "right_elbow_joint": 1.4,
}

# Which of the walking keyframe's entries this pose replaces. Named by joint rather than
# merged over, because the keyframe mixes regexes with exact names and a merge only
# overrides entries whose keys are spelled the same way. Dropping in
# .*_shoulder_pitch_joint next to the keyframe's own left_shoulder_pitch_joint leaves the
# arms exactly where they were, silently and with no error to read
_ARM_JOINTS = re.compile(r"shoulder|elbow")

# Posture tolerances for the arm joints, replacing the walking task's. Wider, because those
# were tuned for arms that only swing: at 0.15 rad the posture reward charges most of its
# weight for the third of a radian of shoulder travel separating a hand resting on a crate
# from a hand pressing into one. Wide enough to reach and lean, narrow enough that the arms
# come back to the front when there is nothing to push
PUSH_ARM_STD: dict[str, float] = {
  r".*shoulder_pitch.*": 0.4,
  r".*shoulder_roll.*": 0.25,
  r".*shoulder_yaw.*": 0.2,
  r".*elbow.*": 0.4,
  r".*wrist.*": 0.3,
}

##
# The crate.
##

# Nominal mass of the one metre cube, in kg.
#
# The size makes it a big object, the mass makes it a learnable one, and the two are set
# independently on purpose. At the friction below, shifting the crate costs 0.4 of its
# weight in horizontal force:
#
#     12 kg  ->  about 47 N, roughly an eighth of what a 35 kg G1 weighs
#
# which is a firm push through two arms rather than a shove. This is the one number to raise
# for a heavier task. Past about 30 kg it asks for more traction than the feet have on a bad
# friction draw
BOX_MASS = 12.0

# Sliding friction of the crate against the ground, and the priority that makes it stick.
#
# A crate pushed high up tips instead of sliding, once the push force passes
#
#     weight * (half its width) / (push height)
#
# At the ground's own friction of 1.0 that threshold is below the force needed to slide it
# at all, so a metre cube pushed at 0.87 m topples every time and the task becomes tumbling.
# At 0.4 it slides at about 47 N and tips at about 68 N, so pushing wins.
#
# The priority is what makes the number mean anything. MuJoCo takes the elementwise maximum
# of two geoms' friction unless one declares a higher priority, and the terrain does not, so
# at the default priority of 0 this crate would slide on the terrain's 1.0 however low its
# own friction was set
BOX_FRICTION = 0.4
BOX_PRIORITY = 1

# Where the crate spawns, relative to the robot and in its heading frame. Forward is
# measured centre to centre, so the near face sits half a metre closer than the number says
# and the hands, held out front by the push pose, start roughly 0.2 to 0.7 m from it. Far
# enough that getting there is walking, near enough that it happens in a second or two
BOX_FORWARD_RANGE = (1.1, 1.6)
BOX_LATERAL_RANGE = (-0.25, 0.25)

# Yaw scatter of the crate itself. Enough that the policy meets a face at an angle as often
# as square on, small enough that it never has to push a corner
BOX_YAW_RANGE = (math.radians(-20.0), math.radians(20.0))

##
# What the push is asked for.
##

# Environment steps per command curriculum stage. common_step_counter advances once per env
# step, so at the default 24 steps per env per iteration this is about 2000 iterations
_SPEED_STAGE = 2000 * 24

# Strictly forward, never zero. Asking the robot to reverse away from a crate is not a
# push, and an env commanded to stand still collects the top rung for free the moment it
# stands next to a stationary crate, since a crate matching a zero command is one doing
# nothing
SPEED_STAGES: list[mdp.VelocityStage] = [
  {
    "step": 0,
    "lin_vel_x": (0.3, 0.6),
    "lin_vel_y": (-0.2, 0.2),
    "ang_vel_z": (-0.3, 0.3),
  },
  {
    "step": _SPEED_STAGE,
    "lin_vel_x": (0.3, 0.9),
    "lin_vel_y": (-0.2, 0.2),
    "ang_vel_z": (-0.3, 0.3),
  },
  {
    "step": 2 * _SPEED_STAGE,
    "lin_vel_x": (0.3, 1.2),
    "lin_vel_y": (-0.2, 0.2),
    "ang_vel_z": (-0.3, 0.3),
  },
]

##
# Curriculum.
##

# Environment steps per stage of the push weight ramp: off, then half strength, then full.
# About 500 iterations a stage, enough for a G1 starting in its own stance keyframe to be
# walking before the crate's velocity starts paying
PUSH_STAGE = 500 * 24

# Final weights of the ladder. Declared here rather than inline because the curriculum ramps
# to exactly the third one, and the two drifting apart would silently leave the task
# training at a weight nobody chose.
#
# The penalty is set against the contact rung, not the goal. A policy leaning its chest on
# the crate still collects the contact rung with its palms flat on the same face, so what it
# chooses between is
#
#     hands-only push   W_HANDS + W_PUSH
#     body push         W_HANDS + W_PUSH - W_BODY
#
# At twice the contact rung the body route is worse by a clear margin at every stage of the
# curriculum, including the first, where the goal term is still switched off
W_REACH = 1.0
W_HANDS = 1.0
W_PUSH = 3.0
W_BODY = -2.0


def g1_push_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the push environment.

  Args:
    play: Drop the observation noise and the curriculum, leave the reward weights at
      their final values, and fix the command range at the top of the speed schedule so a
      trained policy is watched at the speed it was trained up to. Inherited from the
      flat velocity task, which also stops the clock and stops shoving the robot.
  """
  cfg = unitree_g1_flat_env_cfg(play=play)

  ##
  # Scene
  ##

  # Arms forward, everything else as the walking task left it. Written as a replace on the
  # flat task's keyframe rather than a new one, so a change to the G1's stance reaches this
  # task without being copied into it
  robot_cfg = cfg.scene.entities["robot"]
  base_pose = robot_cfg.init_state.joint_pos or {}
  legs = {k: v for k, v in base_pose.items() if not _ARM_JOINTS.search(k)}
  robot_cfg.init_state = replace(
    robot_cfg.init_state, joint_pos={**legs, **PUSH_ARM_POSE}
  )

  cfg.scene.entities["box"] = get_box_cfg(
    half_size=mdp.BOX_HALF_SIZE,
    # Overwritten by the reset event on the first step. Matters only to a raw viewer that
    # never resets
    init_x=BOX_FORWARD_RANGE[0],
    mass=BOX_MASS,
    friction=BOX_FRICTION,
    priority=BOX_PRIORITY,
    color=(0.55, 0.4, 0.25, 1.0),
  )

  # The two hand spheres against the crate. Geom primaries rather than a subtree, because
  # the pattern has to name exactly the two domes the asset seats where the G1's rubber hands
  # would be, and a subtree would drag the forearm in with them
  hands_box_cfg = ContactSensorCfg(
    name=mdp.HANDS_BOX_SENSOR,
    primary=ContactMatch(mode="geom", pattern=mdp.HAND_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="box_collision", entity="box"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  # The whole robot against the crate, hands included. Kept alongside the sensor above so an
  # illegal touch is the difference between the two contact counts. See mdp.body_on_box for
  # why, rather than naming the thirty collision geoms that are not hands
  robot_box_cfg = ContactSensorCfg(
    name=mdp.ROBOT_BOX_SENSOR,
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="box_collision", entity="box"),
    fields=("found",),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (hands_box_cfg, robot_box_cfg)

  # The crate brings its own contacts: four corners on the ground plus however many the
  # robot makes against it, on top of the fourteen foot capsules the G1 already puts down.
  # The failure when this is too small is quiet. MuJoCo Warp prints "nefc overflow" and then
  # drops the constraints past the limit, so the run keeps going while the physics stops
  # being right. Grep a training log for it before trusting a result
  cfg.sim.njmax = 400
  cfg.sim.contact_sensor_maxmatch = 128

  ##
  # Observations
  ##

  # The crate and the hands, in the robot's own heading frame. Without these the policy is
  # asked to track a velocity while something invisible blocks it
  box_terms = {
    "box_pos": ObservationTermCfg(
      func=mdp.box_pos_b, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    "box_vel": ObservationTermCfg(
      func=mdp.box_vel_b, noise=Unoise(n_min=-0.1, n_max=0.1)
    ),
    "hand_box_offset": ObservationTermCfg(
      func=mdp.hand_box_offset_b,
      params={"hands_cfg": mdp.hands_cfg()},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "hands_contact": ObservationTermCfg(func=mdp.hands_contact),
  }
  cfg.observations["actor"].terms.update(box_terms)
  cfg.observations["critic"].terms.update(
    {
      **{
        name: ObservationTermCfg(func=term.func, params=term.params)
        for name, term in box_terms.items()
      },
      # How hard the robot is leaning, standing in for the crate's mass. The reset
      # randomizes that and the actor is never told, so a critic without this values a
      # state it cannot see
      "hands_contact_force": ObservationTermCfg(func=mdp.hands_contact_force),
      # Whether the penalty is firing. A step change in the return the critic would
      # otherwise have to infer from limb and crate positions
      "body_contact": ObservationTermCfg(func=mdp.body_contact),
    }
  )

  ##
  # Commands
  ##

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  # Mostly straight ahead, never standing. Sidestepping and turning are what a policy
  # reaches for to get out from behind the crate, and every standing env is one not spent
  # pushing
  twist_cmd.rel_forward_envs = 0.6
  twist_cmd.rel_standing_envs = 0.0
  # No heading environments. A heading target is absolute and sampled over the full circle,
  # and so is the robot's yaw at reset, so a heading env can be asked to turn half a circle
  # away from the crate just placed in front of it. heading_command itself stays on: with no
  # env claiming a heading the target is sampled and ignored, whereas turning the flag off
  # changes what ranges.heading is allowed to be and has gone wrong here before
  twist_cmd.rel_heading_envs = 0.0
  # Held longer than the stock 3 to 8 s. A crate takes a second or two to get moving, and a
  # command resampling on that timescale means the policy spends its life starting pushes
  # and never finishing one
  twist_cmd.resampling_time_range = (6.0, 12.0)

  # The stage-zero range. The curriculum overwrites it from the first step, but setting it
  # here keeps the config honest about where it begins
  first = SPEED_STAGES[0]
  assert first["lin_vel_x"] is not None
  assert first["lin_vel_y"] is not None
  assert first["ang_vel_z"] is not None
  twist_cmd.ranges.lin_vel_x = first["lin_vel_x"]
  twist_cmd.ranges.lin_vel_y = first["lin_vel_y"]
  twist_cmd.ranges.ang_vel_z = first["ang_vel_z"]

  ##
  # Events
  ##

  # Appended, so they run after the robot is placed: the crate goes down relative to where
  # the robot actually ended up, and its spawn is recorded off that
  cfg.events["reset_box"] = EventTermCfg(
    func=mdp.reset_box_in_front,
    mode="reset",
    params={
      "forward_range": BOX_FORWARD_RANGE,
      "lateral_range": BOX_LATERAL_RANGE,
      "yaw_range": BOX_YAW_RANGE,
    },
  )
  # Mass and inertia together, which is what pseudo_inertia buys over body_mass: the latter
  # leaves the inertia tensor behind and gives a crate resisting translation and rotation by
  # different amounts. alpha is a log density scale, so this range is about 0.6 to 1.6 times
  # BOX_MASS, roughly 9 to 20 kg
  cfg.events["box_inertia"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("box", body_names=("box",)),
      "alpha_range": (-0.25, 0.25),
    },
  )

  ##
  # Rewards
  ##

  # Room to reach without the posture reward charging for it. Every other joint keeps the
  # walking task's numbers
  cfg.rewards["pose"].params["std_walking"].update(PUSH_ARM_STD)
  cfg.rewards["pose"].params["std_running"].update(PUSH_ARM_STD)

  cfg.rewards["reach_box"] = RewardTermCfg(
    func=mdp.reach_box,
    weight=W_REACH,
    params={
      # The hands start 0.2 to 0.7 m off the crate's face, so this puts the spawn partway
      # up the kernel rather than out on its flat tail
      "std": 0.5,
      "hands_cfg": mdp.hands_cfg(),
    },
  )
  cfg.rewards["hands_on_box"] = RewardTermCfg(
    func=mdp.hands_on_box_reward, weight=W_HANDS
  )
  cfg.rewards["body_contact"] = RewardTermCfg(
    func=mdp.body_contact_penalty, weight=W_BODY
  )
  cfg.rewards["push_tracking"] = RewardTermCfg(
    func=mdp.push_tracking,
    weight=W_PUSH,
    params={"command_name": mdp.COMMAND_NAME},
  )

  ##
  # Metrics
  ##

  cfg.metrics["push_hands_contact_rate"] = MetricsTermCfg(func=mdp.hands_contact_rate)
  cfg.metrics["push_body_contact_rate"] = MetricsTermCfg(func=mdp.body_contact_rate)
  cfg.metrics["push_box_speed"] = MetricsTermCfg(func=mdp.box_speed)
  # Furthest the crate got, not the average over a trajectory that started at zero
  cfg.metrics["push_box_displacement"] = MetricsTermCfg(
    func=mdp.box_displacement, reduce="max"
  )

  ##
  # Curriculum
  ##

  if not play:
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
      func=mdp.commands_vel,
      params={"command_name": mdp.COMMAND_NAME, "velocity_stages": SPEED_STAGES},
    )
    cfg.curriculum["push_tracking"] = CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "push_tracking",
        "weight_stages": mdp.ramp(W_PUSH, PUSH_STAGE),
      },
    )
  else:
    # Watch it at the speed it was trained to, not the speed it started at. The flat task
    # already cleared the curriculum and widened the command range for play, and both are
    # wrong here, so both are set again
    cfg.curriculum = {}
    last = SPEED_STAGES[-1]
    assert last["lin_vel_x"] is not None
    twist_cmd.ranges.lin_vel_x = last["lin_vel_x"]
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (-0.2, 0.2)

  return cfg
