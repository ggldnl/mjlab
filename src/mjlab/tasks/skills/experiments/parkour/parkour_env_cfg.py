"""One env cfg factory, three goal-conditioned primitive motion skills.

walk, run and jump are the same environment pointed at a different clip folder. That is
the whole difference: `data/lafan1_g1/clips/<skill>/` says what the motion looks like and
what goals it realizes, and everything else -- observation space, action space, reward
structure, terminations -- is shared. It has to be shared, because the bridging
experiment evaluates all three frozen skills against one env, and a skill whose
observation is a different width cannot be run there at all.

Each skill is a DeepMimic tracker with a goal (see mdp.py):

- the tracking half pays for reproducing the reference clip -- joint pose, joint
  velocity, end-effector placement, root height and orientation, root velocity, foot
  contact pattern. That is what makes the motion recognizably a walk, a run or a jump,
  without anybody writing down what those look like;
- the goal half pays for realizing what the clip realizes -- the displacement it covers,
  the heading it turns through, the apex it reaches. That is what makes the skill
  addressable afterwards.

Built on the G1 flat *tracking* env, whose robot, action space, domain randomization and
sensors are already right for this; the command, observations, rewards and terminations
are replaced wholesale.

The observation is deliberately free of global position and heading. The clips were
normalized to remove both (see dataset.py), the reference carries velocities in the
root's own yaw frame, and the goal is a displacement from wherever the episode started.
A policy trained here cannot tell which way it is facing, which is what lets the same
skill mean the same thing anywhere on the plane -- and, downstream, what lets a bridge be
compared against it without the comparison being dominated by where the robot happened to
be standing.
"""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.skills.experiments.parkour import mdp as parkour_mdp
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# Where dataset.py filed the reference clips: one folder per skill.
CLIP_ROOT = Path("data/lafan1_g1/clips")

# The three skills. Sprint is gone: one source clip, a speed band overlapping run's, and
# nothing the corridor asks of it that run does not already cover.
SKILL_NAMES = ("walk", "run", "jump")

# The bodies whose offsets the clips store, in the order dataset.py stored them.
END_EFFECTOR_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)

COMMAND_NAME = "motion"

# How far ahead the policy is shown the reference, in control steps (50 Hz, so 0.04 s,
# 0.10 s and 0.20 s). Seeing where the reference is going rather than only where it is
# now is what lets a motion be prepared instead of chased, and it is the single change
# that moves tracking quality the most on the fast behaviors: a jump's takeoff has to be
# set up before the frame that asks for it.
FUTURE_OFFSETS = (2, 5, 10)

# The episode cap, in training only. Clips are shorter than this and end the episode
# themselves (`motion_finished`), so this only catches an env that somehow outlives its
# reference. Play mode keeps the tracking env's effectively-infinite cap, so a played
# skill runs clip after clip until the viewer is closed.
EPISODE_LENGTH_S = 10.0


def _observation_terms(privileged: bool) -> dict[str, ObservationTermCfg]:
  """The observation, in one place so the actor and the critic cannot drift apart.

  The critic gets the same terms plus root linear velocity, which a real robot has no
  clean way to measure but a critic is free to use.
  """
  noise = (lambda n: n) if not privileged else (lambda _: None)

  terms: dict[str, ObservationTermCfg] = {
    # Proprioception: what the robot can actually feel.
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=noise(Unoise(n_min=-0.2, n_max=0.2)),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=noise(Unoise(n_min=-0.05, n_max=0.05)),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=noise(Unoise(n_min=-0.01, n_max=0.01)),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=noise(Unoise(n_min=-0.5, n_max=0.5)),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    # The reference: where the motion says the robot should be, now.
    "reference_joint_pos": ObservationTermCfg(
      func=parkour_mdp.reference_joint_pos_error,
      params={"command_name": COMMAND_NAME},
    ),
    "reference_joint_vel": ObservationTermCfg(
      func=parkour_mdp.reference_joint_vel_error,
      params={"command_name": COMMAND_NAME},
    ),
    "reference_root_pose": ObservationTermCfg(
      func=parkour_mdp.reference_root_pose,
      params={"command_name": COMMAND_NAME},
    ),
    "reference_root_velocity": ObservationTermCfg(
      func=parkour_mdp.reference_root_velocity,
      params={"command_name": COMMAND_NAME},
    ),
    "reference_contacts": ObservationTermCfg(
      func=parkour_mdp.reference_contacts,
      params={"command_name": COMMAND_NAME},
    ),
    # Where the reference is going next.
    "reference_future": ObservationTermCfg(
      func=parkour_mdp.reference_future_joint_pos,
      params={"command_name": COMMAND_NAME, "offsets": FUTURE_OFFSETS},
    ),
    # Where in the clip this is.
    "phase": ObservationTermCfg(
      func=parkour_mdp.motion_phase,
      params={"command_name": COMMAND_NAME},
    ),
    # The goal, and how much of it is left.
    "goal": ObservationTermCfg(
      func=parkour_mdp.goal_command,
      params={"command_name": COMMAND_NAME},
    ),
    "goal_remaining": ObservationTermCfg(
      func=parkour_mdp.goal_remaining,
      params={"command_name": COMMAND_NAME},
    ),
  }

  if privileged:
    terms["base_lin_vel"] = ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    )
  return terms


def parkour_skill_env_cfg(
  skill: str, clip_root: str | Path = CLIP_ROOT, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """The env for one skill: its clip folder, and nothing else per-skill."""
  if skill not in SKILL_NAMES:
    raise ValueError(f"Unknown skill '{skill}'; known: {SKILL_NAMES}.")

  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  clip_dir = str(Path(clip_root) / skill)

  ##
  # Sensors: the tracking env has self-collision; the contact reward needs the feet.
  ##

  feet_contact = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_contact,)

  ##
  # Command
  ##

  # Resampling is driven by the clip running out, not by a timer, so the interval is
  # effectively infinite: `motion_finished` ends the episode and the reset resamples.
  commands: dict[str, CommandTermCfg] = {
    COMMAND_NAME: parkour_mdp.SkillMotionCommandCfg(
      entity_name="robot",
      clip_dir=clip_dir,
      end_effector_names=END_EFFECTOR_NAMES,
      resampling_time_range=(1.0e9, 1.0e9),
      max_start_fraction=0.7,
      # Small on purpose. This is randomization so the policy does not memorize one
      # spawn pose, not a curriculum in recovering from a shove; the goal it is asked for
      # is the same either way.
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.05, 0.05),
        "pitch": (-0.05, 0.05),
      },
      velocity_range={
        "x": (-0.1, 0.1),
        "y": (-0.1, 0.1),
        "z": (-0.1, 0.1),
        "roll": (-0.15, 0.15),
        "pitch": (-0.15, 0.15),
        "yaw": (-0.15, 0.15),
      },
      joint_position_range=(-0.05, 0.05),
    )
  }
  cfg.commands = commands

  ##
  # Observations
  ##

  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=_observation_terms(privileged=False),
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=_observation_terms(privileged=True),
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Rewards
  ##

  # The tracking weights follow DeepMimic's ordering: pose dominates, end-effectors close
  # behind (a small hip error becomes a large foot error, and the foot is what shows),
  # root pose next, and joint velocity well down since it is noisy and matching it
  # exactly is neither achievable nor useful.
  #
  # The goal weights are deliberately not small. A goal reward that is a rounding error
  # next to the tracking stack is a goal the policy is free to ignore, and the skill then
  # looks conditioned while steering nowhere.
  rewards: dict[str, RewardTermCfg] = {
    "track_joint_pos": RewardTermCfg(
      func=parkour_mdp.track_joint_pos_exp,
      weight=2.0,
      params={"command_name": COMMAND_NAME, "std": math.sqrt(0.5)},
    ),
    "track_joint_vel": RewardTermCfg(
      func=parkour_mdp.track_joint_vel_exp,
      weight=0.2,
      params={"command_name": COMMAND_NAME, "std": 10.0},
    ),
    "track_end_effector": RewardTermCfg(
      func=parkour_mdp.track_end_effector_exp,
      weight=1.5,
      params={"command_name": COMMAND_NAME, "std": 0.15},
    ),
    "track_root_pose": RewardTermCfg(
      func=parkour_mdp.track_root_pose_exp,
      weight=1.0,
      params={"command_name": COMMAND_NAME, "std": 0.3},
    ),
    "track_root_velocity": RewardTermCfg(
      func=parkour_mdp.track_root_velocity_exp,
      weight=0.6,
      params={"command_name": COMMAND_NAME, "std": 1.5},
    ),
    "track_foot_contact": RewardTermCfg(
      func=parkour_mdp.track_foot_contact,
      weight=0.5,
      params={
        "command_name": COMMAND_NAME,
        "sensor_name": "feet_ground_contact",
        "force_threshold": 1.0,
      },
    ),
    "goal_displacement": RewardTermCfg(
      func=parkour_mdp.goal_displacement_exp,
      weight=1.0,
      params={"command_name": COMMAND_NAME, "std": 0.3},
    ),
    # Regularization: what keeps the robot from hurting itself getting there.
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "joint_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-5.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-2.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }

  # Per-skill goal terms, wired only where the skill's clips actually carry the channel.
  # `SkillMotionCommand` reports zero error for a channel its library does not have, so a
  # mis-wired term would score a constant 1.0 and quietly do nothing.
  if skill in ("walk", "run"):
    rewards["goal_heading"] = RewardTermCfg(
      func=parkour_mdp.goal_heading_exp,
      weight=0.5,
      params={"command_name": COMMAND_NAME, "std": 0.5},
    )
  if skill == "jump":
    rewards["goal_apex"] = RewardTermCfg(
      func=parkour_mdp.goal_apex_exp,
      weight=1.5,
      params={"command_name": COMMAND_NAME, "std": 0.05},
    )

  cfg.rewards = rewards

  ##
  # Terminations
  ##

  cfg.terminations = {
    # Reaching the end of the reference is success, so it is a time-out and bootstraps
    # rather than being scored as a failure.
    "motion_finished": TerminationTermCfg(
      func=parkour_mdp.motion_finished,
      params={"command_name": COMMAND_NAME},
      time_out=True,
    ),
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=parkour_mdp.fell_over,
      params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    ),
    # Loose enough that a jump's flight phase is not mistaken for losing the reference,
    # tight enough that a robot which has stopped tracking is not left running.
    "lost_reference": TerminationTermCfg(
      func=parkour_mdp.bad_motion_tracking,
      params={"command_name": COMMAND_NAME, "threshold": 0.5},
    ),
    "bad_root_height": TerminationTermCfg(
      func=parkour_mdp.bad_root_height,
      params={"command_name": COMMAND_NAME, "threshold": 0.35},
    ),
  }

  cfg.curriculum.pop("command_vel", None)

  if not play:
    cfg.episode_length_s = EPISODE_LENGTH_S

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    command = cfg.commands[COMMAND_NAME]
    assert isinstance(command, parkour_mdp.SkillMotionCommandCfg)
    command.pose_range = {}
    command.velocity_range = {}
    command.joint_position_range = (0.0, 0.0)
    # Start at the top of a clip so a played skill shows the whole motion.
    command.max_start_fraction = 0.0

  return cfg
