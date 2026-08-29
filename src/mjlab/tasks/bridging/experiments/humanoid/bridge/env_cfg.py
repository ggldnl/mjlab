"""The bridging task, as an ordinary mjlab RL environment.

Run:

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.skills
    uv run train Mjlab-G1-Bridge --env.scene.num-envs 4096
    uv run play Mjlab-G1-Bridge

Being ordinary is the point. A bridge is a policy driving a robot through a simulator, so
the failures that killed the kinematic attempts (a foot through the floor, a body hovering
where the recording jumped, a pose averaged out of two incompatible crossings) cannot be
expressed here at all.

Two observation groups, actor and critic, differing only in noise. One thing about them is
not standard for this repository:

  base_lin_vel is in the actor group, not just the critic's. The locomotion tasks privilege
  it because hardware cannot measure it. Here the bridge is asked to arrive carrying a
  particular momentum, so a policy told the target velocity but not its own is closing a
  gap it cannot see. The previous version came out at 0.41 root velocity error against a
  standing robot's 0.56. Estimating velocity from an acceleration history is a separate
  problem, for a student distilled from this teacher later.

Neither group carries anything about the middle of the window. There is no reference to
observe, which is what makes this a bridge and not a tracker.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.bridging.experiments.humanoid.bridge import mdp
from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.dataset import (
  DEFAULT_DATASET,
)
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

COMMAND = "bridge"
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_SITES = ("left_foot", "right_foot")
FEET_CONTACT = "feet_ground_contact"

MAX_DEADLINE_S = 4.0
"""Longest window the episode has to fit, in seconds.

Longer than `deadline_range` allows, because a pair that does not fit the deadline it drew
gets the deadline stretched rather than being thrown away, and a stretched window runs to
about three seconds. Only a backstop; `deadline_reached` is what ends a window."""


def bridge_env_cfg(
  play: bool = False,
  split: str = "train",
  dataset_path: Path = DEFAULT_DATASET,
  leaving: tuple[str, ...] | None = None,
  entering: tuple[str, ...] | None = None,
) -> ManagerBasedRlEnvCfg:
  """The bridging environment.

  `leaving` and `entering` restrict which skills the start and the target may come from.
  None means any skill in the dataset, which is the default: the goal is one bridge for the
  whole pool. Set them to study a single pair.
  """
  command = mdp.BridgeCommandCfg(
    entity_name="robot",
    dataset_path=dataset_path,
    split=split,
    leaving=leaving,
    entering=entering,
    # A window is drawn on reset, never mid-episode. One that changed halfway through
    # would be two questions scored as one
    resampling_time_range=(1.0e9, 1.0e9),
    start_noise=0.0 if play else 1.0,
    curriculum_steps=0 if play else 60_000,
    debug_vis=True,
  )

  proprioception = {
    "base_lin_vel": ObservationTermCfg(
      func=base_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
    ),
    "base_ang_vel": ObservationTermCfg(
      func=base_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
    ),
    "projected_gravity": ObservationTermCfg(
      func=base_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    "joint_pos": ObservationTermCfg(
      func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
    ),
    "joint_vel": ObservationTermCfg(
      func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "actions": ObservationTermCfg(func=base_mdp.last_action),
  }
  goal = {
    "command": ObservationTermCfg(
      func=base_mdp.generated_commands, params={"command_name": COMMAND}
    )
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms={**goal, **proprioception},
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms={**goal, **{k: replace(v, noise=None) for k, v in proprioception.items()}},
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    # The G1's own per-joint scales, not a flat number. An action means "this far from the
    # default pose", and the per-joint scales differ from a flat 0.5 by up to 7x. A policy
    # trained against one convention does something else under the other
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=G1_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  # Air time and slip both need to know when a foot is on the floor. Copied from the
  # jump's config so the two tasks agree on what a foot contact is. track_air_time is what
  # makes feet_air_time readable at all
  feet_ground_cfg = ContactSensorCfg(
    name=FEET_CONTACT,
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

  feet = SceneEntityCfg("robot", body_names=FOOT_BODIES)
  rewards: dict[str, RewardTermCfg] = {
    # The objective. Weighted to dominate: everything else exists to make it reachable,
    # not to compete with it
    "arrival": RewardTermCfg(
      func=mdp.arrival, weight=8.0, params={"command_name": COMMAND, "sharpness": 3.0}
    ),
    # The gradient for the first half of a window, nothing more. Well below arrival on
    # purpose: loitering near the target must never pay better than matching it
    "approach": RewardTermCfg(
      func=mdp.approach,
      weight=1.0,
      params={"command_name": COMMAND, "pos_std": 1.5, "ori_std": 1.0},
    ),
    # Small and positive, so no step of any episode is worth nothing at all. See
    # mdp/rewards.py on why every arrival term here is positive
    "alive": RewardTermCfg(func=base_mdp.is_alive, weight=0.5),
    ##
    # What the motion looks like, as opposed to where it ends up.
    #
    # The arrival terms do not care how the robot gets there, and left alone PPO found a
    # high-frequency hop: it covers ground, it does not fall, nothing objected. A warm
    # start does not fix that, because hopping is what the objective pays for. These
    # terms make it cost something.
    ##
    # Second difference of the action, where action_rate is the first. This is the one
    # that sees a vibration: a joint target oscillating step to step has a small rate and
    # a huge acceleration. Weighted well under action_rate because the quantity it squares
    # is several times larger for the same motion
    "action_acc": RewardTermCfg(func=base_mdp.action_acc_l2, weight=-2e-3),
    # Pays for a foot spending a normal step's worth of time off the ground. A hop is many
    # contacts of a few tens of milliseconds, under threshold_min, so it scores nothing: a
    # walking gait can collect this and a hopping one cannot. command_name is None because
    # the velocity tasks gate this on being told to move and a bridge is always going
    # somewhere.
    #
    # The only term here with no precedent in this repository, since every robot's velocity
    # config zeroes it. It carries the smallest weight of the three and is the first thing
    # to turn off if the robot marches on the spot to farm it. It is also the only positive
    # one, which is what makes that failure available
    "air_time": RewardTermCfg(
      func=velocity_mdp.feet_air_time,
      weight=0.1,
      params={
        "sensor_name": FEET_CONTACT,
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": None,
      },
    ),
    # The legs counter-rotating until the knees face each other. Threshold measured against
    # the dataset, not guessed: see mdp/rewards.py. Zero inside everything walk and run do,
    # and inside all but the top fraction of a percent of the jump
    "knees_inward": RewardTermCfg(
      func=mdp.knees_inward,
      weight=-5.0,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "left_joint": "left_hip_yaw_joint",
        "right_joint": "right_hip_yaw_joint",
        "threshold": 0.4,
      },
    ),
    # A foot on the ground moving sideways. Skating covers ground without paying to pick a
    # foot up, and is the other half of what an unphysical gait looks like
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": FEET_CONTACT,
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
      },
    ),
    # Back to what the locomotion tasks use, after a detour worth recording. It was cut to
    # -0.01 on a 150-iteration comparison where both weights reached the same arrival score
    # and the smaller one survived more windows. 150 iterations is long before a gait
    # artifact appears, and this is the term that prices one. The runs that followed hopped
    "action_rate": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-1e-1),
    "joint_limits": RewardTermCfg(
      func=base_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    # Falling or walking away costs something outright, on top of the steps it forfeits.
    # is_terminated excludes the time-out, so reaching a deadline never pays it
    "failed": RewardTermCfg(func=base_mdp.is_terminated, weight=-50.0),
    # Should stay at zero. Floor penetration is the defect that killed the kinematic
    # approach, and a term pinned at zero in the log is the evidence the simulator fixed it
    "feet_below_ground": RewardTermCfg(
      func=mdp.feet_below_ground, weight=-50.0, params={"asset_cfg": feet}
    ),
  }

  terminations: dict[str, TerminationTermCfg] = {
    "deadline_reached": TerminationTermCfg(
      func=mdp.deadline_reached, params={"command_name": COMMAND}, time_out=True
    ),
    "strayed": TerminationTermCfg(
      func=mdp.strayed, params={"command_name": COMMAND, "margin": 1.5}
    ),
    "fell_over": TerminationTermCfg(
      func=mdp.fell_over,
      params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.7},
    ),
  }

  commands: dict[str, CommandTermCfg] = {COMMAND: command}

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      entities={"robot": get_g1_robot_cfg()},
      sensors=(feet_ground_cfg,),
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    # Empty on purpose. Every reset event mjlab normally runs places the robot, and the
    # command term places it instead from a dataset row. Two writers of the root state in
    # one reset is a race decided by manager ordering
    events={},
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="pelvis",
      distance=3.0,
      elevation=-10.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=35, njmax=250, mujoco=MujocoCfg(timestep=0.005, iterations=10)
    ),
    decimation=4,
    # Only a backstop, and only while training. deadline_reached ends every window well
    # inside this. Play mode wants no backstop: the global episode timer would cut a viewer
    # off mid-window, and tests/test_task_configs.py requires play configs to be unbounded
    episode_length_s=1.0e9 if play else MAX_DEADLINE_S,
  )
