"""The bridging task as an ordinary mjlab RL environment.

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.dataset.dataset
    uv run train Mjlab-G1-Bridge --env.scene.num-envs 4096
    uv run play Mjlab-G1-Bridge

Being ordinary is most of the point. A bridge is a policy driving a robot through a
simulator, and the failures that condemned every attempt before it -- a foot through the
floor, a body hovering where the recording jumped, a pose averaged out of two
incompatible ways of crossing -- are not expressible here. Nothing in a kinematic
objective knows that ground reaction force is what separates a jump from a hover.

##
# The two observation groups, and the one thing that is different about them
##

Proprioception is the usual set, with one addition: `base_lin_vel` is in the actor's
group and not only the critic's. That is a deliberate break with how the locomotion tasks
in this repository are set up, where linear velocity is privileged because it cannot be
measured on hardware.

The reason is that velocity is not a detail of this task, it is half of it. A bridge is
asked to arrive carrying a particular momentum, and the previous version came out of
training with a root linear velocity error of 0.41 against a standing robot's 0.56 --
barely better than not moving, and on one run actually worse. A policy told what velocity
to arrive with, and not told what velocity it currently has, is being asked to close a gap
it cannot see. Estimating it from a history of accelerations is a solvable problem and a
different one; it can be handed back to a student later, when there is a teacher worth
distilling.

What is *not* in either group is anything about the middle of the window. There is no
reference to observe because there is no reference, and that is what makes this a bridge
rather than a tracker.
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
from mjlab.tasks.bridging.experiments.humanoid.bridge.dataset.dataset import (
  DEFAULT_BANK,
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
"""The longest window the episode has to accommodate, in seconds.

Longer than `deadline_range` allows, because a pair whose velocity change or joint travel
does not fit the deadline it drew has the deadline stretched rather than the pair thrown
away, and a stretched window can run to about three seconds. The episode length is only a
backstop; `deadline_reached` is what actually ends a window."""


def bridge_env_cfg(
  play: bool = False,
  split: str = "train",
  bank_path: Path = DEFAULT_BANK,
  leaving: tuple[str, ...] | None = None,
  entering: tuple[str, ...] | None = None,
) -> ManagerBasedRlEnvCfg:
  """The bridging environment.

  `leaving` and `entering` name which skills a start and a target may be drawn from, or
  None for any skill in the bank. Restricting them is how a single pair is studied; the
  default is every pair the bank can form, because one bridge for the whole pool is the
  thing being built and training it on one couple would be building something else.
  """
  command = mdp.BridgeCommandCfg(
    entity_name="robot",
    bank_path=bank_path,
    split=split,
    leaving=leaving,
    entering=entering,
    # A window is drawn on reset and never mid-episode. One that changed halfway through
    # would be two different questions scored as one.
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
    # default pose", and the per-joint scales differ from a flat 0.5 by up to a factor of
    # seven; a policy trained against one convention does something else under the other.
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=G1_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  # Air time and slip both need to know when a foot is on the floor. Copied from the
  # jump's config rather than invented, so the two tasks agree about what a foot contact
  # is; `track_air_time` is what makes `feet_air_time` readable at all.
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
    # The objective. Weighted to dominate, because everything else here exists to make it
    # reachable rather than to compete with it.
    "arrival": RewardTermCfg(
      func=mdp.arrival, weight=8.0, params={"command_name": COMMAND, "sharpness": 3.0}
    ),
    # The gradient for the first half of a window, and nothing more than that. Weighted
    # well below `arrival` on purpose: a bridge that finds it can collect more by loitering
    # near the target than by matching it has been handed the wrong task.
    "approach": RewardTermCfg(
      func=mdp.approach,
      weight=1.0,
      params={"command_name": COMMAND, "pos_std": 1.5, "ori_std": 1.0},
    ),
    # Small, positive, and mostly there to make sure no step of any episode is worth
    # nothing at all. See mdp/rewards.py on why every arrival term here is positive.
    "alive": RewardTermCfg(func=base_mdp.is_alive, weight=0.5),
    ##
    # What the motion has to look like, as opposed to where it has to end up.
    #
    # None of these say anything about arriving. They exist because the arrival terms are
    # indifferent to how the robot gets there, and left to itself PPO found a
    # high-frequency hop: it covers ground, it does not fall, and nothing objected. A
    # warm start does not fix that -- hopping is what the objective was paying for, so
    # training rediscovers it wherever it starts from. These are what make it cost
    # something.
    ##
    # The second difference of the action, where `action_rate` is the first. It is the one
    # that actually sees a vibration: a joint target oscillating step to step has a small
    # rate and an enormous acceleration, which is exactly the signature being hunted. The
    # weight is well under `action_rate`'s because the quantity it squares is several
    # times larger for the same motion.
    "action_acc": RewardTermCfg(func=base_mdp.action_acc_l2, weight=-2e-3),
    # Pays for a foot spending an ordinary step's worth of time off the ground. A hop is
    # many contacts of a few tens of milliseconds, which falls under `threshold_min` and
    # scores nothing, so this is a reward a hopping gait cannot collect and a walking one
    # can. `command_name` is None: the velocity tasks gate this on being told to move, and
    # a bridge is always being asked to go somewhere.
    #
    # The one term here with no precedent in this repository -- every robot's velocity
    # config sets its weight to zero -- so it carries the smallest weight of the three and
    # is the first thing to turn off if the robot starts marching on the spot to farm it.
    # It is also the only positive one, which is what makes that failure available at all.
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
    # The legs counter-rotating until the knees face each other. Measured against the bank
    # rather than guessed: see mdp/rewards.py. Zero inside everything walk and run do, and
    # inside all but the top fraction of a percent of the jump.
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
    # A foot on the ground moving sideways. Skating covers ground without the cost of
    # picking a foot up, and it is the other half of what an unphysical gait looks like.
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": FEET_CONTACT,
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
      },
    ),
    # Back to what the locomotion tasks use, after a detour worth recording. It was cut to
    # -0.01 on the evidence of a 150-iteration comparison in which the two weights reached
    # the same arrival score and the smaller one survived more windows. That comparison was
    # too short to see what it was measuring: 150 iterations is long before a gait artifact
    # appears, and this is the term that prices one. The runs that followed hopped.
    "action_rate": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-1e-1),
    "joint_limits": RewardTermCfg(
      func=base_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    # Falling or walking away costs something outright, on top of the steps it forfeits.
    # `is_terminated` excludes the time-out, so running a window to its deadline never
    # pays it.
    "failed": RewardTermCfg(func=base_mdp.is_terminated, weight=-50.0),
    # Should stay at zero. It is here because floor penetration is the defect that
    # condemned the kinematic approach, and a term pinned at zero in the log is the
    # evidence that moving into a simulator fixed it.
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
    # Empty on purpose. Every reset event mjlab would normally run places the robot, and
    # the command term places it instead, from a bank row. Two things writing the root
    # state in one reset is a race whose winner depends on manager ordering.
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
    # Only a backstop, and only while training. `deadline_reached` ends every window well
    # inside this, so nothing should ever reach it. Play mode wants no backstop at all:
    # the global episode timer would cut a viewer off mid-window, and `tests/
    # test_task_configs.py` requires every play config to be effectively unbounded.
    episode_length_s=1.0e9 if play else MAX_DEADLINE_S,
  )
