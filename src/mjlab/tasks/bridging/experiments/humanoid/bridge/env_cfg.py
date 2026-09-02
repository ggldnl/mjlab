"""The bridge as an ordinary mjlab RL environment.

    observations   actor and critic, same terms, differing only in noise
    actions        29 joint position targets, G1 per-joint action scale
    rewards        mdp/rewards.py
    terminations   mdp/terminations.py
    command        mdp/commands.py, one window per episode
    events         none. The command term places the robot; two writers would race

Run
---

    uv run python -m mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.tracker
    uv run train Mjlab-G1-Bridge --env.scene.num-envs 4096
    uv run play Mjlab-G1-Bridge

Being an ordinary RL task is the point. A bridge is a policy driving a robot through a
simulator, so the failures that killed the kinematic attempts cannot be expressed here at
all: a foot through the floor, a body hovering where the recording jumped, a pose averaged
out of two incompatible crossings.

Observations
------------

`base_lin_vel` is in the actor group, not only the critic's. That is not this repo's
convention: the locomotion tasks keep it privileged because hardware cannot measure it. Here
the bridge has to arrive carrying a particular momentum, so a policy told the target
velocity but not its own is closing a gap it cannot see. Measured 0.41 root velocity error
against a standing robot's 0.56. Estimating velocity from an acceleration history is a
separate problem, for a student distilled from this teacher later.

Neither group carries anything about the middle of the window. There is no reference to
observe, which is what makes this a bridge and not a tracker. `mdp.guidance` reads the
recorded crossing, but it is a reward, not an input.
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
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

COMMAND = "bridge"
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_SITES = ("left_foot", "right_foot")
FEET_CONTACT = "feet_ground_contact"

MAX_DEADLINE_S = 2.0
"""Longest window the episode has to fit, in seconds. A backstop that never fires.

`deadline_reached` is what ends a window, and no window outlives
`BridgeCommandCfg.duration_s_range`, so this sits above that range with room to spare."""


def bridge_env_cfg(
  play: bool = False,
  split: str = "train",
  dataset_path: Path = DEFAULT_DATASET,
  sources: tuple[str, ...] | None = None,
) -> ManagerBasedRlEnvCfg:
  """Build the bridging environment.

  Args:
    play: no observation noise, no start perturbation, no episode length cap.
    split: "train" or "eval". The split is by recording environment, not by frame.
    dataset_path: which corpus to draw windows from.
    sources: restrict windows to these skills or clips. None means any, which is the
      default: the goal is one bridge for the whole pool. Set it to study one skill.
  """
  command = mdp.BridgeCommandCfg(
    entity_name="robot",
    dataset_path=dataset_path,
    split=split,
    sources=sources,
    # A window is drawn on reset, never mid-episode. One that changed halfway through
    # would be two questions scored as one
    resampling_time_range=(1.0e9, 1.0e9),
    start_noise=0.0 if play else 1.0,
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

  # Slip needs to know when a foot is on the floor. Copied from the jump's config so the two
  # tasks agree on what a foot contact is.
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
    # A broad full-state gradient for the first half of a window. It uses the same channels
    # as arrival, so crouching and arm posture are never opposed by a locomotion prior.
    "approach": RewardTermCfg(
      func=mdp.approach, weight=1.0, params={"command_name": COMMAND}
    ),
    # The shaping, and the only thing this task has that the plain bridge does not. Dense,
    # on every step, and gone by `BridgeCommandCfg.guide_steps`, after which the two tasks
    # are the same.
    #
    # Two is a per-step rate against a per-step arrival that is near zero for most of a
    # window, so while it is at full weight this is the largest thing in the reward over the
    # stretch that has no other signal. That is what a term meant to lead a search has to
    # be worth, and it is also why it has to go away: at full weight for the whole run it
    # would be the objective
    "guidance": RewardTermCfg(
      func=mdp.guidance,
      weight=2.0,
      params={"command_name": COMMAND, "tolerance_scale": 4.0},
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
    # What replaced the positive air_time reward. That one kept the feet from chattering by
    # paying for a gait, which is the wrong thing to want from a policy whose target is
    # often a crouch or a planted stance. This charges only for a flight or a stance too
    # short to be a step, so a planted foot and a real step both cost nothing
    "feet_chatter": RewardTermCfg(
      func=mdp.feet_chatter,
      weight=-1.0,
      params={"sensor_name": FEET_CONTACT, "min_time": 0.2},
    ),
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
