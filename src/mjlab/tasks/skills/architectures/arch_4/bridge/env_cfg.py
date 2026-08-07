"""The bridging task as an ordinary mjlab RL environment.

Being ordinary is the point. The previous attempt at this component was a supervised model
that produced motion; it scored well and produced motion no body could perform, putting a
foot through the floor in two windows out of five and hovering where the recording jumped.
Nothing in a kinematic objective knows that ground reaction force is what separates a jump
from a hover. So the bridge is now a policy driving a robot through a simulator, and the
things that were wrong are no longer expressible.

    uv run python -m mjlab.tasks.skills.architectures.arch_4.bridge.dataset --view False
    uv run train Mjlab-Bridge-Flat-G1 --env.scene.num-envs 4096
    uv run play Mjlab-Bridge-Flat-G1

The observation is proprioception plus the command, and the command is the target strip
and the phase. The reference inside the hole reaches the reward and never the observation,
which is what makes this a bridge rather than a tracker.
"""

from __future__ import annotations

from dataclasses import replace

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
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.skills.architectures.arch_4.bridge import mdp
from mjlab.tasks.skills.architectures.arch_4.bridge.dataset import CorpusCfg, WindowCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def bridge_env_cfg(play: bool = False, split: str = "train") -> ManagerBasedRlEnvCfg:
  """The bridging environment. `split` picks which subjects the windows come from."""
  command = mdp.BridgeCommandCfg(
    entity_name="robot",
    corpus=CorpusCfg(),
    windows=WindowCfg(),
    split=split,
    # Windows are redrawn only when an episode ends, never mid-episode: a hole that
    # changed halfway through would be two different questions scored as one.
    resampling_time_range=(1.0e9, 1.0e9),
    start_noise=0.0 if play else 1.0,
  )

  proprioception = {
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
  # The whole instruction: where to arrive, and how much of the hole is left. What is
  # absent is the reference inside the hole.
  goal = {
    "command": ObservationTermCfg(
      func=base_mdp.generated_commands, params={"command_name": "bridge"}
    )
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms={**goal, **proprioception},
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms={
        **goal,
        **{k: replace(v, noise=None) for k, v in proprioception.items()},
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    # The same scale the parkour arena uses, and not a round number of my choosing. An
    # action means "this far from the default pose", and if the two environments disagree
    # about how far, a policy trained here does something else there. The per-joint
    # scales differ from a flat 0.5 by up to a factor of seven, which is not a detail
    # that shows up as slightly worse behaviour.
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=G1_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  feet = SceneEntityCfg("robot", body_names=FOOT_BODIES)
  rewards: dict[str, RewardTermCfg] = {
    "joint_pos": RewardTermCfg(
      func=mdp.joint_pos_error_exp,
      weight=1.0,
      params={"command_name": "bridge", "std": 0.3},
    ),
    "joint_vel": RewardTermCfg(
      func=mdp.joint_vel_error_exp,
      weight=0.5,
      params={"command_name": "bridge", "std": 3.0},
    ),
    "root_pos": RewardTermCfg(
      func=mdp.root_pos_error_exp,
      weight=1.0,
      params={"command_name": "bridge", "std": 0.3},
    ),
    "root_ori": RewardTermCfg(
      func=mdp.root_ori_error_exp,
      weight=0.5,
      params={"command_name": "bridge", "std": 0.4},
    ),
    "root_lin_vel": RewardTermCfg(
      func=mdp.root_lin_vel_error_exp,
      weight=1.0,
      params={"command_name": "bridge", "std": 1.0},
    ),
    "root_ang_vel": RewardTermCfg(
      func=mdp.root_ang_vel_error_exp,
      weight=0.5,
      params={"command_name": "bridge", "std": 3.14},
    ),
    "action_rate": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-1e-1),
    "joint_limits": RewardTermCfg(
      func=base_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    # Should stay at zero. It is here because floor penetration is what condemned the
    # approach this replaced, and a term pinned at zero in the log is the evidence.
    "feet_below_ground": RewardTermCfg(
      func=mdp.feet_below_ground, weight=-50.0, params={"asset_cfg": feet}
    ),
  }

  terminations: dict[str, TerminationTermCfg] = {
    "hole_closed": TerminationTermCfg(
      func=mdp.hole_closed, params={"command_name": "bridge"}, time_out=True
    ),
    "lost_tracking": TerminationTermCfg(
      func=mdp.lost_tracking, params={"command_name": "bridge", "threshold": 0.6}
    ),
    "fell_over": TerminationTermCfg(
      func=mdp.fell_over,
      params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.7},
    ),
  }

  commands: dict[str, CommandTermCfg] = {"bridge": command}

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      entities={"robot": get_g1_robot_cfg()},
    ),
    observations=observations,
    actions=actions,
    commands=commands,
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
    # The longest hole plus its tail, at the control rate. Nothing should ever reach this:
    # `hole_closed` ends an episode first, and this is only the backstop.
    episode_length_s=(WindowCfg().gap_range[1] + 10) / 50.0 + 0.5,
  )
