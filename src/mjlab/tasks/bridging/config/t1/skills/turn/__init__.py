"""Conditioned T1 turns in place.

Train: uv run train Mjlab-T1-Turn --agent.run-name sokoban_turn
Play:  uv run play Mjlab-T1-Turn
Panel: --viewer viser, Commands > Twist
Manual heading: world heading in degrees and turn speed in rad/s
Manual heading off: random targets every 20 to 24 seconds
Orange arrow: target heading; cyan arrow: current heading
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.bridging.config.t1.skills.turn.command import (
  TurnCommandCfg,
  heading_reward,
  turn_command,
  turn_rate_penalty,
  turn_rate_reward,
)
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.t1.env_cfgs import booster_t1_flat_env_cfg
from mjlab.tasks.velocity.config.t1.rl_cfg import booster_t1_ppo_runner_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

TURN_TASK_ID = "Mjlab-T1-Turn"


def make_turn_env_cfg(
  cfg: ManagerBasedRlEnvCfg, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Add heading conditioned turns to a robot's flat velocity task"""
  cfg.scene.num_envs = 1 if play else 4096
  if not play:
    cfg.episode_length_s = 24.0
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  cfg.commands["twist"] = TurnCommandCfg(
    entity_name="robot",
    resampling_time_range=(20.0, 24.0),
    rel_standing_envs=0.1,
    ranges=UniformVelocityCommandCfg.Ranges((0.0, 0.0), (0.0, 0.0), (-0.5, 0.5)),
    viz=twist.viz,
    gui=play,
    debug_vis=play,
  )
  for group in cfg.observations.values():
    group.terms["turn_angle"] = ObservationTermCfg(func=turn_command)
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=turn_rate_reward, weight=4.0, params={"std": 0.1}
  )
  cfg.rewards["turn_heading"] = RewardTermCfg(
    func=heading_reward, weight=2.0, params={"std": 0.25}
  )
  cfg.rewards["turn_rate_error"] = RewardTermCfg(func=turn_rate_penalty, weight=-2.0)
  cfg.rewards["pose"].params["walking_threshold"] = 0.02
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.15}
  for pattern in cfg.rewards["pose"].params["std_walking"]:
    if "hip_yaw" in pattern.lower():
      cfg.rewards["pose"].params["std_walking"][pattern] = 0.3
  cfg.rewards["air_time"].weight = 0.5
  cfg.rewards["air_time"].params["command_threshold"] = 0.02
  cfg.rewards["air_time"].params["threshold_max"] = 0.35
  for name in ("foot_clearance", "foot_swing_height"):
    cfg.rewards[name].params["target_height"] = 0.06
    cfg.rewards[name].params["command_threshold"] = 0.02
  cfg.rewards["foot_slip"].weight = -0.5
  cfg.rewards["foot_slip"].params["command_name"] = None
  cfg.curriculum.pop("command_vel", None)
  return cfg


def t1_turn_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_turn_env_cfg(booster_t1_flat_env_cfg(play=play), play)


def t1_turn_ppo_runner_cfg():
  cfg = replace(booster_t1_ppo_runner_cfg(), experiment_name="t1_turn")
  assert cfg.actor.distribution_cfg is not None
  cfg.actor.distribution_cfg["init_std"] = 0.5
  return cfg


register_mjlab_task(
  task_id=TURN_TASK_ID,
  env_cfg=t1_turn_env_cfg(),
  play_env_cfg=t1_turn_env_cfg(play=True),
  rl_cfg=t1_turn_ppo_runner_cfg(),
)
