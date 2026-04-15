"""
Defines observation groups, typically one for the actor (noisy, sim-to-real-safe sensors only)
and one for the critic (privileged information that's only available in sim, like true linear
velocity or ground truth foot contact forces). The critic group usually starts with the
actor_terms and adds privileged terms on top.
"""

from mjlab.envs.mdp.observations import (
  builtin_sensor,
  generated_commands,
  joint_pos_rel,
  joint_vel_rel,
  last_action,
  projected_gravity,
  height_scan,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.velocity.mdp.observations import (
  foot_air_time,
  foot_contact,
  foot_contact_forces,
  foot_height,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise
import torch
import math

from experiments.robots.crawler.constants import CRAWLER_FOOT_GEOM_NAMES
from experiments.robots.crawler.sensors import TERRAIN_SCAN


# At 100 Hz / decimation 4 = 25 Hz policy. 10 frames = 400 ms,
# enough to observe roughly one full stride cycle.
_HISTORY = 10

# Diagonal-pair trot offsets: legs 0 and 2 are in phase, legs 1 and 3 are anti-phase.
# To use a different gait (walk, bound), change these offsets only — nothing else.
_LEG_PHASE_OFFSETS = torch.tensor([0.0, math.pi, 0.0, math.pi])


def gait_phase_clock(
  env: ManagerBasedRlEnv,
  frequency: float = 1.5,
) -> torch.Tensor:
  # Initialize lazily
  if not hasattr(env, "_phase_clock"):
    env._phase_clock = torch.zeros(env.num_envs, device=env.device)

  env._phase_clock[env.episode_length_buf <= 1] = 0.0

  dt = env.physics_dt * env.cfg.decimation
  env._phase_clock += 2.0 * math.pi * frequency * dt

  offsets = _LEG_PHASE_OFFSETS.to(env.device)
  phases = env._phase_clock.unsqueeze(1) + offsets  # [B, 4]

  # Sin and cos together give a smooth, non-discontinuous phase signal.
  # The policy sees both where in the cycle each leg is AND the rate of change.
  return torch.cat([torch.sin(phases), torch.cos(phases)], dim=1)  # [B, 8]


# Actor: sim-to-real-safe sensors only, with noise.
actor_terms = {
  "base_lin_vel": ObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "imu_lin_vel"},
    noise=Unoise(n_min=-0.5, n_max=0.5),
    history_length=_HISTORY,
  ),
  "base_ang_vel": ObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "imu_ang_vel"},
    noise=Unoise(n_min=-0.2, n_max=0.2),
    history_length=_HISTORY,
  ),
  # Gravity projection: slow signal, single frame is sufficient
  "projected_gravity": ObservationTermCfg(
    func=projected_gravity,
    noise=Unoise(n_min=-0.05, n_max=0.05),
  ),
  # Joint history implicitly encodes leg phase
  "joint_pos": ObservationTermCfg(
    func=joint_pos_rel,
    noise=Unoise(n_min=-0.01, n_max=0.01),
    history_length=_HISTORY,
  ),
  "joint_vel": ObservationTermCfg(
    func=joint_vel_rel,
    noise=Unoise(n_min=-1.5, n_max=1.5),
    history_length=_HISTORY,
  ),
  # Action history lets the policy detect and correct its own oscillations
  "actions": ObservationTermCfg(
    func=last_action,
    history_length=_HISTORY,
  ),
  # Command is constant within a resampling window — no history needed
  "command": ObservationTermCfg(
    func=generated_commands,
    params={"command_name": "twist"},
  ),
  # Absolute phase reference the policy cannot infer from proprioception.
  # Joint positions tell the policy where the legs are; the clock tells it
  # where they should be going. No history: it is already a temporal signal.
  "gait_phase": ObservationTermCfg(
    func=gait_phase_clock,
    params={"frequency": 1.5},
  ),
  # Terrain: slow signal, no history to avoid inflating input size
  "height_scan": ObservationTermCfg(
    func=height_scan,
    params={"sensor_name": "terrain_scan"},
    noise=Unoise(n_min=-0.1, n_max=0.1),
    scale=1 / TERRAIN_SCAN.max_distance,
  ),
}

# Critic: everything the actor sees + clean ground truth + privileged contact info.
# No noise, no history needed since the critic sees exact state.
critic_terms = {
  **actor_terms,
  "true_base_lin_vel": ObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "imu_lin_vel"},
  ),
  "true_base_ang_vel": ObservationTermCfg(
    func=builtin_sensor,
    params={"sensor_name": "imu_ang_vel"},
  ),
  "true_joint_pos": ObservationTermCfg(
    func=joint_pos_rel,
  ),
  "true_joint_vel": ObservationTermCfg(
    func=joint_vel_rel,
  ),
  "true_height_scan": ObservationTermCfg(
    func=height_scan,
    params={"sensor_name": "terrain_scan"},
    scale=1 / TERRAIN_SCAN.max_distance,
  ),
  "feet_contact": ObservationTermCfg(
    func=foot_contact,
    params={"sensor_name": "feet_ground_contact"},
  ),
  "feet_air_time": ObservationTermCfg(
    func=foot_air_time,
    params={"sensor_name": "feet_ground_contact"},
  ),
  "feet_contact_forces": ObservationTermCfg(
    func=foot_contact_forces,
    params={"sensor_name": "feet_ground_contact"},
  ),
  "feet_height": ObservationTermCfg(
    func=foot_height,
    params={"asset_cfg": SceneEntityCfg("robot", geom_names=CRAWLER_FOOT_GEOM_NAMES)},
  ),
}

observations = {
  "actor": ObservationGroupCfg(
    terms=actor_terms,
    concatenate_terms=True,
    enable_corruption=True,
  ),
  "critic": ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
  ),
}