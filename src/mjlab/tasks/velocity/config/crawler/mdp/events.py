"""
Defines stochastic events that fire at specific moments: startup (once at initialization),
reset (every episode), or interval (randomly during an episode). This is where to configure:
- Episode resets (base pose, joint positions)
- Perturbations (random pushes)
- Domain randomization (friction, encoder bias, CoM offsets)
Keeping this separate from rewards/observations lets us swap DR profiles independently
(e.g. a light-DR version for debugging, a heavy-DR version for sim-to-real transfer).
"""

from mjlab.envs.mdp.dr import body_com_offset, geom_friction
from mjlab.envs.mdp.events import (
  push_by_setting_velocity,
  reset_joints_by_offset,
  reset_root_state_uniform,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab.asset_zoo.robots.crawler.crawler_constants import CRAWLER_BASE_NAME, CRAWLER_FOOT_GEOM_NAMES
from mjlab.asset_zoo.robots.crawler.actuators import SMALLEST_ABS_LIM, MG90S_VELOCITY_LIMIT


events = {

  "reset_base": EventTermCfg(
    func=reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.001, 0.01),
        "yaw": (-3.14, 3.14),
      },
      "velocity_range": {},
    },
  ),

  "reset_robot_joints": EventTermCfg(
    func=reset_joints_by_offset,
    mode="reset",
    params={
        # Small random offsets around default joint positions
        # Chosen to be safe across all joints (10% of smallest limits)
        "position_range": (- 0.1 * SMALLEST_ABS_LIM, 0.1 * SMALLEST_ABS_LIM),

        # Small initial joint velocities to avoid perfectly static starts (10% of max velocity)
        "velocity_range": (-0.1 * MG90S_VELOCITY_LIMIT, 0.1 * MG90S_VELOCITY_LIMIT),

        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    }
  ),

  "push_robot": EventTermCfg(
    func=push_by_setting_velocity,
    mode="interval",
    interval_range_s=(8.0, 15.0),
    params={
      "velocity_range": {
        "x": (-0.3, 0.3),
        "y": (-0.3, 0.3),
        "z": (-0.1, 0.1),
        "roll": (-0.3, 0.3),
        "pitch": (-0.3, 0.3),
        "yaw": (-0.5, 0.5),
      },
    },
  ),

  # Domain randomization

  "foot_friction": EventTermCfg(
    mode="reset",
    func=geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=CRAWLER_FOOT_GEOM_NAMES),
      "operation": "abs",
      "ranges": (0.3, 1.2),
      "shared_random": True,  # All foot geoms share the same friction
    },
  ),

  "base_com": EventTermCfg(
    mode="reset",
    func=body_com_offset,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=CRAWLER_BASE_NAME),
      "operation": "add",
      "ranges": {
          0: (-0.015, 0.015),
          1: (-0.015, 0.015),
          2: (-0.01,  0.01),
      },
    },
  ),
}