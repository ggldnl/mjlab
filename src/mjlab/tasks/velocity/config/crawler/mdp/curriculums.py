"""
Curriculum: how difficulty ramps over training (e.g. expanding velocity ranges at step thresholds)
"""

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.velocity.mdp.curriculums import commands_vel
from mjlab.tasks.velocity.mdp import reward_weight


_HORIZON = 24

# Curriculum stage thresholds (common_step_counter values, not iteration numbers).
_S0 = 0
_S1 = 50 * _HORIZON   # Phase 1 -> 2: robot is moving, introduce gait structure rewards
_S2 = 100 * _HORIZON  # Phase 2 -> 3: gait exists, introduce posture and expand velocity
_S3 = 150 * _HORIZON  # Phase 3 -> 4: posture stable, introduce polish and expand velocity
_S4 = 200 * _HORIZON  # Phase 4: final ceiling

curriculum = {

  "command_vel": CurriculumTermCfg(
    func=commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": _S0, "lin_vel_x": (-0.10, 0.10), "lin_vel_y": (-0.10, 0.10), "ang_vel_z": (-0.10, 0.10)},
        {"step": _S1, "lin_vel_x": (-0.25, 0.25), "lin_vel_y": (-0.25, 0.25), "ang_vel_z": (-0.15, 0.15)},
        {"step": _S2, "lin_vel_x": (-0.35, 0.35), "lin_vel_y": (-0.35, 0.35), "ang_vel_z": (-0.25, 0.25)},
        {"step": _S3, "lin_vel_x": (-0.40, 0.40), "lin_vel_y": (-0.40, 0.40), "ang_vel_z": (-0.30, 0.30)},
        {"step": _S4, "lin_vel_x": (-0.50, 0.50), "lin_vel_y": (-0.50, 0.50), "ang_vel_z": (-0.40, 0.40)},
      ],
    },
  ),

  # Stability

  "w_upright": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "upright",
      "weight_stages": [
        {"step": _S0, "weight": 0.2},
        {"step": _S2, "weight": 0.5},
        {"step": _S3, "weight": 1.0},
      ],
    },
  ),

  "w_base_stability": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "base_stability",
      "weight_stages": [
        {"step": _S0, "weight": -0.1},
        {"step": _S3, "weight": -0.5},
        {"step": _S4, "weight": -1.0},
      ],
    },
  ),

  "w_base_height": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "base_height",
      "weight_stages": [
        {"step": _S0, "weight": 0.1},
        {"step": _S2, "weight": 0.5},
        {"step": _S3, "weight": 1.0},
      ],
    },
  ),

  "w_action_rate_l2": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "action_rate_l2",
      "weight_stages": [
        {"step": _S0, "weight": -0.05},
        {"step": _S3, "weight": -0.10},
        {"step": _S4, "weight": -0.20},
      ],
    },
  ),

  # Foot

  "w_foot_slip": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "foot_slip",
      "weight_stages": [
        {"step": _S0, "weight": -0.1},
        {"step": _S2, "weight": -0.2},
        {"step": _S3, "weight": -0.5},
      ],
    },
  ),

  "w_foot_swing_height": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "foot_swing_height",
      "weight_stages": [
        {"step": _S0, "weight": -0.1},
        {"step": _S2, "weight": -0.2},
        {"step": _S3, "weight": -0.5},
      ],
    },
  ),
}