"""
Curriculum: how difficulty ramps over training (e.g. expanding velocity ranges at step thresholds)
"""

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.velocity.mdp.curriculums import commands_vel

_STEPS_PER_ITER = 50_000

# Curriculum stage thresholds (common_step_counter values, not iteration numbers).
_S0 = 0
_S1 = 50 * _STEPS_PER_ITER   # Phase 1 -> 2: robot is moving, introduce gaait structure rewards
_S2 = 150 * _STEPS_PER_ITER  # Phase 2 -> 3: gait exists, introduce posture and expand velocity
_S3 = 300 * _STEPS_PER_ITER  # Phase 3 -> 4: posture stable, introduce polish and expand velocity
_S4 = 500 * _STEPS_PER_ITER  # Phase 4: final ceiling

curriculum = {

  "command_vel": CurriculumTermCfg(
    func=commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": _S0, "lin_vel_x": (-0.18, 0.18), "lin_vel_y": (-0.18, 0.18), "ang_vel_z": (-0.10, 0.10)},
        {"step": _S1, "lin_vel_x": (-0.25, 0.25), "lin_vel_y": (-0.25, 0.25), "ang_vel_z": (-0.15, 0.15)},
        {"step": _S2, "lin_vel_x": (-0.35, 0.35), "lin_vel_y": (-0.35, 0.35), "ang_vel_z": (-0.25, 0.25)},
        {"step": _S3, "lin_vel_x": (-0.40, 0.40), "lin_vel_y": (-0.40, 0.40), "ang_vel_z": (-0.30, 0.30)},
        {"step": _S4, "lin_vel_x": (-0.50, 0.50), "lin_vel_y": (-0.50, 0.50), "ang_vel_z": (-0.40, 0.40)},
      ],
    },
  ),
}