"""
Curriculum: how difficulty ramps over training (e.g. expanding velocity ranges at step thresholds)
"""

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.velocity.mdp.curriculums import commands_vel


# Curriculum stage thresholds.
# These are common_step_counter values, not iteration numbers.
# Measure _STEPS_PER_ITER from your logs and adjust accordingly.
# At ~50k steps/iter: S1=iter10, S2=iter30, S3=iter60, S4=iter100.
_S0 = 0
_S1 = 500    # Phase 1 → 2: robot should be moving, tighten contact penalty
_S2 = 1500   # Phase 2 → 3: introduce posture once locomotion exists
_S3 = 3000   # Phase 3 → 4: gait quality after posture is stable
_S4 = 5000   # Phase 4: final polish, velocity ceiling

curriculum = {

  # Velocity expands in three steps. Each step is ~1.4x the previous ceiling
  # so the policy always has a comfortable margin above what it mastered.
  "command_vel": CurriculumTermCfg(
    func=commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": _S0, "lin_vel_x": (-0.18, 0.18), "lin_vel_y": (-0.18, 0.18), "ang_vel_z": (-0.10, 0.10)},
        {"step": _S1, "lin_vel_x": (-0.25, 0.25), "lin_vel_y": (-0.25, 0.25), "ang_vel_z": (-0.15, 0.15)},
        {"step": _S3, "lin_vel_x": (-0.40, 0.40), "lin_vel_y": (-0.40, 0.40), "ang_vel_z": (-0.30, 0.30)},
        {"step": _S4, "lin_vel_x": (-0.50, 0.50), "lin_vel_y": (-0.50, 0.50), "ang_vel_z": (-0.40, 0.40)},
      ],
    },
  ),
}