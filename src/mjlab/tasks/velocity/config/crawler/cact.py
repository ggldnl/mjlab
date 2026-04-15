"""
Commands, Actions, Curriculum, Terminations
Four tightly related things that define the agent's interface with the task:
- Commands: what goals are sampled and sent to the policy each step
    (e.g. target velocity [vx, vy, ωz])
- Actions: how the policy output maps onto the robot
    (e.g. joint position targets with a scale factor)
- Curriculum: how difficulty ramps over training
    (e.g. expanding velocity ranges at step thresholds)
- Terminations: episode-ending conditions
    (timeout, robot fell over)

These four belong together because they all answer the question:
what is the task contract between environment and policy?
"""

import math
from typing import Dict

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import bad_orientation, time_out
from mjlab.managers import CommandTermCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg, reward_weight
from mjlab.tasks.velocity.mdp.curriculums import commands_vel

from experiments.robots.crawler.actuators import CRAWLER_ACTION_SCALES


commands: Dict[str, CommandTermCfg] = {
  "twist": UniformVelocityCommandCfg(
    entity_name="robot",
    rel_standing_envs=0.1,
    rel_heading_envs=0.3,
    heading_command=True,
    heading_control_stiffness=0.5,
    debug_vis=True,
    resampling_time_range=(0.5, 5.0),
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-0.8, 0.8),
      lin_vel_y=(-0.8, 0.8),
      ang_vel_z=(-0.5, 0.5),
      heading=(-math.pi, math.pi),
    ),
  ),
}

actions: dict[str, ActionTermCfg] = {
  "joint_pos": JointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=CRAWLER_ACTION_SCALES,
    use_default_offset=True,
  ),
}


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

  # Phase 2: non-foot contact penalty turns on once the robot is moving.
  # Introduced alone so its effect is clearly observable in logs.
  "w_nonfeet_ground_contact": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "nonfeet_ground_contact",
      "weight_stages": [
        {"step": _S0, "weight":  0.0},
        {"step": _S1, "weight": -0.5},
        {"step": _S2, "weight": -1.0},
        {"step": _S3, "weight": -2.0},
      ],
    },
  ),

  # Phase 3: posture pair introduced together.
  # Upright and base_height are the same conceptual signal so there is no
  # reason to stagger them — they ramp in lockstep.
  "w_upright": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "upright",
      "weight_stages": [
        {"step": _S0, "weight": 0.0},
        {"step": _S2, "weight": 0.5},
        {"step": _S3, "weight": 1.0},
      ],
    },
  ),

  "w_base_height": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "base_height",
      "weight_stages": [
        {"step": _S0, "weight": 0.0},
        {"step": _S2, "weight": 0.5},
        {"step": _S3, "weight": 1.0},
      ],
    },
  ),

  # Foot slip introduced alongside posture: both concern contact quality.
  "w_foot_slip": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "foot_slip",
      "weight_stages": [
        {"step": _S0, "weight":  0.0},
        {"step": _S2, "weight": -0.2},
        {"step": _S3, "weight": -0.5},
      ],
    },
  ),

  # Phase 4: dynamic stability only after posture has had time to settle.
  # Introduced alone at S3 so it does not compete with the posture ramp.
  "w_base_stability": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "base_stability",
      "weight_stages": [
        {"step": _S0, "weight":  0.0},
        {"step": _S3, "weight": -0.3},
        {"step": _S4, "weight": -0.6},
      ],
    },
  ),

  # Action rate ramps only after posture is solid. Early on -0.02 (set in
  # rewards.py directly) is enough to suppress noise without blocking exploration.
  "w_action_rate_l2": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "action_rate_l2",
      "weight_stages": [
        {"step": _S0, "weight": -0.02},
        {"step": _S3, "weight": -0.10},
        {"step": _S4, "weight": -0.20},
      ],
    },
  ),

  # Joint velocity: last to appear, kept small.
  # At 12 joints × (2 rad/s)² = 48 raw L2, -0.003 gives -0.14/step.
  # Only increase this after verifying it does not suppress the gait.
  "w_joint_vel_l2": CurriculumTermCfg(
    func=reward_weight,
    params={
      "reward_name": "joint_vel_l2",
      "weight_stages": [
        {"step": _S0, "weight":  0.000},
        {"step": _S4, "weight": -0.003},
      ],
    },
  ),
}


# Terminations: only the two that are unambiguous.
# All custom terminations (stand_still, poor_tracking) have been removed.
# They caused premature episode deaths during exploration and their job
# is now done structurally by phase_contact + is_terminated.
terminations = {
  "time_out": TerminationTermCfg(
    func=time_out,
    time_out=True,
  ),
  "fell_over": TerminationTermCfg(
    func=bad_orientation,
    params={"limit_angle": math.radians(90.0)},
  ),
}