"""
Commands: what goals are sampled and sent to the policy each step (e.g. target velocity [vx, vy, ωz])
"""

import math
from typing import Dict

from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.managers import CommandTermCfg


commands: Dict[str, CommandTermCfg] = {
  "twist": UniformVelocityCommandCfg(
    entity_name="robot",
    rel_standing_envs=0.1,
    rel_heading_envs=0.3,
    heading_command=True,
    heading_control_stiffness=0.5,
    debug_vis=True,
    resampling_time_range=(2.0, 6.0),
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-0.35, 0.35),
      lin_vel_y=(-0.35, 0.35),
      ang_vel_z=(-0.25, 0.25),
      heading=(-math.pi, math.pi),
    ),
  ),
}
