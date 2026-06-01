"""
Terminations: episode-ending conditions (timeout, robot fell over, ...)
"""

import math

from mjlab.envs.mdp.terminations import bad_orientation, time_out
from mjlab.managers.termination_manager import TerminationTermCfg

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
    params={"limit_angle": math.radians(70.0)},
  ),
}
