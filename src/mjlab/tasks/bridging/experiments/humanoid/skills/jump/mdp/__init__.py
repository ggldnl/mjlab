"""MDP terms for the parkour jump.

The tracking task's terms come in first and the jump-specific ones are layered on
top, so a config can reach everything through this one namespace. The jump command
exposes the same properties as mjlab's `MotionCommand`, which is what lets the
tracking reward and termination terms be reused verbatim.
"""

from mjlab.tasks.tracking.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F403
from .observations import *  # noqa: F403
from .rewards import *  # noqa: F403
from .terminations import *  # noqa: F403
