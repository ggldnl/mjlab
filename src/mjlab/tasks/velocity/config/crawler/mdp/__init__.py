"""MDP terms for the crawler velocity task.

Re-exports the shared velocity MDP plus the crawler-specific abstraction and the
thin helpers that wire abstraction signals/references into the reward and
observation managers.
"""

from mjlab.tasks.velocity.mdp import *  # noqa: F401, F403

from .abstractions import (
  TrotGaitAbstraction as TrotGaitAbstraction,
)
from .abstractions import (
  TrotGaitAbstractionCfg as TrotGaitAbstractionCfg,
)
from .observations import abstraction_obs as abstraction_obs
from .rewards import abstraction_signal as abstraction_signal
