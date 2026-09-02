"""The bridge's MDP terms: the window (commands), what it pays (rewards), how it ends
(terminations).

ROOT_STATE_DIM is re-exported from the dataset: a state's layout is part of this vocabulary
everywhere it is read.
"""

from mjlab.tasks.bridging.experiments.humanoid.bridge.datasets.dataset import (
  ROOT_STATE_DIM as ROOT_STATE_DIM,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.commands import (
  CHANNELS as CHANNELS,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.commands import (
  BridgeCommand as BridgeCommand,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.commands import (
  BridgeCommandCfg as BridgeCommandCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.commands import (
  Tolerances as Tolerances,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.commands import (
  arm_mask as arm_mask,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.commands import (
  arrival_score as arrival_score,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.commands import (
  arrived as arrived,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.commands import (
  channel_errors as channel_errors,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.rewards import (
  approach as approach,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.rewards import (
  arrival as arrival,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.rewards import (
  feet_below_ground as feet_below_ground,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.rewards import (
  feet_chatter as feet_chatter,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.rewards import (
  feet_slip as feet_slip,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.rewards import (
  guidance as guidance,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.rewards import (
  knees_inward as knees_inward,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.terminations import (
  deadline_reached as deadline_reached,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.terminations import (
  fell_over as fell_over,
)
from mjlab.tasks.bridging.experiments.humanoid.bridge.mdp.terminations import (
  strayed as strayed,
)
