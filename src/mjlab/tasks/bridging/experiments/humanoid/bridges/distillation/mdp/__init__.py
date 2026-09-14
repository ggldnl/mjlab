"""Distillation bridge MDP terms.

    commands.py       the window, with its interior exposed and maskable
    observations.py   the two views of that interior

Rewards and terminations are the imitation bridge's, re-exported rather than rewritten. The
two architectures answer the same question and are meant to be comparable on it, which only
holds if the same code decides what an arrival is and when a window ends.
"""

from mjlab.tasks.bridging.experiments.humanoid.bridges.dataset.dataset import (
  ROOT_STATE_DIM as ROOT_STATE_DIM,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.distillation.mdp.commands import (
  MaskedBridgeCommand as MaskedBridgeCommand,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.distillation.mdp.commands import (
  MaskedBridgeCommandCfg as MaskedBridgeCommandCfg,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.distillation.mdp.observations import (
  dense_reference as dense_reference,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.distillation.mdp.observations import (
  keyframes as keyframes,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp import (
  CHANNELS as CHANNELS,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp import (
  Tolerances as Tolerances,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp import (
  arrival as arrival,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp import (
  arrival_score as arrival_score,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp import (
  arrived as arrived,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp import (
  channel_errors as channel_errors,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp import (
  feet_below_ground as feet_below_ground,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp import (
  feet_chatter as feet_chatter,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp import (
  feet_slip as feet_slip,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp import (
  fell_over as fell_over,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp import (
  guidance as guidance,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp import (
  out_of_patience as out_of_patience,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.imitation.mdp import (
  strayed as strayed,
)
