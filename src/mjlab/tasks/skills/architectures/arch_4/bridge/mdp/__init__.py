"""The MDP terms specific to bridging: the window command, its reward and its endings.

Everything generic (proprioception, action penalties, the standard time-out) comes from
`mjlab.envs.mdp` and is not re-exported here.
"""

from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.commands import (
  BridgeCommand as BridgeCommand,
)
from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.commands import (
  BridgeCommandCfg as BridgeCommandCfg,
)
from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.commands import (
  SpliceCfg as SpliceCfg,
)
from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.rewards import (
  approach as approach,
)
from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.rewards import (
  feet_below_ground as feet_below_ground,
)
from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.rewards import (
  joint_pos_error_exp as joint_pos_error_exp,
)
from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.rewards import (
  joint_vel_error_exp as joint_vel_error_exp,
)
from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.rewards import (
  resumed as resumed,
)
from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.rewards import (
  root_ang_vel_error_exp as root_ang_vel_error_exp,
)
from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.rewards import (
  root_lin_vel_error_exp as root_lin_vel_error_exp,
)
from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.rewards import (
  root_ori_error_exp as root_ori_error_exp,
)
from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.rewards import (
  root_pos_error_exp as root_pos_error_exp,
)
from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.terminations import (
  fell_over as fell_over,
)
from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.terminations import (
  lost_tracking as lost_tracking,
)
from mjlab.tasks.skills.architectures.arch_4.bridge.mdp.terminations import (
  window_done as window_done,
)
