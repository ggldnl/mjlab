"""What the teacher reads that the student does not.

    keyframes(masked=False)   every interior constraint of the window. Teacher and critic
    keyframes(masked=True)    a random subset of the same ones. Student
    dense_reference           the recorded state for this very tick. Teacher and critic

Two terms and one flag, so the two groups cannot describe the window differently by
accident. A checkpoint is tied to the width and order of its group, and the student's
group has to be the teacher's with information removed rather than a second description
of the same thing.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.bridging.experiments.humanoid.bridges.distillation.mdp.commands import (
  MaskedBridgeCommand,
)


def _command(env: ManagerBasedRlEnv, command_name: str) -> MaskedBridgeCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, MaskedBridgeCommand)
  return term


def keyframes(
  env: ManagerBasedRlEnv, command_name: str, masked: bool = True
) -> torch.Tensor:
  """The interior of the window, in full or through this window's mask.

  Fixed width either way. Masking zeroes channels and clears their bits, it never changes
  the shape: an MLP reads one vector and a slot has to sit in the same place whether or not
  it is readable this window.
  """
  return _command(env, command_name).constraints(masked=masked)


def dense_reference(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """The recorded crossing at this tick, and whether there is one.

  The privileged half of the teacher's observation, and the reason phase one converges: a
  policy told the next frame of a motion is a tracker, and this project's trackers train.
  Never in the student's group, which is the whole point of distilling.
  """
  return _command(env, command_name).dense_reference()
