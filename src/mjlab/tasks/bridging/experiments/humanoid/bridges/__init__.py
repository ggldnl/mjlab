"""Bridge policies available to runtime tools."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from mjlab.tasks.bridging.experiments.humanoid.bridges.diffusion.runtime import (
  DiffusionRuntime,
)
from mjlab.tasks.bridging.experiments.humanoid.bridges.interface import (
  Bridge,
  BridgeCommand,
)

ActionMixer = Callable[[torch.Tensor, torch.Tensor, BridgeCommand], torch.Tensor]


def residual_action(
  base: torch.Tensor, residual: torch.Tensor, command: BridgeCommand
) -> torch.Tensor:
  """Apply the residual bridge only in its trained terminal region."""
  remaining = (command.window_steps - command.step).float() / command.fps
  active = (
    (command.target_errors() <= command.tolerances * 3.0).all(dim=-1)
    & (remaining <= 0.20)
    & (remaining > 0.0)
  )
  return base + active[:, None] * 0.25 * residual.clamp(-1.0, 1.0)


@dataclass(frozen=True)
class BridgeSpec:
  name: str
  task: str
  runtime: type[Bridge] | None = None
  base: str | None = None
  mix: ActionMixer | None = None

  @property
  def learned(self) -> bool:
    return self.runtime is None


BRIDGES = {
  "no-op": BridgeSpec("no-op", "Mjlab-G1-Imitation-Bridge", Bridge),
  "imitation": BridgeSpec("imitation", "Mjlab-G1-Imitation-Bridge"),
  "cvae": BridgeSpec("cvae", "Mjlab-G1-CVAE-Bridge"),
  "diffusion": BridgeSpec("diffusion", "Mjlab-G1-Imitation-Bridge", DiffusionRuntime),
  "goal-cvae": BridgeSpec("goal-cvae", "Mjlab-G1-Goal-CVAE-Bridge"),
  "residual": BridgeSpec(
    "residual",
    "Mjlab-G1-Residual-Bridge",
    base="imitation",
    mix=residual_action,
  ),
}
