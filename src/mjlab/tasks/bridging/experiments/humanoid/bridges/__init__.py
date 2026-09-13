"""The bridge architectures. One sub-package each, and the registry that picks between them.

    imitation      PPO against tracker rollouts, guidance annealed to zero
    diffusion      not implemented
    distillation   not implemented

Every architecture answers the same question, "get the robot from this dynamic state to that
one", and differs only in how its policy is produced. They read one corpus (dataset/) and
each registers its own task id, so their checkpoints live in separate log directories and
two of them can be compared on the same transition.

Selecting one

Every transition script and the parkour demo takes --bridge, and the name is all they need:
the spec below carries the task id to load a policy from, the experiment name to find the
newest checkpoint under, and the env config the staging arena is built on.

    uv run python -m ...tests.transitions.walk2jump --bridge imitation
    uv run python -m ...demos.parkour.run --bridge imitation

Adding one

1. Write bridges/<name>/, registering a task id the way imitation does
2. Export BRIDGE = BridgeSpec(...) from its __init__
3. Add the name to BridgeKind below

A stub leaves env_cfg None, which is what makes it selectable but not runnable: `resolve`
refuses it with a message naming the package to write, instead of failing somewhere deeper
with a missing task id.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Literal, get_args

from mjlab.envs import ManagerBasedRlEnvCfg

BridgeKind = Literal["imitation", "diffusion", "distillation"]
"""Which architecture to use. tyro turns this into the --bridge choices."""

DEFAULT_BRIDGE: BridgeKind = "imitation"
"""The only one with a trained checkpoint."""

BRIDGE_KINDS: tuple[str, ...] = get_args(BridgeKind)


@dataclass(frozen=True)
class BridgeSpec:
  """What a script needs to know about one architecture, without importing its internals."""

  kind: str
  task_id: str
  """Registered task, for loading the policy and its observation group."""
  experiment: str
  """Log directory under logs/rsl_rl, for finding the newest checkpoint."""
  env_cfg: Callable[..., ManagerBasedRlEnvCfg] | None = None
  """Builds the task config. None marks an architecture that has no code yet."""


def resolve(kind: BridgeKind | str) -> BridgeSpec:
  """The spec for one architecture, or a message saying it has not been written.

  Imported on demand rather than at module load, so a stub costs nothing and a broken
  architecture under development does not take every other script down with it.
  """
  if kind not in BRIDGE_KINDS:
    raise SystemExit(f"Unknown bridge '{kind}'. Known: {', '.join(BRIDGE_KINDS)}.")
  spec = importlib.import_module(f"{__name__}.{kind}").BRIDGE
  assert isinstance(spec, BridgeSpec)
  if spec.env_cfg is None:
    raise SystemExit(
      f"The {kind} bridge is a stub with no task behind it yet. Write "
      f"bridges/{kind}/, or pass --bridge {DEFAULT_BRIDGE}."
    )
  return spec
