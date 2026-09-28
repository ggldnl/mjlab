"""Robot-specific skill registries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco

from mjlab.asset_zoo.robots.booster_t1.t1_constants import get_spec as get_t1_spec
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec as get_g1_spec
from mjlab.tasks.bridging.config.g1 import SKILLS as G1_SKILLS
from mjlab.tasks.bridging.config.t1 import SKILLS as T1_SKILLS


@dataclass(frozen=True)
class RobotCfg:
  """Robot assets and task ids used by shared bridging tools."""

  skills: dict[str, str]
  get_spec: Callable[[], mujoco.MjSpec]


ROBOTS = {
  "g1": RobotCfg(G1_SKILLS, get_g1_spec),
  "t1": RobotCfg(T1_SKILLS, get_t1_spec),
}


def get_robot(name: str) -> RobotCfg:
  """Return a robot config by its command-line name."""
  try:
    return ROBOTS[name]
  except KeyError:
    raise ValueError(
      f"Unknown robot '{name}'. Available: {', '.join(ROBOTS)}"
    ) from None
