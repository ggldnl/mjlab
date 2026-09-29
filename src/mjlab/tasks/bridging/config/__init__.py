"""Robot-specific skill registries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import mujoco

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  T1_ACTION_SCALE,
  get_g1_robot_cfg,
  get_t1_robot_cfg,
)
from mjlab.asset_zoo.robots.booster_t1.t1_constants import get_spec as get_t1_spec
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec as get_g1_spec
from mjlab.entity import EntityCfg
from mjlab.tasks.bridging.config.g1 import SKILLS as G1_SKILLS
from mjlab.tasks.bridging.config.t1 import SKILLS as T1_SKILLS

RobotAlias = Literal["g1", "t1"]
RobotAssetName = Literal["unitree_g1", "booster_t1"]


@dataclass(frozen=True)
class RobotCfg:
  """Robot assets and task ids used by shared bridging tools."""

  skills: dict[str, str]
  get_spec: Callable[[], mujoco.MjSpec]
  robot_name: RobotAssetName
  get_entity_cfg: Callable[[], EntityCfg]
  action_scale: dict[str, float]
  base_body_name: str
  foot_geom_names: tuple[str, ...]
  babel_dataset: str


ROBOTS = {
  "g1": RobotCfg(
    G1_SKILLS,
    get_g1_spec,
    "unitree_g1",
    get_g1_robot_cfg,
    G1_ACTION_SCALE,
    "pelvis",
    (r"^(left|right)_foot[1-7]_collision$",),
    "unitree_g1_locomotion_v1",
  ),
  "t1": RobotCfg(
    T1_SKILLS,
    get_t1_spec,
    "booster_t1",
    get_t1_robot_cfg,
    T1_ACTION_SCALE,
    "Trunk",
    (r"^(left|right)_foot_sphere.*link$",),
    "booster_t1",
  ),
}


def get_robot(name: str) -> RobotCfg:
  """Return a robot config by its command-line name."""
  try:
    return ROBOTS[name]
  except KeyError:
    raise ValueError(
      f"Unknown robot '{name}'. Available: {', '.join(ROBOTS)}"
    ) from None
