"""Cartpole constants."""

import math
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# MJCF and assets.
##

CARTPOLE_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "cartpole" / "xmls" / "cartpole.xml"
)
assert CARTPOLE_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(CARTPOLE_XML))


##
# Actuator config.
##

CARTPOLE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XmlActuatorCfg(target_names_expr=("slider",)),),
)

##
# Keyframe config.
##

CARTPOLE_BALANCE_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={"slider": 0.0, "hinge_1": 0.0},
  joint_vel={".*": 0.0},
)

CARTPOLE_SWINGUP_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={"slider": 0.0, "hinge_1": math.pi},
  joint_vel={".*": 0.0},
)

##
# Final config.
##


def get_cartpole_robot_cfg(swing_up: bool = False) -> EntityCfg:
  """Get a fresh cartpole robot configuration instance.

  Args:
    swing_up: If True, initialize the pole hanging down (swing-up task) instead of
      upright (balance task).

  Returns a new EntityCfg instance each time to avoid mutation issues when the
  config is shared across multiple places.
  """
  return EntityCfg(
    spec_fn=get_spec,
    articulation=CARTPOLE_ARTICULATION,
    init_state=CARTPOLE_SWINGUP_INIT if swing_up else CARTPOLE_BALANCE_INIT,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_cartpole_robot_cfg())

  viewer.launch(robot.spec.compile())
