"""Differential drive robot constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# MJCF and assets.
##

DIFFDRIVE_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "diffdrive" / "xmls" / "diffdrive.xml"
)
assert DIFFDRIVE_XML.exists()


def get_spec() -> mujoco.MjSpec:
  robot = mujoco.MjSpec.from_file(str(DIFFDRIVE_XML))
  robot.delete(robot.geom("floor"))
  robot.delete(robot.light("light"))
  return robot


##
# Actuator config.
##

DIFFDRIVE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlActuatorCfg(
      target_names_expr=(
        "left_wheel",
        "right_wheel",
      )
    ),
  ),
)

##
# Keyframe config.
##

DIFFDRIVE_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={"left_wheel": 0.0, "right_wheel": 0.0},
  joint_vel={".*": 0.0},
)

##
# Final config.
##


def get_diffdrive_robot_cfg() -> EntityCfg:
  """Get a fresh diffdrive robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when the
  config is shared across multiple places.
  """
  return EntityCfg(
    spec_fn=get_spec,
    articulation=DIFFDRIVE_ARTICULATION,
    init_state=DIFFDRIVE_INIT,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_diffdrive_robot_cfg())

  # Add a ground plane and light to the world body.
  spec = robot.spec
  spec.worldbody.add_geom(
    name="ground",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0, 0, 0.025],
    rgba=[0.8, 0.8, 0.8, 1.0],
    contype=1,
    conaffinity=1,
  )
  spec.worldbody.add_light(
    name="main_light",
    pos=[0, 0, 3],
    dir=[0, 0, -1],
  )

  model = spec.compile()
  viewer.launch(model)
