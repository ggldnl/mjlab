"""Table tennis racket constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# MJCF and assets.
##

RACKET_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "objects" / "racket" / "xmls" / "racket.xml"
)
assert RACKET_XML.exists()


def get_spec() -> mujoco.MjSpec:
  object = mujoco.MjSpec.from_file(str(RACKET_XML))
  return object


def get_racket_object_cfg() -> EntityCfg:
  """Get a table tennis racket configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when the
  config is shared across multiple places.
  """
  return EntityCfg(
    spec_fn=get_spec,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  object = Entity(get_racket_object_cfg())

  # Add a ground plane and light to the world body.
  spec = object.spec
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
