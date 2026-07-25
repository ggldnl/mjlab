"""Car-like (Ackermann) robot constants.

A four-wheeled car: the two front wheels steer (a position-servo knuckle each) and
the two rear wheels are driven (a velocity servo each).
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# MJCF and assets.
##

CARLIKE_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "carlike" / "xmls" / "carlike.xml"
)
assert CARLIKE_XML.exists()

STEER_JOINTS = ("front_left_steer", "front_right_steer")
DRIVE_JOINTS = ("rear_left_wheel", "rear_right_wheel")


def get_spec() -> mujoco.MjSpec:
  robot = mujoco.MjSpec.from_file(str(CARLIKE_XML))
  robot.delete(robot.geom("floor"))
  robot.delete(robot.light("light"))
  # The env drives everything through mjlab's own PD actuators (below), which add
  # their own <motor> transmissions. Drop the XML's standalone actuators so they are
  # not duplicated; the standalone .xml keeps them for direct use.
  for name in (*STEER_JOINTS, *DRIVE_JOINTS):
    robot.delete(robot.actuator(name))
  return robot


##
# Actuator config.
##

# Front-wheel steering: a position servo. Stiffness holds the commanded steer angle
# against the scrub torque the tires feed back; the effort limit caps how hard it can
# hold. The action commands a steer angle [rad].
STEER_STIFFNESS = 8.0
STEER_DAMPING = 0.3
STEER_EFFORT_LIMIT = 4.0

# Rear-wheel drive: a velocity servo (stiffness 0, so torque = damping * (vel_target
# - vel), clamped to +-effort_limit). The action commands a target wheel angular
# velocity [rad/s]; the effort limit is the torque ceiling that ramps the car up to
# speed and bounds its traction.
DRIVE_STIFFNESS = 0.0
DRIVE_DAMPING = 0.4
DRIVE_EFFORT_LIMIT = 1.5

CARLIKE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    IdealPdActuatorCfg(
      target_names_expr=STEER_JOINTS,
      stiffness=STEER_STIFFNESS,
      damping=STEER_DAMPING,
      effort_limit=STEER_EFFORT_LIMIT,
    ),
    IdealPdActuatorCfg(
      target_names_expr=DRIVE_JOINTS,
      stiffness=DRIVE_STIFFNESS,
      damping=DRIVE_DAMPING,
      effort_limit=DRIVE_EFFORT_LIMIT,
    ),
  ),
)

##
# Keyframe config.
##

CARLIKE_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

##
# Final config.
##


def get_carlike_robot_cfg() -> EntityCfg:
  """Get a fresh car-like robot configuration instance."""
  return EntityCfg(
    spec_fn=get_spec,
    articulation=CARLIKE_ARTICULATION,
    init_state=CARLIKE_INIT,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_carlike_robot_cfg())
  spec = robot.spec
  spec.worldbody.add_geom(
    name="ground",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0, 0, 0.025],
    rgba=[0.8, 0.8, 0.8, 1.0],
    contype=1,
    conaffinity=1,
  )
  spec.worldbody.add_light(name="main_light", pos=[0, 0, 3], dir=[0, 0, -1])
  viewer.launch(spec.compile())
