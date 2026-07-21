"""KUKA iiwa14 constants.

The XML (standard MuJoCo Menagerie ``kuka_iiwa_14``) already defines tuned
``<general>`` position-servo actuators for all 7 joints, so ``XmlActuatorCfg`` wraps
them as-is rather than re-specifying gains.

An ``attachment_site`` sits at the tip of ``link7``. This is where we can attach
something to the end-effector.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# MJCF and assets.
##

KUKA_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "kuka_iiwa_14" / "xmls" / "iiwa14.xml"
)
assert KUKA_XML.exists()


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(KUKA_XML))
  # The XML ships its own "home" keyframe; drop it so only KUKA_HOME_KEYFRAME
  # (below) applies, matching every other asset_zoo robot.
  for key in list(spec.keys):
    spec.delete(key)
  return spec


##
# Actuator config.
##

KUKA_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))

KUKA_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XmlActuatorCfg(target_names_expr=KUKA_JOINT_NAMES),),
)

##
# Keyframe config.
##

# Matches the XML's own <keyframe name="home"> qpos.
KUKA_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  joint_pos=dict(
    zip(KUKA_JOINT_NAMES, (0.0, 0.785398, 0.0, -1.5708, 0.0, 0.0, 0.0), strict=True)
  ),
  joint_vel={".*": 0.0},
)

##
# Final config.
##


def get_kuka_robot_cfg() -> EntityCfg:
  """Get a fresh KUKA iiwa14 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when the
  config is shared across multiple places.
  """
  return EntityCfg(
    spec_fn=get_spec,
    articulation=KUKA_ARTICULATION,
    init_state=KUKA_HOME_KEYFRAME,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_kuka_robot_cfg())
  model = robot.spec.compile()

  # The XML's own <option integrator="implicitfast"/> is lost during
  # Entity/MjSpec.attach() (attach() does not propagate <option> fields), so the
  # model would otherwise silently fall back to the default Euler integrator.
  # The arm's actuators (kp=2000, kd=200) are stiff enough that Euler is
  # numerically unstable at this timestep -- joints blow up into high-frequency
  # noise around 0 within milliseconds. A real training env must set this via
  # SimulationCfg(mujoco=MujocoCfg(integrator=...)) instead; this is only for
  # this standalone viewer.
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

  viewer.launch(model)
