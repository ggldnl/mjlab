"""Diffdrive environment configuration."""

from pathlib import Path

import mujoco

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

_DIFFDRIVE_XML = Path(__file__).parent / "diffdrive.xml"


def get_spec() -> mujoco.MjSpec:
  robot = mujoco.MjSpec.from_file(str(_DIFFDRIVE_XML))
  robot.delete(robot.geom("floor"))
  robot.delete(robot.light("light"))
  return robot


_DIFFDRIVE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlActuatorCfg(
      target_names_expr=(
        "left_wheel",
        "right_wheel",
      )
    ),
  ),
)

_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={"left_wheel": 0.0, "right_wheel": 0.0},
  joint_vel={".*": 0.0},
)


def get_diffdrive_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_spec, articulation=_DIFFDRIVE_ARTICULATION, init_state=_INIT
  )
