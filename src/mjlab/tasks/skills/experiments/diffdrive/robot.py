"""The differential-drive robot: the real diffdrive.xml plus its controls.

This class attaches the MuJoCo model into a world, reads the reduced state
[x, y, theta, v, omega] back from MjData (state_from_mjdata), and turns a
target-twist command (v*, omega*) into wheel torques via a velocity servo
(twist_to_torque). The robot is actuated (no fake joints and teleporting)
so residual speed must be shed through the wheels.

    uv run python -m mjlab.tasks.skills.experiments.diffdrive.robot
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
import numpy.typing as npt

from mjlab.tasks.skills.config.diffdrive.diffdrive_env_cfg import get_spec
from mjlab.tasks.skills.interfaces import Command, Robot, State

# State indices s = [x, y, theta, v, omega].
X, Y, THETA, V, OMEGA = 0, 1, 2, 3, 4
STATE = (X, Y, THETA, V, OMEGA)
STATE_DIM = 5
ACTION_DIM = (
  2  # analytic action u = [v_dot, omega_dot]: bounded linear / angular accel.
)

# Physical constants of diffdrive.xml (kept here so the model and its maths agree).
WHEEL_RADIUS = 0.06  # wheel cylinder radius (m).
TRACK = 0.22  # lateral wheel separation (m): the two wheels sit at y = +/-0.11.
BASE_HEIGHT = 0.06  # height of the base above the floor (m).


@dataclass
class DiffDrive(Robot):
  """The diff-drive robot, posed and driven through a compiled MuJoCo model.

  kv is the wheel-velocity servo gain; the torque it produces is bounded by the
  actuators' ctrlrange (set in the XML), which is what makes acceleration bounded.
  """

  kv: float = 3.0
  z: float = BASE_HEIGHT
  _qadr: int = field(init=False, default=-1)  # freejoint qpos start
  _vadr: int = field(init=False, default=-1)  # freejoint qvel (dof) start
  _ldof: int = field(init=False, default=-1)  # left-wheel dof
  _rdof: int = field(init=False, default=-1)  # right-wheel dof

  def spec(self) -> mujoco.MjSpec:
    """The robot's own spec, with its floor and light dropped (the world provides both)."""
    return get_spec()

  def attach_to(self, world: mujoco.MjSpec) -> None:
    world.attach(self.spec(), prefix="robot/", frame=world.worldbody.add_frame())

  def bind(self, model: mujoco.MjModel) -> None:
    self._qadr = int(model.joint("robot/root").qposadr[0])
    self._vadr = int(model.joint("robot/root").dofadr[0])
    self._ldof = int(model.joint("robot/left_wheel").dofadr[0])
    self._rdof = int(model.joint("robot/right_wheel").dofadr[0])

  def sense(self, model: mujoco.MjModel, data: mujoco.MjData) -> State:
    return state_from_mjdata(model, data, prefix="robot/")

  def actuate(
    self, model: mujoco.MjModel, data: mujoco.MjData, command: Command
  ) -> np.ndarray:
    return twist_to_torque(model, data, command, kv=self.kv, prefix="robot/")

  def reset(self, model: mujoco.MjModel, data: mujoco.MjData, state: State) -> None:
    """
    Initializes the MuJoCo state so that the robot starts at a desired reduced state: [x, y, omega, v, w].

    """
    s = np.asarray(state, float)
    x, y, theta, v, omega = (float(s[i]) for i in (X, Y, THETA, V, OMEGA))
    q, dof = self._qadr, self._vadr

    # Free joint position
    # Position (x, y, z) + quaternion (qw, qx, qy, qz)
    # A rotation around the z-axis by angle theta has quaternion:
    # q = [cos(theta/2), 0, 0, sin(theta/2)]
    data.qpos[q : q + 7] = (
      x,
      y,
      self.z,
      math.cos(theta / 2),
      0.0,
      0.0,
      math.sin(theta / 2),
    )

    # Free joint velocity
    data.qvel[dof : dof + 6] = (
      v * math.cos(theta),
      v * math.sin(theta),
      0.0,
      0.0,
      0.0,
      omega,
    )

    # Start the wheels already rolling
    left, right = wheel_speeds((v, omega))
    data.qvel[self._ldof] = left
    data.qvel[self._rdof] = right


# MuJoCo bridge: reduced state <-> the real diff-drive model.


def wheel_speeds(twist: npt.ArrayLike) -> tuple[float, float]:
  """Target wheel angular velocities (left, right) realizing body twist (v, omega)."""
  v, omega = float(np.asarray(twist)[0]), float(np.asarray(twist)[1])
  left = (v - omega * TRACK / 2.0) / WHEEL_RADIUS
  right = (v + omega * TRACK / 2.0) / WHEEL_RADIUS
  return left, right


def state_from_mjdata(
  model: mujoco.MjModel, data: mujoco.MjData, *, prefix: str = "robot/"
) -> np.ndarray:
  """Read the reduced state [x, y, theta, v, omega] of the base from MjData."""
  base = model.body(prefix + "base").id
  x, y = float(data.xpos[base][0]), float(data.xpos[base][1])
  qw, qx, qy, qz = data.xquat[base]
  theta = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
  vel = np.zeros(6)  # [angular(3), linear(3)] in the world frame
  mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base, vel, 0)
  v = float(vel[3] * np.cos(theta) + vel[4] * np.sin(theta))  # speed along heading
  omega = float(vel[2])  # world yaw rate
  return np.array([x, y, theta, v, omega], float)


def twist_to_torque(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  twist: npt.ArrayLike,
  *,
  kv: float = 2.0,
  prefix: str = "robot/",
) -> np.ndarray:
  """Wheel-velocity servo: motor torques tracking the target body twist (v*, omega*).

  The twist sets target wheel speeds (wheel_speeds); each motor torque is
  proportional to its wheel-speed error and clipped to the actuator's ctrlrange.
  That clip is the bounded-acceleration crux: finite torque means wheel, hence body,
  acceleration is bounded, so residual speed cannot be canceled instantly.
  Returns a full (nu,) ctrl vector.
  """
  left_star, right_star = wheel_speeds(twist)
  ldof = model.joint(prefix + "left_wheel").dofadr[0]
  rdof = model.joint(prefix + "right_wheel").dofadr[0]
  lact = model.actuator(prefix + "left_wheel").id
  ract = model.actuator(prefix + "right_wheel").id
  ctrl = np.zeros(model.nu)
  ctrl[lact] = kv * (left_star - data.qvel[ldof])
  ctrl[ract] = kv * (right_star - data.qvel[rdof])
  lo, hi = model.actuator_ctrlrange[[lact, ract]].T
  ctrl[[lact, ract]] = np.clip(ctrl[[lact, ract]], lo, hi)
  return ctrl


def main() -> None:
  """Drive the robot through a few constant twists on a bare floor (a physics check)."""
  from dataclasses import dataclass as _dataclass

  import tyro

  from mjlab.tasks.skills import play

  @_dataclass
  class Args:
    v: float = 1.0  # target forward speed (m/s)
    omega: float = 0.0  # target yaw rate (rad/s)

  args = tyro.cli(Args)
  robot = DiffDrive()
  spec = mujoco.MjSpec()
  spec.worldbody.add_light(pos=[0, 0, 4])
  floor = spec.worldbody.add_geom()
  floor.type = mujoco.mjtGeom.mjGEOM_PLANE
  floor.size = np.array([5.0, 5.0, 0.1])
  robot.attach_to(spec)
  model = spec.compile()
  robot.bind(model)

  twist = np.array([args.v, args.omega])
  start = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

  def policy(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return robot.actuate(model, data, twist)

  def on_reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    robot.reset(model, data, start)

  play.run(model, policy, decimation=4, on_reset=on_reset)


if __name__ == "__main__":
  main()
